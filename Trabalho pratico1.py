#################################################################################
import collections
import itertools
import sys
from heapq import heappush, heappop, heapify

import logging
import pickle
from pathlib import Path
from typing import Union, Any

_log = logging.getLogger(__name__)


class _EndOfFileSymbol:
    """
    Internal class for "end of file" symbol to be able
    to detect the end of the encoded bit stream,
    which does not necessarily align with byte boundaries.
    """

    def __repr__(self):
        return '_EOF'

    # Because _EOF will be compared with normal symbols (strings, bytes),
    # we have to provide a minimal set of comparison methods.
    # We'll make _EOF smaller than the rest (meaning lowest frequency)
    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __eq__(self, other):
        return other.__class__ == self.__class__

    def __hash__(self):
        return hash(self.__class__)


# Singleton-like "end of file" symbol
_EOF = _EndOfFileSymbol()


# TODO store/load code table from file
# TODO Directly encode to and decode from file

def _guess_concat(data):
    """
    Guess concat function from given data
    """
    return {
        type(u''): u''.join,
        type(b''): bytes,
    }.get(type(data), list)


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    assert path.is_dir()
    return path


class PrefixCodec:
    """
    Prefix code codec, using given code table.
    """

    def __init__(self, code_table, concat=list, check=True, eof=_EOF):
        """
        Initialize codec with given code table.

        :param code_table: mapping of symbol to code tuple (bitsize, value)
        :param concat: function to concatenate symbols
        :param check: whether to check the code table
        :param eof: "end of file" symbol (customizable for advanced usage)
        """
        # Code table is dictionary mapping symbol to (bitsize, value)
        self._table = code_table
        self._concat = concat
        self._eof = eof
        if check:
            assert isinstance(self._table, dict) and all(
                isinstance(b, int) and b >= 1 and isinstance(v, int) and v >= 0
                for (b, v) in self._table.values()
            )
            # TODO check if code table is actually a prefix code

    def get_code_table(self):
        """
        Get code table
        :return: dictionary mapping symbol to code tuple (bitsize, value)
        """
        return self._table

    
    def get_code_len(self):
        """
        Author: RPP, 2020.11.09
        Get code len
        :return: 2 lists: symbols and code length per symbol
        """
        t = self._table
        symbols = sorted(t.keys())  #symbols
        values = [t[s] for s in symbols] #list(t.values())[1:]
        lengths = [v[0] for v in values] #symbol lengths
        
        return symbols, lengths


    def print_code_table(self, out=sys.stdout):
        """
        Print code table overview
        """
        # TODO: add sort options?
        # Render table cells as string
        columns = list(zip(*itertools.chain(
            [('Bits', 'Code', 'Value', 'Symbol')],
            (
                (str(bits), bin(val)[2:].rjust(bits, '0'), str(val), repr(symbol))
                for symbol, (bits, val) in self._table.items()
            )
        )))
        # Find column widths and build row template
        widths = tuple(max(len(s) for s in col) for col in columns)
        template = '{0:>%d} {1:%d} {2:>%d} {3}\n' % widths[:3]
        for row in zip(*columns):
            out.write(template.format(*row))

    def encode(self, data):
        """
        Encode given data.

        :param data: sequence of symbols (e.g. byte string, unicode string, list, iterator)
        :return: byte string
        """
        return bytes(self.encode_streaming(data))

    def encode_streaming(self, data):
        """
        Encode given data in streaming fashion.

        :param data: sequence of symbols (e.g. byte string, unicode string, list, iterator)
        :return: generator of bytes (single character strings in Python2, ints in Python 3)
        """
        # Buffer value and size
        buffer = 0
        size = 0
        for s in data:
            # TODO: raise custom EncodeException instead of KeyError?
            b, v = self._table[s]
            # Shift new bits in the buffer
            buffer = (buffer << b) + v
            size += b
            while size >= 8:
                byte = buffer >> (size - 8)
                yield byte
                buffer = buffer - (byte << (size - 8))
                size -= 8

        # Handling of the final sub-byte chunk.
        # The end of the encoded bit stream does not align necessarily with byte boundaries,
        # so we need an "end of file" indicator symbol (_EOF) to guard against decoding
        # the non-data trailing bits of the last byte.
        # As an optimization however, while encoding _EOF, it is only necessary to encode up to
        # the end of the current byte and cut off there.
        # No new byte has to be started for the remainder, saving us one (or more) output bytes.
        if size > 0:
            b, v = self._table[self._eof]
            buffer = (buffer << b) + v
            size += b
            if size >= 8:
                byte = buffer >> (size - 8)
            else:
                byte = buffer << (8 - size)
            yield byte

    def decode(self, data, concat=None):
        """
        Decode given data.

        :param data: sequence of bytes (string, list or generator of bytes)
        :param concat: optional override of function to concatenate the decoded symbols
        :return:
        """
        return (concat or self._concat)(self.decode_streaming(data))

    def decode_streaming(self, data):
        """
        Decode given data in streaming fashion

        :param data: sequence of bytes (string, list or generator of bytes)
        :return: generator of symbols
        """
        # Reverse lookup table: map (bitsize, value) to symbols
        lookup = {(b, v): s for s, (b, v) in self._table.items()}

        buffer = 0
        size = 0
        for byte in data:
            for m in [128, 64, 32, 16, 8, 4, 2, 1]:
                buffer = (buffer << 1) + bool(byte & m)
                size += 1
                if (size, buffer) in lookup:
                    symbol = lookup[size, buffer]
                    if symbol == self._eof:
                        return
                    yield symbol
                    buffer = 0
                    size = 0

    def save(self, path: Union[str, Path], metadata: Any = None):
        """
        Persist the code table to a file.
        :param path: file path to persist to
        :param metadata: additional metadata
        :return:
        """
        code_table = self.get_code_table()
        data = {
            "code_table": code_table,
            "type": type(self),
            "concat": self._concat,
        }
        if metadata:
            data['metadata'] = metadata
        path = Path(path)
        ensure_dir(path.parent)
        with path.open(mode='wb') as f:
            # TODO also provide JSON option? Requires handling of _EOF and possibly other non-string code table keys.
            pickle.dump(data, file=f)
        _log.info('Saved {c} code table ({l} items) to {p!r}'.format(
            c=type(self).__name__, l=len(code_table), p=str(path)
        ))

    @staticmethod
    def load(path: Union[str, Path]) -> 'PrefixCodec':
        """
        Load a persisted PrefixCodec
        :param path: path to serialized PrefixCodec code table data.
        :return:
        """
        path = Path(path)
        with path.open(mode='rb') as f:
            data = pickle.load(f)
        cls = data['type']
        assert issubclass(cls, PrefixCodec)
        code_table = data['code_table']
        _log.info('Loading {c} with {l} code table items from {p!r}'.format(
            c=cls.__name__, l=len(code_table), p=str(path)
        ))
        return cls(code_table, concat=data['concat'])


class HuffmanCodec(PrefixCodec):
    """
    Huffman coder, with code table built from given symbol frequencies or raw data,
    providing encoding and decoding methods.
    """

    @classmethod
    def from_frequencies(cls, frequencies, concat=None, eof=_EOF):
        """
        Build Huffman code table from given symbol frequencies
        :param frequencies: symbol to frequency mapping
        :param concat: function to concatenate symbols
        :param eof: "end of file" symbol (customizable for advanced usage)
        """
        concat = concat or _guess_concat(next(iter(frequencies)))

        # Heap consists of tuples: (frequency, [list of tuples: (symbol, (bitsize, value))])
        heap = [(f, [(s, (0, 0))]) for s, f in frequencies.items()]
        # Add EOF symbol.
        #if eof not in frequencies:
        #    heap.append((1, [(eof, (0, 0))]))

        # Use heapq approach to build the encodings of the huffman tree leaves.
        heapify(heap)
        while len(heap) > 1:
            # Pop the 2 smallest items from heap
            a = heappop(heap)
            b = heappop(heap)
            # Merge nodes (update codes for each symbol appropriately)
            merged = (
                a[0] + b[0],
                [(s, (n + 1, v)) for (s, (n, v)) in a[1]]
                + [(s, (n + 1, (1 << n) + v)) for (s, (n, v)) in b[1]]
            )
            heappush(heap, merged)

        # Code table is dictionary mapping symbol to (bitsize, value)
        table = dict(heappop(heap)[1])

        return cls(table, concat=concat, check=False, eof=eof)

    @classmethod
    def from_data(cls, data):
        """
        Build Huffman code table from symbol sequence

        :param data: sequence of symbols (e.g. byte string, unicode string, list, iterator)
        :return: HuffmanCoder
        """
        frequencies = collections.Counter(data)
        return cls.from_frequencies(frequencies, concat=_guess_concat(data))


#-------------------- author: RPP, 2020.09.11
def main():
    #codec = HuffmanCodec.from_data('hello world how are you doing today foo bar lorem ipsum')
    codec = HuffmanCodec.from_data([101, 102, 101, 102, 101, 102, 101, 100, 100, 104])
    t = codec.get_code_table()
    print(t)
    s, l = codec.get_code_len()
    print(s)
    print(l)

"""    
if __name__ == "__main__":
    main()
"""








#####################################################
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os
from scipy.io import wavfile
import math



#Imports all data from path and save sit in a dictionary with keys being the directory and the values being tha data itself
def importData(path):
    allData = {}                                             #creats the dict
    for files in os.listdir(path):                          #go all files 1 by 1
        _ ,typ = os.path.splitext(files)                    #save the type of documente in typ
        if typ == ".wav":
            _, data = wavfile.read(path+'/'+files)         #
            allData[files] = data
            print (data)
        elif typ == ".txt":
            texto = open(path+'/'+files,"r+")
            texto = texto.read()
            data = []
            for i in texto:
                data += i
                
            allData[files] = data
        elif typ==".bmp":
            data=img.imread(path+'/'+files)
            dataF = []
            if len(data.shape) == 3:
                for i in range(len(data)):
                    for ii in range(len(data[i])):
                        n = data[i][ii][0]*1000000000 + data[i][ii][1]*1000000 + data[i][ii][2] *1000 + data[i][ii][3]
                        dataF += [n]
            if len(data.shape) == 2:
                for i in range(len(data)):
                    for ii in range(len(data[i])):
                        n = data[i][ii]
                        dataF += [n]
            allData[files] = dataF
    return allData

def importDataTwoTwo(path):
    allData = {}
    for files in os.listdir(path):
        _ ,typ = os.path.splitext(files)
        if typ == ".wav":
            _, data = wavfile.read(path+'/'+files)
            dataF =[]
            if len(data.shape) == 2:
                data1 = data[:,0]
                data2 = data[:,1]
                dataF1 = numbersTwoTwo(data1,1000)
                dataF2 = numbersTwoTwo(data2,1000)
                for i in range(len(dataF1)): #put new values back in one array of arrays [[]]
                    dataF += [[dataF1[i], dataF2[i]]]
                allData[files] = dataF     
            else:
                data =numbersTwoTwo(data,1000)
                allData[files] = data
        elif typ == ".txt":
            texto = open(path+'/'+files,"r+",encoding='ASCII')
            texto = texto.read()
            data = []
            dataF = []
            for i in texto:
                data += i
            for i in range(len(data)//2): 
                    dataF += [data[2*i] + data[2*i+1]]
            allData[files] = dataF
            print("AQUI!!!!")
            print(dataF)
        """elif typ==".bmp":
            data=img.imread(path+'/'+files)
            dataF = []
            if len(data.shape) == 3:
                for i in range(len(data)):
                    for ii in range(len(data[i])):
                        n = data[i][ii][0]*1000000000 + data[i][ii][1]*1000000 + data[i][ii][2] *1000 + data[i][ii][3]
                        dataF += [n]
                dataF = numbersTwoTwoString(dataF)
            if len(data.shape) == 2:
                for i in range(len(data)):
                    for ii in range(len(data[i])):
                        n = data[i][ii]
                        dataF += [n]
                dataF = numbersTwoTwo(dataF,1000)
            allData[files] = dataF"""
    return allData

def importDataWav(path):
    allData = {}                                             #creats the dict
    for files in os.listdir(path):                          #go all files 1 by 1
        _ ,typ = os.path.splitext(files)                    #save the type of documente in typ
        if typ == ".wav":
            _, data = wavfile.read(path+'/'+files)         #
            allData[files] = data
    return allData

def numbersTwoTwo(data, n):
    dataF = []
    for i in range(len(data)//2): 
        n = data[2*i]*n+data[2*i+1]
        dataF += [n]
    return dataF

def numbersTwoTwoString(data):
    dataF = []
    for i in range(len(data)//2):
        n = str(data[2*i]) + str(data[2*i+1])
        dataF += [n]
        print (dataF)
    return dataF
    
     
def histogram(path, data):
    if isinstance(data,list):
        data = np.asarray(data)
    if len(data.shape) == 2:
        plt.subplot(121)
        histogramSimple(data[:,0])
        plt.xlabel('Caracteres no ficheiro: ' + path)
        plt.ylabel('Ocorrência')
        plt.title('Canal Esquerdo')
        plt.subplot(122)
        histogramSimple(data[:,1])
        plt.xlabel('Caracteres no ficheiro: ' + path)
        plt.ylabel('Ocorrência')
        plt.title('Canal Direito') 
    else:
        histogramSimple(data)
        plt.xlabel('Caracteres em ficheiro')
        plt.ylabel('Ocorrência')
        plt.title('Ocorrência de caracteres no ficheiro: ' + path)
    plt.tight_layout()
    plt.show()
        
def histogramSimple(data):
    values, nTimes = np.unique(data, return_counts = True)
    high = max(nTimes)
    plt.ylim(0, math.ceil(high + high*0.1))
    plt.bar(values, nTimes)
    

          
def calcEntropy(path, data):
    if isinstance(data,list):
        data = np.asarray(data)
    if len(data.shape) == 2:
        print("Entropia de " + path + " -> Canal Esquerdo: " , end='')
        calcEntropySimple(data[:,0])
        print("Entropia de " + path + " -> Canal Direito: " , end='')
        calcEntropySimple(data[:,1])
    else:
        print("Entropia de " + path + ": " , end='')
        calcEntropySimple(data)
       

def calcEntropySimple(data):
    if isinstance(data,list):
        data = tuple(data)
    _, nTimes = np.unique(data, return_counts = True)
    entropy = 0
    for i in nTimes:
        entropy -= i/sum(nTimes) * np.log2(i/sum(nTimes))
    print(entropy) 
    return entropy

def ex03(dados):
    for x,y in dados.items():
        #histogram(x, y)
        #calcEntropy(x, y)
        huff(x, y)

def ex06(query, target, passo):
    lenQuery = len(query)
    lenTarget = len(target)
    infoMutuaT = []
    alfabeto = np.unique(query + target)
    for i in range((lenTarget-lenQuery+1)//passo):
        infoMutuaT += [calcInfMutua(query, target[i*passo:i*passo+lenQuery], alfabeto)]
    print(infoMutuaT)

def calcInfMutua(data1, data2, alfabeto):
    #entropyData1 = calcEntropySimple(data1)
    entropyData2 = calcEntropySimple(data2)
    values1, nTimes1 = np.unique(data1, return_counts = True)
    values2, nTimes2 = np.unique(data2, return_counts = True)
    n = max(alfabeto)+1
    mat = np.zeros((n,n), dtype = int) 
    for i in data1:
        for ii in data2:
            mat[i][ii] += 1
    total = 0
    totalInt = 0
    prob2 = 0
    probCond = 0
    sums = mat.sum(axis = 0)
    for i in range(len(values1)):
        prob1 = nTimes1[i] / sum(nTimes1)
        for ii in range(len(values2)):
            probCond = mat[values1[i]][values2[ii]] / sums[values2[ii]]
            totalInt += probCond * np.log2(probCond)
        total += prob1 * totalInt
    infMutua = entropyData2 + total
    print (infMutua)
    return infMutua

def huff(path, data):
    if isinstance(data,list):
        data = np.asarray(data)
    if len(data.shape) == 2:
        print("Huff de " + path + " -> Canal Esquerdo: " , end='')
        huffSimple(path, data[:,0])
        print("Huff de " + path + " -> Canal Direito: " , end='')
        huffSimple(path, data[:,1])
    else:
        print("Huff de " + path + ": " , end='')
        huffSimple(path, data)
        
def huffSimple(path,data):
    codec = HuffmanCodec.from_data(data)
    s, l = codec.get_code_len()
    a, nTimes = np.unique(data, return_counts = True)
    
    soma = sum(list(map(lambda x,y: x*y , np.array(nTimes)/sum(nTimes),l)))
    print(soma)       
        
def ex06b():
    _, query = wavfile.read('data/saxriff.wav')             #import data saxriff
    _, t1 = wavfile.read('data/target01 - repeat.wav')      #import data target1
    _, t2 = wavfile.read('data/target02 – repeatNoise.wav') #import data target2
    passo = len(query) // 4
    infM1 = ex06(query, t1, '-', passo)
    infM2 = ex06(query, t2, '-', passo)
    
def ex06c():
    allData = importDataWav('data')
    _, query = wavfile.read('data/saxriff.wav')
    passo = len(query) // 4
    allInfM = []
    infMMax = {}
    for x,y in allData.items():
        infM = ex06(query, y, '-', passo)
        allInfM += [infM]
        infMMax[x] = max(infM)
    
    
        

  



#allData = importData('/Users/ricardovieira/Desktop/UC LEI/2º Ano/1º Semestre/Teoria da Informação/Teórico-Prático/TP1/data')


allData = importData('data')
ex03(allData)
#apresentarInfo()

#ex06([2,6,4,10,5,9,5,8,0,8] , [6,8,9,7,2,4,9,9,4,9,1,4,8,0,1,2,2,6,3,2,0,7,4,9,5,4,8,5,2,7,8,0,7,4,8,5,7,4,3,2,2,7,3,5,2,7,4,9,9,6] , 1)
#ex06c()

    
#allData = importData('/Users/ricardovieira/Desktop/UC LEI/2º Ano/1º Semestre/Teoria da Informação/Teórico-Prático/TP1/data')
#dados = importData('data')
#histogram('/Users/ricardovieira/Desktop/UC LEI/2º Ano/1º Semestre/Teoria da Informação/Teórico-Prático/TP1/data',data1)

"""
ar = np.zeros((10,10))
ar[0][0] = 20
ar[0][1] = 10
print (ar)

isto = np.sum(ar,axis=1)
print(isto)
"""

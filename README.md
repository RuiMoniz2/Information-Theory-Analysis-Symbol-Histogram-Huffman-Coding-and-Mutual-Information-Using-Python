# Information-Theory-Analysis-Symbol-Histogram-Huffman-Coding-and-Mutual-Information-Using-Python
This repository contains a collection of Python functions and scripts for conducting information theory analysis. The main goal of the project is to explore fundamental concepts such as information, redundancy, entropy, and mutual information. The implemented functions enable the visualization of symbol histograms, calculation of the theoretical minimum limit for the average number of bits per symbol, Huffman coding, and computation of mutual information between signals.

## Technologies Used
The project utilizes the following technologies:

Python: The main programming language used for implementation.
NumPy: A library for numerical computations, employed for efficient array manipulation.
WAV and BMP file formats: Used for reading and analyzing audio and image files.

## Features Implemented
#### Symbol Histogram: A function that takes a source of information P with an alphabet A={a1,...,an} and visualizes the histogram of symbol occurrences.

#### Theoretical Minimum Limit: A code snippet that calculates the theoretical minimum limit for the average number of bits per symbol for a given source of information P with an alphabet A={a1,...,an}.

#### Statistical Distribution and Minimum Limit: Functions that determine the statistical distribution (histogram) and the minimum limit for the average number of bits per symbol for various sources of information, including lena.bmp, CT1.bmp, binaria.bmp, saxriff.wav, and texto.txt.

#### Huffman Coding: The provided Huffman coding functions compute the average number of bits per symbol for each source of information using this encoding scheme.

#### Variance Analysis: An analysis of the resulting code length variances and strategies to reduce variance.

#### Mutual Information: A function that calculates the mutual information values between a query signal and a target signal using a sliding window approach. The implementation includes an example with specific query and target signals.

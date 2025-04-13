import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pathlib
from docx import Document
from docx.shared import Inches
from io import BytesIO


#zliczanie jak długo nie wystąpił ten sam symbol
def count_different_symbols(symbols, start):
    """Counts how long the symbol i differs from i+1"""
    count =1
    for i in range(start+1, len(symbols)):
        if symbols[i] != symbols[i-1]:
            count += 1
        else:
            count-=1
            break
    return count
    

#zliczanie ile razy ten sam symbol
def count_repeated_symbols(symbols, start):
    """Counts how many times the same symbol repeats starting from 'start'"""
    count = 1
    while start + count < len(symbols) and symbols[start] == symbols[start + count]:
        count += 1
    return count

def rle_compress(symbols):
    # Preallocate a buffer for the worst-case scenario
    compressed = np.zeros(len(symbols) * 2, dtype=int)
    index = 0
    i = 0

    while i < len(symbols):
        count = count_repeated_symbols(symbols, i)
        compressed[index] = count
        compressed[index + 1] = symbols[i]
        index += 2
        i += count

    # but return only the filled part of the buffer
    return compressed[:index]
def rle_decompress(compressed):
    """Decompresses the RLE compressed data."""
    decompressed = []
    for i in range(0, len(compressed), 2):
        count = compressed[i]
        symbol = compressed[i + 1]
        decompressed.extend([symbol] * count)
    return np.array(decompressed, dtype=int)

def ByteRun_compress(data):
    compressed = np.zeros(len(data) * 2, dtype=int)
    i = 0
    index = 0
    maxPackSize = 128
    # -127 < n <= 0 if repeated
    # 0 < n <= 127 if not repeated
    while i+1 < len(data):
        #if repetition
            if data[i] == data[i + 1]:
                count = count_repeated_symbols(data, i)
                if count > maxPackSize:
                    count -=128
                n= -count +1
                compressed[index] = n
                compressed[index + 1] = data[i]
                index += 2
                i += count
            #if not repetition
            else:
                count = count_different_symbols(data, i)
                if count > maxPackSize:
                    count = maxPackSize
                compressed[index] = count-1
                for j in range(count):
                    compressed[index + 1 + j] = data[i + j]
                index += count+1
                i += count
                
        

    return compressed[:index]

path = pathlib.Path().absolute() / 'l6'
files = ['document.png', 'kolor.png', 'techniczny.png']
#RLE test
"""
ciag = [1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 6, 6, 6, 1]
compressed = rle_compress(ciag)
print(compressed)  # Output: [5 1 3 2 1 3 4 3 4 6]
decompressed = rle_decompress(compressed)
print(decompressed)  # Output: [1 1 1 1 1 2 2 2 3 4 5 6 6 6 6 1]#"""

#byterun test
ciag = [5, 5, 5, 1, 2, 3, 4, 4, 1, 2, 3, 4]

compressed = ByteRun_compress(ciag)
print("input   :", ciag)  # Output expected: [5, 5, 5, 1, 2, 3, 4, 4, 1, 2, 3, 4]
print("got     :", compressed)  # Output expected: [-2,5,2,1,2,3,-1,4,3,1,2,3,4]
print("expected: [-2  5  2  1  2  3 -1  4  3  1  2  3  4]")#"""

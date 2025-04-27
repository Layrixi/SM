import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pathlib
from docx import Document
from docx.shared import Inches
from io import BytesIO
import sys
import scipy
from tqdm import tqdm

def plotSubplot2(x1,y1,x2,y2,title1="plot1",title2="plot2",document=None):
        fig ,axs = plt.subplots(2,1,figsize=(10,7)) # tworzenie plota
        plt.subplot(2,1,1)
        plt.xlim(np.min(x1),np.max(x1))
        plt.ylim(np.min(y1),np.max(y1))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.plot(x1,y1)

        plt.subplot(2,1,2)
        plt.xlim(np.min(x2),np.max(x2))
        plt.ylim(np.min(y2),np.max(y2))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.plot(x2,y2)
        plt.show()

def plotAudio(signal,fs,fsize,document=None,TimeMargin=[0,0.02],title="Audio"):
        fig ,axs = plt.subplots(2,1,figsize=(10,7)) # tworzenie plota
        fig.suptitle(title) 
        plt.subplot(2,1,1)
        plt.xlim(TimeMargin)
        plt.ylim(np.min(signal), np.max(signal)) 
        plt.xlabel("T(s)")
        plt.ylabel("Amplitude")
        plt.plot(np.arange(0,signal.shape[0])/fs,signal)


        plt.subplot(2,1,2)
        yf = scipy.fftpack.fft(signal,fsize)
        xf=np.arange(0,fs/2,fs/fsize)
        dbspectrum = 20 * np.log10(np.abs(yf[:fsize // 2])+np.finfo(np.float64).eps)
        plt.xlabel("Fs(Hz)")
        plt.ylabel("Magnitude(dB)")
        plt.plot(np.arange(0,fs/2,fs/fsize),dbspectrum)
        max_magnitude = np.max(dbspectrum)
        max_frequency = xf[np.argmax(dbspectrum)]
        if document is not None:
                fig.tight_layout(pad=1.5) # poprawa czytelności 
                memfile = BytesIO() # tworzenie bufora
                fig.savefig(memfile) # z zapis do bufora
                document.add_picture(memfile, width=Inches(6)) # dodanie obrazu z bufora do pliku
                document.add_paragraph('Max db: {} dB w fs: {} Hz'.format(max_magnitude,max_frequency))
                memfile.close()
        else:
                plt.show()
        plt.close()

def kwant(data, bit):
        d = 2**bit - 1
        
        if np.issubdtype(data.dtype, np.floating):
                m = -1
                n = 1
        else:
                m = np.iinfo(data.dtype).min
                n = np.iinfo(data.dtype).max

        #1
        DataF = data.astype(float)
        DataF = (DataF - m) / (n - m)

        #2
        DataF = np.round(DataF * d)
        DataF = DataF / d

        #3
        DataF = DataF * (n - m) + m

        return DataF.astype(data.dtype)

#DPCM Z PREDYKCJĄ NIE DZIAŁA POPRAWNIE.
#dpcm bez predykcji
def DPCM_compress(x,bit):
    y=np.zeros(x.shape)
    e=0
    for i in range(0,x.shape[0]):
        y[i]=kwant((x[i]-e).astype(x.dtype),bit)
        e+=y[i]
    return y

#xp jrst dekompresja
def DPCM_decompress(y):
    x=np.zeros(y.shape)
    x[0]=y[0]
    for i in range(1,y.shape[0]):
        x[i]=y[i]+x[i-1]
    return x
#dpcm z predykcją, najprostsza to np.mean
def DPCM_Pred_Compress(x,bit,predictor,n,ceil=False): 
    y=np.zeros(x.shape)
    xp=np.zeros(x.shape)
    e=0
    for i in range(0,x.shape[0]):
        y[i]=kwant(np.floor(x[i]-e).astype(x.dtype),bit)
        xp[i]=y[i]+e
        idx=(np.arange(i-n,i,1,dtype=int)+1)
        idx=np.delete(idx,idx<0)
        e=predictor(xp[idx],ceil=ceil)
    return y

def DPCM_Pred_Decompress(y,predictor,n,ceil=False):
    x=np.zeros(y.shape)
    e=0
    for i in range(0,y.shape[0]):
        x[i]=y[i]+e
        idx=(np.arange(i-n,i,1,dtype=int)+1)
        idx=np.delete(idx,idx<0)
        e=predictor(x[idx],ceil=ceil)
    return x
def no_pred(X):
    return X[-1]

def mean_pred(X, ceil=True):
    if ceil:
        return np.trunc(np.mean(X))
    else:
        return np.mean(X)

def median_pred(X,ceil=True):
    if ceil:
        return np.ceil(np.median(X))
    else:
        return np.median(X)
def ALawCompress(x):
    A=87.6
    y=np.zeros(x.shape)
    idx=np.abs(x)<(1/A)
    y[idx]=np.sign(x[idx]) * ( A*np.abs(x[idx]) / (1+np.log(A)))
    idx=((1/A) <= np.abs(x)) & (np.abs(x) <= 1)
    y[idx]=np.sign(x[idx]) * ( (1+np.log(A*np.abs(x[idx]))) / (1+np.log(A)) ) 
    return y

def ALawDecompress(y):
    A=87.6
    x=np.zeros(y.shape)
    idy=np.abs(y) < (1/(1+np.log(A)))
    x[idy]=np.sign(y[idy]) * (np.abs(y[idy])*(1+np.log(A))/A)
    idy=(1/(1+np.log(A)) <= np.abs(y)) & (np.abs(y) <= 1)
    x[idy]=np.sign(y[idy]) * ( (np.exp( np.abs(y[idy]) * ( 1 + np.log(A)) - 1)) / A )
    return x

def MuLawCompress(x):
    mu=255
    y=np.zeros(x.shape)
    idx=np.abs(x)<=(1)
    y[idx]=np.sign(x[idx]) * (np.log(1+mu*np.abs(x[idx])) / np.log(1+mu))
    return y

def MuLawCompress(y):
    mu=255
    x=np.zeros(y.shape)
    idy=np.abs(y)<=(1)
    x[idy]=np.sign(x[idy]) * (1/mu) * ( (1+mu)**np.abs(y[idy]) - 1)
    return y



"""#mulaw and alaw testing
x=np.linspace(-1,1,1000)
y=0.9*np.sin(np.pi*x*4)
plt.plot(x)
plt.show()
xALawCompressed=ALawCompress(x)
plt.plot(x,xALawCompressed)
plt.show()
xALawyCompressedKwant = kwant(xALawCompressed,6)

xALawDecompressed=ALawDecompress(xALawyCompressedKwant)
plt.plot(xALawDecompressed)
plt.show()

xMuLawCompressed=MuLawCompress(x)
plt.plot(x,xMuLawCompressed)
plt.show()
xMuLawCompressedKwant = kwant(xMuLawCompressed,6)
plt.plot(x,xMuLawCompressedKwant)
plt.show()
#"""

"""#logical indexing test
R=np.random.rand(5,5)
A=np.zeros(R.shape)
B=np.zeros(R.shape)
C=np.zeros(R.shape)


idx=R<0.25
A[idx]=1 # <-
B[idx]+=0.25 # <-
C[idx]=2*R[idx]-0.25 # <-
C[np.logical_not(idx)]=4*R[np.logical_not(idx)]-0.5 # <-
print(R)
print(idx)
print(A)
print(B)
print(C)#"""

"""#tests v2
x=np.linspace(-1,1,1000)
y=0.9*np.sin(np.pi*x*4)
plt.plot(x,y)
plt.show()

sinALawCompressed=kwant(ALawCompress(y),6)

sinMuLawCompressed=kwant(MuLawCompress(y),6)

plotSubplot2(x,sinALawCompressed,x,sinMuLawCompressed,"sinALaw","sinMuLaw")#"""

#DPCM test
#x=np.array([15,16,20,14,5,10,15,13,11,7,10,11,20,1,23],dtype=np.int8)
x=np.linspace(-1,1,1000)
y=0.9*np.sin(np.pi*x*4)
xCompressed=DPCM_compress(y,7)
xDecompressed=DPCM_decompress(xCompressed)
plt.plot(x,xDecompressed)
plt.show()
print("x: \n",x)
print("xCompressed: \n",xCompressed)
print("xDecompressed:\n",xDecompressed)

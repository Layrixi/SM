from io import BytesIO
import pathlib
import random
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from docx import Document
from docx.shared import Inches
from skimage.metrics import structural_similarity as ssim
import math

def mse(K, I):
    return np.mean((K - I) ** 2)

def nmse(K, I):
    numerator = np.sum((K - I) ** 2)
    denominator = np.sum(K ** 2)
    return numerator / denominator

def psnr(K, I, max_I=255.0):
    mse_val = mse(K, I)
    if mse_val == 0:
        return float('inf')
    return 10 * math.log10((max_I ** 2) / mse_val)

def image_fidelity(K, I):
    numerator = np.sum((K - I) ** 2)
    denominator = np.sum(K ** 2)
    return 1 - (numerator / denominator)

def compute_ssim(K, I):
    return ssim(K, I, data_range=I.max() - I.min())

def szum(img,alpha,Noise_range=[-25,25]):
    rand = (Noise_range[1]-Noise_range[0])*np.random.random((img.shape))+Noise_range[0]
    noisy3 = (img + alpha * rand).clip(0,255).astype(np.uint8)
    return noisy3

path = pathlib.Path().absolute() / 'l10'
images = ['z1.png', 'z2.png', 'z3.png', 'z4.png']
methods = [cv2.blur, cv2.GaussianBlur, cv2.medianBlur, cv2.bilateralFilter]
img_meth = dict(zip(images, methods))
alpha = 0.5
compression_params=[5,25,50,70]


img=cv2.imread(path / 'z1.png')
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 5]
result, encimg = cv2.imencode('.jpg', img, encode_param)
decimg = cv2.imdecode(encimg, 1)

fig, axs = plt.subplots(1, 2 , sharey=True   )
axs[0].imshow(img)
axs[1].imshow(decimg)
plt.show()



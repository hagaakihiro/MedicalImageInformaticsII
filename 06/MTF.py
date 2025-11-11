import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
import audioread
import sys


# 三角形関数の定義
def triangular_pulse(x, d):
    return np.maximum(1 - np.abs(x / d), 0)/d
    
# 矩形関数の定義
def rect(x, width=1.0):
    """
    矩形関数を定義する。
    x: 入力配列またはスカラー
    width: 矩形関数の幅（デフォルトは1.0）
    """
    half_width = width / 2
    return np.where(np.abs(x) <= half_width, 1, 0)/np.sqrt(width)

def gaussian_wavepacket(x, x0, sigma):
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) / np.sqrt(np.pi * sigma**2)

xmin = -10
xmax = 10

x = np.linspace(xmin, xmax, 1000)

y = rect(x, 1.0)
#y = triangular_pulse(x, 1.0)
#y = gaussian_wavepacket(x, x0=0, sigma=1.0)

nn = len(x)
plt.plot(x,y)
plt.show()
#sys.exit()

freq = np.fft.fftfreq(len(x), d=(x[1] - x[0]))
fft_result = np.fft.fft(y)
fft_result_shifted = np.fft.fftshift(fft_result)
freq_shifted = np.fft.fftshift(freq)

nmax = np.max(fft_result_shifted.real) + np.max(fft_result_shifted.imag)
plt.plot(freq_shifted,fft_result_shifted.real)
plt.plot(freq_shifted,fft_result_shifted.imag+nmax)
#plt.show()

inn = len(fft_result)
mask0 = np.ones(inn//100)
mask1 = np.zeros(inn*98//100)
mask = np.append(mask1,mask0)
mask = np.append(mask0,mask)
iy = fft_result * mask
ix = freq

#plt.plot(freq_shifted,iy.real)
#plt.plot(freq_shifted,iy.imag+nmax)
plt.show()



ifreq = np.fft.fftfreq(len(ix), d=(ix[1] - ix[0]))
ifft_result = np.fft.ifft(iy)
ifft_result_shifted = np.fft.ifftshift(ifft_result)
ifreq_shifted = np.fft.ifftshift(freq)
mtf = np.abs(ifft_result)
mtf = mtf / np.max(mtf)
plt.plot(x,y)
plt.plot(x,mtf)
plt.show()



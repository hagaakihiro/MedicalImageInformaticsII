import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
import audioread
import sys
from scipy.stats import norm

def load_audioread(filepath, offset=0, duration=None, n_bytes=2):
    # (1)
    with audioread.audio_open(filepath) as f:
        sr         = f.samplerate
        n_channels = f.channels

        s_start = int(np.round(sr * offset)) * n_channels
        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr * duration)) * n_channels)

        # (2) Convert bytebuffer to numpy-1d array and rescale [-32768, 32767] to [-1.0, 1.0)
        scale = 1.0 / float(1 << ((8 * n_bytes) - 1))
        frame_list = [scale * np.frombuffer(frame, f"<i{n_bytes}").astype(np.float32) for frame in f]

        # (3) Select use frames
        y = [None for _ in range(len(frame_list))]
        n = 0
        start_idx, end_idx = 0, -1
        for k, frame in enumerate(frame_list):
            n_prev = n
            n = n + len(frame)
    
            if n < s_start:
                start_idx += 1
                continue

            if s_end < n_prev:
                end_idx = k - start_idx
                break
                
            if s_end < n:
                frame = frame[: s_end - n_prev]
            if n_prev <= s_start <= n:
                frame = frame[(s_start - n_prev) :]
                
            y[k] = frame
        
        if end_idx != -1:
            y = y[start_idx:end_idx]
        else:
            y = y[start_idx::]
            
        y = np.concatenate(y)
        
        # (4) channel-split
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T

    return y, sr

def autocorrelation(data, k):
    """Returns the autocorrelation of the *k*th lag in a time series data.
    
    Parameters
    ----------
    data : one dimentional numpy array
    k : the *k*th lag in the time series data (indexing starts at 0)
    """
    # yの平均
    y_avg = np.mean(data)
    # 分子の計算
    sum_of_covariance = 0
    for i in range(k+1, len(data)):
        covariance = ( data[i] - y_avg ) * ( data[i-(k+1)] - y_avg )
        sum_of_covariance += covariance
    # 分母の計算
    sum_of_denominator = 0
    for u in range(len(data)):
        denominator = ( data[u] - y_avg )**2
        sum_of_denominator += denominator
    
    return sum_of_covariance / sum_of_denominator
    

num = 201
x = np.linspace(-100,100,num)


for ii in range(100):
    noise = np.random.normal(loc=0, scale=1, size=len(x))
    if ii == 0:
        all_noise = noise
    else:
        all_noise += noise
    #plt.acorr(noise.astype('f')-noise.mean(), maxlags=None)
    #plt.show()
    #sys.exit()

    acorr_data = []
    for i in range((num)//2):
        acorr_data.append(autocorrelation(noise, num//2-i-1))
        #acorr_data.append(0)
    acorr_data.append(1)
    for i in range((num-1)//2):
        acorr_data.append(autocorrelation(noise, i))
        #acorr_data.append(0)
    acorr_data = np.asarray(acorr_data)
    if ii == 0:
        all_acorr = acorr_data
    else:
        all_acorr += acorr_data
    #print(len(acorr_data),len(x))
    #plt.plot(x,noise)
    #plt.bar(x,acorr_data,color="orange")
    #plt.show()
    #sys.exit()
    y = acorr_data[:-1]

    freq = np.fft.fftfreq(len(x), d=(x[1] - x[0]))
    fft_result = np.fft.fft(y)
    fft_result_shifted = np.fft.fftshift(fft_result)
    freq_shifted = np.fft.fftshift(freq)

    nmax = np.max(fft_result_shifted.real) + np.max(fft_result_shifted.imag)
    #plt.plot(freq_shifted[:-1],fft_result_shifted.real)
    #plt.plot(freq_shifted[:-1],fft_result_shifted.imag)
    #plt.show()
    
    if ii == 0:
        all_fft_result = np.sqrt(fft_result_shifted.real**2+fft_result_shifted.imag**2)
    else:
        all_fft_result += np.sqrt(fft_result_shifted.real**2+fft_result_shifted.imag**2)
        
        
all_fft_result[num//2] = 100
plt.plot(freq_shifted[:-1],all_fft_result/100)
plt.ylim(-1,2)
plt.show()

plt.bar(x,all_noise/100)
plt.ylim(-1,1)
plt.show()

plt.bar(x,all_acorr/100)
plt.show()



"""
# Histogram　Equalization
img = cv2.imread('prostate.png',0)

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]

hist,bins = np.histogram(img2.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()


equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('res.png',res)
"""

"""
# Dynamic Range Compression

t1 = 100
t2 = 150
img = cv2.imread('prostate.png',0)

hist,bins = np.histogram(img.flatten(),256,[0,256])
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
#plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()


median = cv2.medianBlur(img,5)

mask1 = np.where((t1 <= median), 1, 0)
mask2 = np.where((median <= t2), 1, 0)
mask = mask1*mask2
mask2 = np.where(mask==1, 0, 1)

c = 0.5
re1 = np.where(median < t1, c*(t1-median), 0)
re2 = np.where(median > t2, c*(median-t2), 0)
#re = re1 + re2
img2 = img + re1 - re2

hist2,bins2 = np.histogram(img2.flatten(),256,[0,256])
plt.hist(img2.flatten(),256,[0,256], color = 'b')
plt.xlim([0,256])
plt.show()


plt.imshow(img2, cmap = "gray")
plt.show()

DRC = np.hstack((img,img2)) #stacking images side-by-side
cv2.imwrite('DRC.png',DRC)

"""

"""
# Opening and Closing
img = cv2.imread('prostate.png',0)

t1 = 100
mask1 = np.where(img < t1, 0, 1)
mask1 = mask1.astype('uint8')

plt.imshow(mask1, cmap = "gray")
plt.show()

kernel = np.ones((10,10),np.uint8)
opening = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)

opening2 = np.hstack((mask1,opening)) #stacking images side-by-side
cv2.imwrite('opening.png',opening2)

closing = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

closing2 = np.hstack((mask1,closing)) #stacking images side-by-side
cv2.imwrite('closing.png',closing2)
"""


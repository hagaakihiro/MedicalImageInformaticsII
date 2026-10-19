import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
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
# Dynamic Range Compression　ダイナミックレンジ圧縮

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

"""
# moving average 移動平均

def valid_convolve(xx, size):
    b = np.ones(size)/size
    xx_mean = np.convolve(xx, b, mode="same")

    n_conv = math.ceil(size/2)

    # 補正部分
    xx_mean[0] *= size/n_conv
    for i in range(1, n_conv):
        xx_mean[i] *= size/(i+n_conv)
        xx_mean[-i] *= size/(i + n_conv - (size % 2))
    # size%2は奇数偶数での違いに対応するため

    return xx_mean

numx = 101
dx = 1/(numx-1)
df = pd.DataFrame(np.arange(numx)*dx,columns=["x"])
df["sin_2px"]=df.x.apply(lambda x: math.sin(2*x*math.pi))
df["gauss"]=np.random.normal(0,0.3,numx)
df["noise"]=df[["sin_2px","gauss"]].sum(axis=1)

n_conv = 5
noise_mean = valid_convolve(df.noise, n_conv)
plt.plot(df.x, df.noise,label="data")
plt.plot(df.x, noise_mean,label="Moving Ave.")
plt.legend()
plt.show(block=True)
"""

#cdf = hist.cumsum()
#cdf_normalized = cdf * hist.max()/ cdf.max()

#plt.plot(cdf_normalized, color = 'b')
#plt.hist(img.flatten(),256,[0,256], color = 'r')
#plt.xlim([0,256])
#plt.legend(('cdf','histogram'), loc = 'upper left')
#plt.show()

"""
# 画像間演算と特徴量計算
# 2画像の相関係数、相互情報量を計算
folder = "images/Anonymization_US/000"
img = cv2.imread(f"{folder}/dicom_000-0001.tif", 1)
img2 = cv2.imread(f"{folder}/dicom_000-0002.tif", 1)
# gray scale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
print(img.shape)
print(img2.shape)
plt.imshow(img, cmap = "gray")
plt.show()
plt.imshow(img2, cmap = "gray")
plt.show()
#sys.exit()
# Cross Correlation
# 相互相関係数（normalized cross-correlation coefficient）を計算
result = cv2.matchTemplate(img, img2, cv2.TM_CCORR_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print("Normalized cross-correlation coefficient (max):", max_val)
# imgとimg2の相互情報量を計算
def mutual_information(hgram):
    """ Mutual information for joint histogram
        https://en.wikipedia.org/wiki/Mutual_information
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals

    # Now we can do the calculation using the pxy, px, py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

# 2画像のjoint histogramを計算
hgram, x_edges, y_edges = np.histogram2d(img.ravel(), img2.ravel(), bins=256)
mi = mutual_information(hgram)
print("Mutual Information:", mi)
"""
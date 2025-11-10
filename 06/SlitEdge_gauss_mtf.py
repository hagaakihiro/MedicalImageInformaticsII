import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
import audioread
import sys
from scipy.stats import norm

# load Edge.tif and Slit.tif from current working directory
filenames = ['Edge.tif', 'Slit.tif']
images = {}

for fname in filenames:
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: failed to load {fname}")
        sys.exit(1)
    # convert to grayscale if image has color channels
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)
    images[fname] = img
    print(f"Loaded {fname}: shape={img.shape}, dtype={img.dtype}")

# show the two images side by side
plt.figure(figsize=(8, 4))
for i, fname in enumerate(filenames, 1):
    plt.subplot(1, 2, i)
    plt.imshow(images[fname], cmap='gray', interpolation='nearest')
    plt.title(fname)
    plt.axis('off')
plt.tight_layout()
plt.show()

# extract the center row from each loaded image and plot the intensity profiles
plt.figure(figsize=(8, 4))
for fname in filenames:
    img = images[fname]
    mid_row = img.shape[0] // 2
    signal = img[mid_row, :].astype(np.float64)

    # use edge regions (far from center) to estimate baseline and shift so those regions average to 0
    boundary = max(10, img.shape[1] // 8)  # use at least 10 pixels or 1/8 of width
    edge_samples = np.concatenate([signal[:boundary], signal[-boundary:]])
    if fname == 'Edge.tif':
        deriv = np.gradient(signal)
        baseline = np.mean(np.concatenate([deriv[:boundary], deriv[-boundary:]]))
        deriv_shifted = deriv - baseline
        # normalize so maximum value becomes 1 (fallback to abs max if needed)
        maxv = np.max(deriv_shifted)
        if maxv == 0 or not np.isfinite(maxv):
            maxv = np.max(np.abs(deriv_shifted))
        if maxv != 0 and np.isfinite(maxv):
            deriv_shifted = deriv_shifted / maxv
        plt.plot(deriv_shifted, label=f"{fname} (derivative, shifted, normalized)")
    else:
        baseline = np.mean(edge_samples)
        sig_shifted = signal - baseline
        # normalize so maximum value becomes 1 (fallback to abs max if needed)
        maxv = np.max(sig_shifted)
        if maxv == 0 or not np.isfinite(maxv):
            maxv = np.max(np.abs(sig_shifted))
        if maxv != 0 and np.isfinite(maxv):
            sig_shifted = sig_shifted / maxv
        plt.plot(sig_shifted, label=f"{fname} (shifted, normalized)")

sig = 7
gauss = norm.pdf(np.arange(0, 512), loc=256, scale=sig)/norm.pdf(256, loc=256, scale=sig)
plt.plot(gauss, label="(gauss)")

# compute RMSE between gaussian and the two processed signals (if present)
def compute_rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # truncate to common length
    n = min(a.size, b.size)
    if n == 0:
        return np.nan
    a = a[:n]
    b = b[:n]
    return np.sqrt(np.mean((a - b)**2))

# safely compute and print RMSEs, flush output so prints appear immediately
if 'deriv_shifted' in globals() or 'deriv_shifted' in locals():
    rmse_gauss_deriv = compute_rmse(gauss, deriv_shifted)
    print(f"RMSE(gauss, deriv_shifted) = {rmse_gauss_deriv:.6f}", flush=True)
else:
    print("deriv_shifted is not defined; skipping RMSE(gauss, deriv_shifted)", flush=True)

if 'sig_shifted' in globals() or 'sig_shifted' in locals():
    rmse_gauss_sig = compute_rmse(gauss, sig_shifted)
    print(f"RMSE(gauss, sig_shifted) = {rmse_gauss_sig:.6f}", flush=True)
else:
    print("sig_shifted is not defined; skipping RMSE(gauss, sig_shifted)", flush=True)

plt.xlabel('Pixel')
plt.ylabel('Intensity / Derivative')
plt.title('Center row intensity profile (Edge and Slit images)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
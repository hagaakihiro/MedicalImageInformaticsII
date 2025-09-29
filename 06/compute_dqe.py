import os, json, math, argparse, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------
# 小道具
# ---------------------------
def load_npy_or_png(path):
    if path.endswith(".npy") and os.path.exists(path):
        return np.load(path).astype(np.float32)
    if path.endswith(".png") and os.path.exists(path):
        return np.array(Image.open(path)).astype(np.float32)
    # 拡張子未指定のとき両方を探す
    if os.path.exists(path + ".npy"):
        return np.load(path + ".npy").astype(np.float32)
    if os.path.exists(path + ".png"):
        return np.array(Image.open(path + ".png")).astype(np.float32)
    raise FileNotFoundError(path)

def hann2d(h, w):
    wx = 0.5 - 0.5 * np.cos(2*np.pi*np.arange(w)/(w-1))
    wy = 0.5 - 0.5 * np.cos(2*np.pi*np.arange(h)/(h-1))
    return np.outer(wy, wx).astype(np.float32)

def radial_average(psd2, px_size_mm):
    """2D PSD -> 1D (cycles/mm) の同心円平均"""
    h, w = psd2.shape
    cy, cx = h//2, w//2
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    # 周波数スケール：1/ピクセル単位 -> cycles/mm
    fy_per_pix = 1.0/(h*px_size_mm)
    fx_per_pix = 1.0/(w*px_size_mm)
    # 近似的に正方画素前提で平均（px等方と仮定）
    f_per_pix = (fy_per_pix + fx_per_pix)/2.0

    rmax = int(min(cy, cx))
    prof = []
    freqs = []
    for r in range(rmax):
        mask = (rr >= r) & (rr < r+1)
        if np.any(mask):
            prof.append(psd2[mask].mean())
            freqs.append(r * f_per_pix)
    return np.array(freqs), np.array(prof)

def detrend_mean(roi):
    # 合成データは傾斜がほぼないので平均差し引きのみで十分
    return roi - np.mean(roi)

def fft_psd(roi, px_size_mm, window=True):
    """2D PSD（mm^2単位相当のスケール）を返す（簡便実装）"""
    roi = detrend_mean(roi.astype(np.float32))
    h, w = roi.shape
    win = hann2d(h, w) if window else np.ones((h, w), np.float32)
    roi_w = roi * win
    # 窓のエネルギーで正規化（L2）
    win_energy = (win**2).sum() / (h*w)
    F = np.fft.fftshift(np.fft.fft2(roi_w))
    psd2 = (np.abs(F)**2) / (h*w)  # パワー密度（単位系は相対）
    psd2 = psd2 / max(win_energy, 1e-12)
    # 面積スケール（ΔxΔy）を掛けると NPS の単位は mm^2 相当
    dx = dy = px_size_mm
    psd2 = psd2 * (dx * dy)
    return psd2

def tile_rois(img, roi_h=256, roi_w=256, step=None):
    if step is None:
        step = (roi_h, roi_w)  # 非重複
    H, W = img.shape
    rois = []
    for y in range(0, H - roi_h + 1, step[0]):
        for x in range(0, W - roi_w + 1, step[1]):
            rois.append(img[y:y+roi_h, x:x+roi_w])
    return rois

def estimate_edge_angle_from_meta(meta_files):
    for f in meta_files:
        if f.get("type") == "edge" and "angle_deg" in f:
            return float(f["angle_deg"])
    return None


def slanted_edge_mtf(
    edge_img, px_size_mm, angle_deg=None,
    binsize=0.1,              # ESFのビン幅 [pixel]
    strip_half_width_px=30,   # エッジ両側の帯状平均の半幅 [pixel]
    esf_smooth=15,            # ESFにかける移動平均窓（奇数）
    zero_pad_pow=2            # 2^pow 倍のゼロパディング（例: 2→4倍）
):
    """
    安定版 slanted-edge MTF
      1) 角度（既知 or 自動）→ 法線ベクトル
      2) 帯状ROIでサブピクセル投影 → ESF
      3) ESFを0–1に正規化 → 平滑 → 微分でLSF
      4) LSFにHann窓 & ゼロパディング → rFFT → f=0正規化
    戻り値: f_mm, mtf
    """
    img = edge_img.astype(np.float32)
    H, W = img.shape
    cy, cx = H/2.0, W/2.0

    # --- 角度の決定 ---
    if angle_deg is None:
        # 簡易角度推定：強度しきい＆PCA（堅牢化したい場合はCanny等でもOK）
        roi = img  # 全域を使う
        p20, p80 = np.percentile(roi, [20, 80])
        th = 0.5*(p20 + p80)
        gy, gx = np.gradient(roi)
        grad = np.hypot(gx, gy)
        edge_mask = (np.abs(roi - th) < 1.0*np.std(roi)) | (grad > np.percentile(grad, 95))
        ys, xs = np.nonzero(edge_mask)
        if len(xs) < 50:  # フォールバック
            idx = np.unravel_index(np.argsort(grad.ravel())[-5000:], grad.shape)
            ys, xs = idx[0], idx[1]
        coords = np.column_stack((xs, ys)).astype(np.float32)
        coords -= coords.mean(axis=0, keepdims=True)
        C = np.cov(coords.T)
        _, V = np.linalg.eigh(C)
        t = V[:, 1] / np.linalg.norm(V[:, 1])      # エッジ方向（接線）
        n = np.array([-t[1], t[0]], np.float32)     # 法線
    else:
        theta = math.radians(angle_deg)
        n = np.array([math.cos(theta), math.sin(theta)], np.float32)

    # --- 座標 & 法線距離 ---
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = xx - cx
    y = yy - cy
    d = x*n[0] + y*n[1]      # 各画素の法線方向距離 [pixel]

    # エッジ中心（距離0）の近く±strip幅の帯状ROIだけを使う
    strip_mask = np.abs(d) <= float(strip_half_width_px)
    vals = img[strip_mask]
    d_strip = d[strip_mask]

    # --- サブピクセル投影（ビンニングでESF） ---
    dmin, dmax = d_strip.min(), d_strip.max()
    nbins = max(32, int((dmax - dmin)/binsize) + 1)
    bins = np.linspace(dmin, dmax, nbins)
    esf = np.zeros(nbins-1, np.float64)
    cnt = np.zeros(nbins-1, np.int64)
    idx = np.clip(np.digitize(d_strip, bins) - 1, 0, nbins-2)
    np.add.at(esf, idx, vals.astype(np.float64))
    np.add.at(cnt, idx, 1)
    esf = np.divide(esf, np.maximum(cnt, 1))

    # --- 0–1 正規化（上下端の平均を使用） ---
    m = max(5, len(esf)//20)  # 端部5%相当
    low = np.nanmean(esf[:m]); high = np.nanmean(esf[-m:])
    denom = max(high - low, 1e-12)
    esf_n = (esf - low) / denom

    # --- ESF平滑化 → 微分でLSF ---
    k = esf_smooth if esf_smooth % 2 == 1 else esf_smooth + 1
    if k > 1:
        ker = np.ones(k, dtype=np.float64) / k
        esf_s = np.convolve(esf_n, ker, mode="same")
    else:
        esf_s = esf_n
    lsf = np.gradient(esf_s, binsize)  # binsizeは[px] → LSFの面積 ≈ 1

    # --- 窓掛け & ゼロパディング → rFFT ---
    # 窓は端点効果を抑えて人工ピークを低減
    N = len(lsf)
    hann = 0.5 - 0.5*np.cos(2*np.pi*np.arange(N)/(N-1))
    lsf_w = (lsf - np.mean(lsf)) * hann

    nfft = 2**(int(np.ceil(np.log2(N))) + zero_pad_pow)  # 4倍なら pow=2
    L = np.zeros(nfft, dtype=np.float64)
    L[:N] = lsf_w
    F = np.fft.rfft(L)
    mtf = np.abs(F)
    mtf /= (mtf[0] + 1e-12)  # f=0正規化

    # --- 周波数軸（cycles/mm） ---
    df_pix = 1.0 / (nfft * binsize)   # cycles/pixel
    f_pix = np.arange(len(mtf)) * df_pix
    f_mm = f_pix / px_size_mm         # cycles/mm

    return f_mm, mtf

def save_csv(path, cols, header):
    arr = np.vstack(cols).T
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.7g")

# ---------------------------
# メイン
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute NNPS, MTF, DQE from synthetic dataset")
    ap.add_argument("dataset_dir", help="dqe_synthetic_YYYYmmdd_HHMMSS フォルダ")
    ap.add_argument("--roi", type=int, default=256, help="NNPS用ROIサイズ")
    ap.add_argument("--step", type=int, default=256, help="ROIステップ（重なりなし=ROIサイズ）")
    args = ap.parse_args()
    ds = args.dataset_dir
    #print(ds)
    meta_path = os.path.join(ds, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("metadata.json が見つかりません: " + meta_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    px_mm = float(meta.get("pixel_pitch_mm", meta.get("params", {}).get("pixel_pitch_mm", 0.1)))
    nyq = 1.0/(2*px_mm)

    # ===== NNPS (doseごと) =====
    flats = [f for f in meta["files"] if f.get("type") == "flat"]
    doses = sorted(list({float(f["dose"]) for f in flats}))
    nnps_by_dose = {}
    freqs_common = None

    # 事前に dark 画像の平均を求めておく（同一データセットから）
    dark_paths = [os.path.join(ds, f["path"]) for f in meta["files"] if f.get("type") == "dark"]
    dark_means = [load_npy_or_png(p).mean() for p in dark_paths]
    mean_dark = float(np.mean(dark_means)) if dark_means else 0.0

    for dose in doses:
        paths = [os.path.join(ds, f["path"]) for f in flats if float(f["dose"]) == dose]

        # 画像読み込み（ここでは原画像）
        ims = [load_npy_or_png(p) for p in paths]
        # ---- ROI切り出し（平均引きはfft_psd側で行う前提にする）----
        rois = []
        for im in ims:
            im -= mean_dark
            if args.roi is None:
                rois.append(im)
            else:
                rois.extend(tile_rois(im, roi_h=args.roi, roi_w=args.roi, step=(args.step, args.step)))

        # ---- PSD（=NPS）平均 ----
        psd_list = []
        for r in rois:
            psd2 = fft_psd(r, px_mm, window=True)   # ここでROI内平均は引く（fft_psd側）
            psd_list.append(psd2)
        psd_mean = np.mean(psd_list, axis=0)

        # ---- 正規化用の平均信号 μ は (flat - dark) の平均 ----
        # 画像全体の平均でも良いが、ROIと同じ領域で取るなら厳密
        mean_flat = float(np.mean([im.mean() for im in ims]))
        mu = max(mean_flat - mean_dark, 1e-12)      # 0割回避

        # ---- 2D→1D（同心円平均）----
        f_nps, nps_1d = radial_average(psd_mean, px_mm)

        # ---- NNPS = NPS / μ^2 ----
        nnps_1d = nps_1d / (mu**2)

        # （任意）DC近傍の1–2ビンはデトレンド不足の影響を受けやすいので無効化
        # nnps_1d[:2] = np.nan

        nnps_by_dose[dose] = (f_nps, nnps_1d)
        if freqs_common is None or len(f_nps) < len(freqs_common):
            freqs_common = f_nps

    # ===== 出力 =====
    out_dir = os.path.join(ds, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # NNPS 各線量
    plt.figure()
    for d in doses:
        f_n, n1d = nnps_by_dose[d]
        plt.plot(f_n, n1d, label=f"dose={d}")
    plt.axvline(nyq, linestyle="--")
    plt.xlabel("Spatial frequency [cycles/mm]")
    plt.ylabel("NNPS [mm^2] (relative)")
    plt.title("NNPS(f)")
    plt.legend()
    plt.grid(True, which="both", linestyle=":")
    plt.savefig(os.path.join(out_dir, "nnps.png"), dpi=160, bbox_inches="tight")
    sys.exit()
    # ===== MTF（中間線量） =====
    dose_mid = sorted(doses)[len(doses)//2]
    edge_files = [f for f in meta["files"] if f.get("type") == "edge"]
    if not edge_files:
        raise RuntimeError("edge 画像が見つかりません。")
    angle_meta = estimate_edge_angle_from_meta(meta["files"])
    # 代表1枚で MTF（平均化したければ loop で複数平均してOK）
    edge_path = os.path.join(ds, edge_files[0]["path"])
    edge_img = load_npy_or_png(edge_path)
    f_mtf, mtf = slanted_edge_mtf(edge_img, px_mm, angle_deg=angle_meta)

    # ===== DQE（中間線量のNNPSを使用）=====
    f_nnps, nnps_mid = nnps_by_dose[dose_mid]
    # 軸合わせ（共通の周波数範囲まで補間）
    fmax = min(f_mtf.max(), f_nnps.max(), nyq)
    f_axis = np.linspace(0, fmax, 800)
    mtf_i = np.interp(f_axis, f_mtf, mtf)
    nnps_i = np.interp(f_axis, f_nnps, nnps_mid)

    # 入射量子数 q（metadata の q_mean_at_dose1 × dose_mid）
    q1 = float(meta.get("params", {}).get("q_mean_at_dose1", 5000.0))
    q = q1 * float(dose_mid)

    dqe = (mtf_i**2) / (q * np.maximum(nnps_i, 1e-30))

    # ===== 出力 =====
    out_dir = os.path.join(ds, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # NNPS 各線量
    plt.figure()
    for d in doses:
        f_n, n1d = nnps_by_dose[d]
        plt.plot(f_n, n1d, label=f"dose={d}")
    plt.axvline(nyq, linestyle="--")
    plt.xlabel("Spatial frequency [cycles/mm]")
    plt.ylabel("NNPS [mm^2] (relative)")
    plt.title("NNPS(f)")
    plt.legend()
    plt.grid(True, which="both", linestyle=":")
    plt.savefig(os.path.join(out_dir, "nnps.png"), dpi=160, bbox_inches="tight")

    # MTF
    plt.figure()
    plt.plot(f_mtf, mtf)
    plt.axvline(nyq, linestyle="--")
    plt.xlabel("Spatial frequency [cycles/mm]")
    plt.ylabel("MTF")
    plt.title("MTF(f) (slanted-edge)")
    plt.grid(True, which="both", linestyle=":")
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(out_dir, "mtf.png"), dpi=160, bbox_inches="tight")

    # DQE
    plt.figure()
    plt.plot(f_axis, dqe)
    plt.axvline(nyq, linestyle="--")
    plt.xlabel("Spatial frequency [cycles/mm]")
    plt.ylabel("DQE")
    plt.title(f"DQE(f)  (dose={dose_mid}, q={q:.1f})")
    plt.grid(True, which="both", linestyle=":")
    plt.ylim(0, 1.05*np.nanmax(dqe))
    plt.savefig(os.path.join(out_dir, "dqe.png"), dpi=160, bbox_inches="tight")

    # CSV 保存
    save_csv(os.path.join(out_dir, "nnps_mid.csv"),
             [f_nnps, nnps_mid], "f_cyc_per_mm,NNPS_mm2_relative")
    save_csv(os.path.join(out_dir, "mtf.csv"),
             [f_mtf, mtf], "f_cyc_per_mm,MTF")
    save_csv(os.path.join(out_dir, "dqe.csv"),
             [f_axis, dqe], "f_cyc_per_mm,DQE")

    print("Saved results to:", out_dir)
    print("  - nnps.png / mtf.png / dqe.png")
    print("  - nnps_mid.csv / mtf.csv / dqe.csv")

if __name__ == "__main__":
    main()

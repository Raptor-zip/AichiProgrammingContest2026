# https://github.com/opencv/opencv/blob/4.x/doc/acircles_pattern.png

import cv2
import numpy as np
import glob
import json
import os

# ---- 設定 ----
pattern_size = (4, 11)        # (列, 行)
circle_spacing = 20.0         # mm（または任意の単位）
pattern_flag = cv2.CALIB_CB_ASYMMETRIC_GRID  # 非対称円グリッド

# ---- 3D座標生成 ----
# 非対称円グリッドの場合はx方向に半分ずらした配置になるため、
# OpenCVのドキュメントどおり (2*j + i%2) * (spacing/2) を使う。
num_cols, num_rows = pattern_size  # (列, 行)
objp = np.zeros((num_cols * num_rows, 3), np.float32)
idx = 0
for i in range(num_rows):
    for j in range(num_cols):
        x = (2 * j + i % 2) * (circle_spacing / 2.0)
        y = i * circle_spacing
        objp[idx, 0] = x
        objp[idx, 1] = y
        idx += 1

# ---- データ格納 ----
objpoints = []  # 3D点
imgpoints = []  # 2D点

def collect_images():
    patterns = [
        './circle_images/*.png',
        './circle_images/*.jpg',
        './circle_images/*.jpeg',
        './circle_images/*.bmp'
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    # 自然順に並べる
    files.sort(key=lambda x: (os.path.dirname(x), os.path.basename(x)))
    return files

images = collect_images()
print(f"{len(images)} 枚の画像を検出しました。")

show_preview = False  # GUI不要ならFalseのまま
save_visualization = True
save_dir = os.path.join('circle_images', 'detected')
if save_visualization:
    os.makedirs(save_dir, exist_ok=True)

# BlobDetectorパラメータ設定（必要に応じて調整）
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 20
params.maxArea = 1_000_000
params.filterByCircularity = True
params.minCircularity = 0.7
params.filterByConvexity = False
params.filterByInertia = True
params.minInertiaRatio = 0.1
params.minThreshold = 10
params.maxThreshold = 220
params.thresholdStep = 10

blobDetector = cv2.SimpleBlobDetector_create(params)

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"警告: 画像を読み込めませんでした -> {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    ret, centers = cv2.findCirclesGrid(
        gray, pattern_size, flags=pattern_flag, blobDetector=blobDetector
    )
    if ret:
        objpoints.append(objp)
        imgpoints.append(centers)
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern_size, centers, ret)
        if save_visualization:
            base = os.path.basename(fname)
            name, ext = os.path.splitext(base)
            out_path = os.path.join(save_dir, f"{name}_detected{ext}")
            cv2.imwrite(out_path, vis)
        if show_preview:
            cv2.imshow('Detected Circles', vis)
            cv2.waitKey(200)

if show_preview:
    cv2.destroyAllWindows()

# ---- キャリブレーション ----
if len(objpoints) == 0:
    print("エラー: パターンを検出できた画像がありません。calibrateCameraは実行しません。")
    print("ヒント: ")
    print(" - pattern_size が実際の列×行と一致しているか確認 (現在: ", pattern_size, ")")
    print(" - 非対称円グリッドの向き（回転）を変えて撮影してみる")
    print(" - BlobDetectorのminArea, minCircularity, minInertiaRatioを調整")
    print(" - 照明や露出を調整し、コントラストを上げる")
    raise SystemExit(1)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("=== キャリブレーション結果 ===")
print("カメラ行列 (mtx):\n", mtx)
print("歪み係数 (dist):\n", dist.ravel())

# ---- JSON保存 ----
params = {
    "pattern_size": list(pattern_size),
    "circle_spacing": circle_spacing,
    "pattern_flag": int(pattern_flag),
    "camera_matrix": mtx.tolist(),
    "dist_coeff": dist.tolist(),
    "reprojection_error": float(ret)
}

with open("camera_params.json", "w") as f:
    json.dump(params, f, indent=4)

print("\n✅ camera_params.json に保存しました。")

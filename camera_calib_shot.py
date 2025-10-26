"""Simple capture script for camera calibration images.

This script fetches a JPEG snapshot from an HTTP camera endpoint and
saves one or more images into `./circle_images/` so that
`camera_calib_calc.py` can later use them for calibration.

Features:
- CLI: set URL, count, interval, save-dir
- Optional pattern detection: only save images where the asymmetric
  circle grid is detected (useful to avoid saving failed frames)
"""

import argparse
import os
import time
from datetime import datetime

import requests
import numpy as np
import cv2


def make_blob_detector(
    min_area=200,
    max_area=1_000_000,
    min_circularity=0.5,
    min_inertia=0.05,
    blob_color=0,
):
    """Create a SimpleBlobDetector with more permissive defaults suitable for
    very high-resolution images. Defaults are intentionally conservative so the
    detector works on 12MP images; caller can override via CLI args.
    """
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = int(min_area)
    params.maxArea = int(max_area)
    params.filterByCircularity = True
    params.minCircularity = float(min_circularity)
    params.filterByConvexity = False  # 凸性は無視
    params.filterByInertia = True
    params.minInertiaRatio = float(min_inertia)
    # --- しきい値設定（明るさベースの2値化） ---
    params.minThreshold = 5
    params.maxThreshold = 255
    params.thresholdStep = 5
    # --- 色（黒/白ブロブ） ---
    params.filterByColor = True
    params.blobColor = int(blob_color)
    return cv2.SimpleBlobDetector_create(params)


def fetch_image_from_url(url, timeout=5.0):
    try:
        r = requests.get(url, timeout=timeout)
    except Exception as e:
        print(f"エラー: 取得に失敗しました: {e}")
        return None
    if r.status_code != 200:
        print(f"警告: ステータスコード {r.status_code}")
        return None
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def capture_images(
    url,
    count=5,
    interval=1.0,
    save_dir="./circle_images",
    autodetect=False,
    pattern_size=(4, 11),
    detector_min_area=200,
    detector_min_circularity=0.5,
    detector_min_inertia=0.05,
    detector_blob_color=0,
):
    os.makedirs(save_dir, exist_ok=True)

    debug_dir = os.path.join(save_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    detector = (
        make_blob_detector(
            min_area=detector_min_area,
            min_circularity=detector_min_circularity,
            min_inertia=detector_min_inertia,
            blob_color=detector_blob_color,
        )
        if autodetect
        else None
    )
    saved = []
    for i in range(count):
        ts_dbg = datetime.now().strftime("%Y%m%d_%H%M%S")

        img = fetch_image_from_url(url)
        if img is None:
            print(f"{i+1}/{count}: 取得失敗、{interval}s後に再試行")
            time.sleep(interval)
            continue

        save_it = True
        if autodetect:
            # Try multiple preprocessing pipelines because very high-res images
            # may need resizing / histogram equalization / adaptive thresholding.
            gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pipelines = []

            # 1) basic blurred full-res
            pipelines.append(("blur", cv2.GaussianBlur(gray_full, (5, 5), 0)))

            # 2) CLAHE (local contrast enhancement)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            pipelines.append(("clahe", clahe.apply(gray_full)))

            # 3) equalize histogram
            pipelines.append(("equalize", cv2.equalizeHist(gray_full)))

            # 4) resized (half) - sometimes detector expects smaller blobs
            h, w = gray_full.shape[:2]
            if max(w, h) > 1500:
                resized = cv2.resize(
                    gray_full, (w // 2, h // 2), interpolation=cv2.INTER_AREA
                )
                pipelines.append(("resized_half", cv2.GaussianBlur(resized, (5, 5), 0)))

            # 5) adaptive threshold (binary) and blur
            adapt = cv2.adaptiveThreshold(
                gray_full, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            pipelines.append(("adaptive_thresh", cv2.GaussianBlur(adapt, (5, 5), 0)))

            found = False
            centers = None
            found_pipeline = None
            for name, proc in pipelines:
                # If we used a resized image, need to pass the correct image to findCirclesGrid
                img_for_search = proc
                # For detection with a resized image, use the same detector (it expects blobs sized to that image)
                try:
                    ok, pts = cv2.findCirclesGrid(
                        img_for_search,
                        pattern_size,
                        flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                        blobDetector=detector,
                    )
                except Exception as e:
                    print(f"findCirclesGrid error on pipeline {name}: {e}")
                    ok = False
                    pts = None

                if ok:
                    found = True
                    centers = pts
                    found_pipeline = name
                    print(f"{i+1}/{count}: パターン検出 成功 — パイプライン: {name}")
                    break

            # Draw and save debug images (annotated)
            if centers is not None:
                # If centers are from a resized image, they are in resized coords. Attempt to draw on original.
                try:
                    cv2.drawChessboardCorners(img, pattern_size, centers, found)
                except Exception:
                    # Fallback: ignore drawing failure
                    pass
            debug_filename = f"debug_{ts_dbg}_{i+1}.jpg"
            debug_path = os.path.join(debug_dir, debug_filename)
            cv2.imwrite(debug_path, img)

            if not found:
                print(f"{i+1}/{count}: パターン検出失敗 — 保存しません")
                save_it = False
            else:
                print(
                    f"{i+1}/{count}: パターン検出 成功 (pipeline={found_pipeline}) — 保存します"
                )

        if save_it:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{ts}_{i+1}.jpg"
            path = os.path.join(save_dir, filename)
            cv2.imwrite(path, img)
            saved.append(path)
            print(f"保存: {path}")

        time.sleep(interval)

    print(f"完了: {len(saved)} / {count} 枚を保存しました。")
    return saved


def main():
    p = argparse.ArgumentParser(description="Capture images for camera calibration")
    p.add_argument(
        "--url", required=True, help="JPEG snapshot URL (e.g. http://ip:port/photo.jpg)"
    )
    p.add_argument("--count", type=int, default=5, help="Number of images to capture")
    p.add_argument(
        "--interval", type=float, default=0.4, help="Seconds between captures"
    )
    p.add_argument(
        "--save-dir",
        default="./circle_images",
        help="Directory to save captured images",
    )
    p.add_argument(
        "--autodetect",
        type=bool,
        default=True,
        help="Only save images where the calibration pattern is detected",
    )
    p.add_argument(
        "--pattern-cols",
        type=int,
        default=4,
        help="Pattern columns (as used in camera_calib_calc)",
    )
    p.add_argument(
        "--pattern-rows",
        type=int,
        default=11,
        help="Pattern rows (as used in camera_calib_calc)",
    )
    # add detector tuning args (before parsing)
    p.add_argument(
        "--detector-min-area",
        type=int,
        default=200,
        help="SimpleBlobDetector minArea (px) for blob detection",
    )
    p.add_argument(
        "--detector-min-circularity",
        type=float,
        default=0.5,
        help="SimpleBlobDetector minCircularity",
    )
    p.add_argument(
        "--detector-min-inertia",
        type=float,
        default=0.05,
        help="SimpleBlobDetector minInertiaRatio",
    )
    p.add_argument(
        "--detector-blob-color",
        type=int,
        choices=[0, 255],
        default=0,
        help="Blob color to detect: 0=dark blobs, 255=light blobs",
    )
    args = p.parse_args()

    pattern = (args.pattern_cols, args.pattern_rows)
    capture_images(
        args.url,
        args.count,
        args.interval,
        args.save_dir,
        args.autodetect,
        pattern,
        detector_min_area=args.detector_min_area,
        detector_min_circularity=args.detector_min_circularity,
        detector_min_inertia=args.detector_min_inertia,
        detector_blob_color=args.detector_blob_color,
    )


if __name__ == "__main__":
    main()

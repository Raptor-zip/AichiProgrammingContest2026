"""
画像処理関連の関数
- ホワイトバランス補正
- 回転補正
- 緑背景による紙検出・透視変換
- デバッグ画像の描画
"""

import cv2
import numpy as np
from typing import cast, Optional, Tuple


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    4点を左上、右上、右下、左下の順に並べ替える
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # 右上
    rect[3] = pts[np.argmax(d)]  # 左下
    return rect


def sample_edge_color(image: np.ndarray, sample_size: int = 20) -> np.ndarray:
    """
    画像の縁から緑色のサンプルを取得してHSV範囲を推定する
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 4辺からサンプルを取得
    samples = []

    # 上辺
    samples.append(hsv[0:sample_size, :, :].reshape(-1, 3))
    # 下辺
    samples.append(hsv[h-sample_size:h, :, :].reshape(-1, 3))
    # 左辺
    samples.append(hsv[:, 0:sample_size, :].reshape(-1, 3))
    # 右辺
    samples.append(hsv[:, w-sample_size:w, :].reshape(-1, 3))

    all_samples = np.vstack(samples)

    # 緑色っぽいピクセルをフィルタリング（H: 35-85が緑系）
    green_mask = (all_samples[:, 0] > 25) & (all_samples[:, 0] < 95) & (all_samples[:, 1] > 30)
    green_samples = all_samples[green_mask]

    if len(green_samples) < 100:
        print(f"[GreenDetect] Not enough green samples: {len(green_samples)}")
        return None

    # 中央値を基準にHSV範囲を決定
    h_median = np.median(green_samples[:, 0])
    s_median = np.median(green_samples[:, 1])
    v_median = np.median(green_samples[:, 2])

    print(f"[GreenDetect] Green HSV median: H={h_median:.0f}, S={s_median:.0f}, V={v_median:.0f}")

    return np.array([h_median, s_median, v_median])


def detect_paper_on_green(image: np.ndarray) -> Optional[np.ndarray]:
    """
    緑色の背景から紙の4頂点を検出する

    Returns:
        検出した4点の座標 (4, 2) [左上、右上、右下、左下]、検出できない場合はNone
    """
    h, w = image.shape[:2]
    print(f"[GreenDetect] Image size: {w}x{h}")

    # 縁から緑色をサンプリング
    green_hsv = sample_edge_color(image)
    if green_hsv is None:
        # フォールバック: 一般的な緑色範囲を使用
        print("[GreenDetect] Using default green range")
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
    else:
        # サンプルから範囲を決定（±15の幅）
        h_range = 20
        s_range = 50
        v_range = 80
        lower_green = np.array([
            max(0, green_hsv[0] - h_range),
            max(30, green_hsv[1] - s_range),
            max(30, green_hsv[2] - v_range)
        ])
        upper_green = np.array([
            min(179, green_hsv[0] + h_range),
            255,
            255
        ])
        print(f"[GreenDetect] Green range: {lower_green} - {upper_green}")

    # HSVに変換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 緑色マスクを作成
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # モルフォロジー処理でノイズ除去と穴埋め
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 緑色の反転 = 紙の領域（白い部分が紙）
    paper_mask = cv2.bitwise_not(green_mask)

    # さらにモルフォロジー処理
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 輪郭を検出
    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[GreenDetect] No contours found")
        return None

    # 最大面積の輪郭を取得
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:
        area = cv2.contourArea(contour)
        min_area = h * w * 0.05  # 画像の5%以上

        if area < min_area:
            continue

        # 輪郭を近似
        peri = cv2.arcLength(contour, True)

        for epsilon in [0.02, 0.03, 0.04, 0.05, 0.06]:
            approx = cv2.approxPolyDP(contour, epsilon * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2).astype(np.float32)
                ordered = order_points(pts)

                # アスペクト比チェック
                width1 = np.linalg.norm(ordered[1] - ordered[0])
                width2 = np.linalg.norm(ordered[2] - ordered[3])
                height1 = np.linalg.norm(ordered[3] - ordered[0])
                height2 = np.linalg.norm(ordered[2] - ordered[1])

                avg_width = (width1 + width2) / 2
                avg_height = (height1 + height2) / 2

                if avg_width == 0 or avg_height == 0:
                    continue

                aspect = max(avg_width, avg_height) / min(avg_width, avg_height)

                if aspect < 3.0:  # 極端な形状を除外
                    print(f"[GreenDetect] Found paper: area={area:.0f}, aspect={aspect:.2f}")
                    return ordered

    print("[GreenDetect] No valid quadrilateral found")
    return None


def perspective_transform_to_a4(
    image: np.ndarray,
    corners: np.ndarray,
    orientation: str = "auto",
    dpi: int = 150
) -> Optional[np.ndarray]:
    """
    検出した4点からA4サイズに透視変換する

    Args:
        image: 入力画像
        corners: 4点の座標 [左上、右上、右下、左下]
        orientation: "portrait"（縦）, "landscape"（横）, "auto"（自動判定）
        dpi: 出力解像度

    Returns:
        変換後の画像
    """
    if corners is None:
        return None

    # A4サイズ: 210mm x 297mm
    a4_width_mm = 210
    a4_height_mm = 297

    # DPIからピクセルサイズを計算
    mm_to_px = dpi / 25.4
    a4_width_px = int(a4_width_mm * mm_to_px)
    a4_height_px = int(a4_height_mm * mm_to_px)

    # 自動で縦横を判定
    if orientation == "auto":
        width = np.linalg.norm(corners[1] - corners[0])
        height = np.linalg.norm(corners[3] - corners[0])
        if width > height:
            orientation = "landscape"
        else:
            orientation = "portrait"

    if orientation == "landscape":
        output_width = a4_height_px
        output_height = a4_width_px
    else:
        output_width = a4_width_px
        output_height = a4_height_px

    # 変換先の座標
    dst_points = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype=np.float32)

    # 透視変換行列を計算
    matrix = cv2.getPerspectiveTransform(corners, dst_points)

    # 透視変換を適用
    transformed = cv2.warpPerspective(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    print(f"[GreenDetect] Transformed to {output_width}x{output_height} ({orientation})")
    return transformed


def auto_enhance_document(image: np.ndarray) -> np.ndarray:
    """
    ドキュメント画像を自動で見やすく補正する
    """
    # LAB色空間でCLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced


def replace_green_with_white(image: np.ndarray) -> np.ndarray:
    """
    画像内の緑色ピクセルを白に置き換える
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 緑色の範囲（広めに取る）
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])

    # 緑色マスク
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # マスクを少し膨張させて境界も含める
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask = cv2.dilate(green_mask, kernel, iterations=2)

    # 緑色部分を白に置き換え
    result = image.copy()
    result[green_mask > 0] = [255, 255, 255]

    return result


def correct_orientation_by_aruco(image: np.ndarray) -> np.ndarray:
    """
    ArUcoマーカーを検出して画像の向きを補正する
    マーカーの上辺が「上」を向くように回転させる（縦向きの紙に対応）
    """
    import cv2.aruco as aruco
    from config_loader import get_config

    config = get_config()

    # ArUco検出（設定から辞書タイプを取得）
    dict_type_name = config.get_aruco_dict_type()
    dict_type = getattr(aruco, dict_type_name, aruco.DICT_4X4_50)
    aruco_dict = aruco.getPredefinedDictionary(dict_type)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, params)

    print(f"[Orientation] Using ArUco dictionary: {dict_type_name}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print("[Orientation] No ArUco marker found, keeping original orientation")
        return image

    # マーカーの角度を計算（上辺のベクトル方向）
    # angle = 0° → 右向き、90° → 下向き、-90° → 上向き、180° → 左向き
    angle = calculate_marker_rotation(corners)
    print(f"[Orientation] Detected marker angle: {angle:.1f}°")

    # マーカーの上辺が「上」を向くように補正（目標角度: -90°）
    # 現在の角度から -90° にするために必要な回転量を計算
    # 例: angle=0° → -90°回転が必要、angle=90° → -180°回転が必要
    #     angle=-90° → 回転不要、angle=180° → +90°回転が必要

    # 最も近い90度の倍数を見つける（-90°を基準として）
    target_angle = -90  # マーカーの上辺が上を向く角度
    rotation_needed = target_angle - angle

    # -180° ～ 180° の範囲に正規化
    while rotation_needed > 180:
        rotation_needed -= 360
    while rotation_needed < -180:
        rotation_needed += 360

    # 最も近い90度の倍数に丸める
    rotation_options = [-180, -90, 0, 90, 180]
    rotation_needed = min(rotation_options, key=lambda x: abs(rotation_needed - x))

    if abs(rotation_needed) < 1:
        print("[Orientation] No rotation needed")
        return image

    print(f"[Orientation] Applying rotation: {rotation_needed}°")

    # 回転を適用
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 90度単位の回転はcv2.rotateを使うと高速で正確
    if rotation_needed == 90 or rotation_needed == -270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_needed == -90 or rotation_needed == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif abs(rotation_needed) == 180:
        return cv2.rotate(image, cv2.ROTATE_180)

    return image


def detect_aruco_rotation(image: np.ndarray) -> int:
    """
    ArUcoマーカーを検出して必要な回転量を返す（90度単位）

    Returns:
        回転量（0, 90, 180, -90）。マーカーが見つからない場合は0
    """
    import cv2.aruco as aruco
    from config_loader import get_config

    config = get_config()

    # ArUco検出（設定から辞書タイプを取得）
    dict_type_name = config.get_aruco_dict_type()
    dict_type = getattr(aruco, dict_type_name, aruco.DICT_4X4_50)
    aruco_dict = aruco.getPredefinedDictionary(dict_type)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, params)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print("[Orientation] No ArUco marker found in original image")
        return 0

    # マーカーの角度を計算
    angle = calculate_marker_rotation(corners)
    print(f"[Orientation] Detected marker angle in original: {angle:.1f}°")

    # マーカーの上辺が「上」を向くように補正（目標角度: -90°）
    target_angle = -90
    rotation_needed = target_angle - angle

    # -180° ～ 180° の範囲に正規化
    while rotation_needed > 180:
        rotation_needed -= 360
    while rotation_needed < -180:
        rotation_needed += 360

    # 最も近い90度の倍数に丸める
    rotation_options = [-180, -90, 0, 90, 180]
    rotation_needed = min(rotation_options, key=lambda x: abs(rotation_needed - x))

    # 180と-180は同じ
    if rotation_needed == 180:
        rotation_needed = 180
    elif rotation_needed == -180:
        rotation_needed = 180

    print(f"[Orientation] Required rotation: {rotation_needed}°")
    return int(rotation_needed)


def apply_rotation(image: np.ndarray, rotation: int) -> np.ndarray:
    """
    画像を指定された角度で回転（90度単位）
    """
    if rotation == 0:
        return image
    elif rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == -90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif abs(rotation) == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def ensure_portrait(image: np.ndarray) -> np.ndarray:
    """
    画像が横長の場合、90度回転させて縦長にする
    """
    h, w = image.shape[:2]
    if w > h:
        print(f"[Orientation] Image is landscape ({w}x{h}), rotating to portrait")
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def process_with_green_background(image: np.ndarray, enhance: bool = True) -> Tuple[np.ndarray, bool]:
    """
    緑背景を使って紙を検出・変換する

    処理順序:
    1. 緑背景から紙を検出
    2. A4サイズに透視変換
    3. 残った緑を白に変換
    4. ドキュメント補正
    5. 縦長になるように回転

    Returns:
        (処理後の画像, 成功したかどうか)
    """
    # 紙を検出
    corners = detect_paper_on_green(image)

    if corners is not None:
        # A4サイズに変換
        result = perspective_transform_to_a4(image, corners)
        if result is not None:
            # 残った緑を白に変換
            result = replace_green_with_white(result)

            if enhance:
                result = auto_enhance_document(result)

            # 最後に必ず縦長にする
            result = ensure_portrait(result)

            return result, True

    # 検出失敗時は元画像を返す
    if enhance:
        return auto_enhance_document(image), False
    return image, False


def auto_white_balance(image, corners):
    """
    ArUcoマーカーを6×6グリッドに分割し、各グリッドの白色と黒色の中央値を基準に
    画像全体のホワイトバランスを自動調整する（精度向上版）

    Args:
        image: 入力画像（BGR形式）
        corners: ArUcoマーカーの角の座標

    Returns:
        tuple: (補正後の画像, 可視化情報, 白色BGR値, 黒色BGR値)
    """
    if corners is None or len(corners) == 0:
        return image, None, None, None

    # 最初のマーカーの領域を取得
    marker_corners = corners[0].reshape(-1, 2).astype(np.float32)

    # マーカーのバウンディングボックスを取得
    x, y, w, h = cv2.boundingRect(marker_corners.astype(int))

    # マーカー領域を6×6グリッドに分割
    grid_size = 6
    cell_w = w / grid_size
    cell_h = h / grid_size

    white_samples = []  # 白色サンプル（BGRチャンネル別）
    black_samples = []  # 黒色サンプル（BGRチャンネル別）
    white_cells = []  # 白色と判定されたセルの中心座標
    black_cells = []  # 黒色と判定されたセルの中心座標

    # グレースケール画像を作成（明度判定用）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # グリッド情報を保存（描画用）
    grid_info = {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "grid_size": grid_size,
        "cell_w": cell_w,
        "cell_h": cell_h,
    }

    # 各グリッドセルを解析
    for i in range(grid_size):
        for j in range(grid_size):
            # セルの座標を計算
            cell_x = int(x + j * cell_w)
            cell_y = int(y + i * cell_h)
            cell_x2 = int(x + (j + 1) * cell_w)
            cell_y2 = int(y + (i + 1) * cell_h)

            # セル領域を抽出
            cell_gray = gray[cell_y:cell_y2, cell_x:cell_x2]
            cell_bgr = image[cell_y:cell_y2, cell_x:cell_x2]

            if cell_gray.size == 0 or cell_bgr.size == 0:
                continue

            # セル内の明度の中央値を取得
            median_gray = np.median(cell_gray)

            # セルの中心座標
            center_x = int(cell_x + cell_w / 2)
            center_y = int(cell_y + cell_h / 2)

            # 閾値で白黒を判定（中央値が128より大きければ白、小さければ黒）
            if median_gray > 128:
                # 白色セル: BGR各チャンネルの中央値を取得
                for ch in range(3):
                    white_samples.append(np.median(cell_bgr[:, :, ch]))
                white_cells.append((center_x, center_y))
            else:
                # 黒色セル: BGR各チャンネルの中央値を取得
                for ch in range(3):
                    black_samples.append(np.median(cell_bgr[:, :, ch]))
                black_cells.append((center_x, center_y))

    # サンプルが十分に取得できなかった場合は元の画像を返す
    if len(white_samples) < 3 or len(black_samples) < 3:
        return image, None, None, None

    # 白色と黒色の代表値を計算（全サンプルの中央値）
    # BGRチャンネルごとに分けて計算
    white_samples = np.array(white_samples)
    black_samples = np.array(black_samples)

    # 3チャンネル分に再構成
    num_white_pixels = len(white_samples) // 3
    num_black_pixels = len(black_samples) // 3

    if num_white_pixels == 0 or num_black_pixels == 0:
        return image, None, None, None

    white_bgr = np.array(
        [
            np.median(white_samples[0::3]),  # B
            np.median(white_samples[1::3]),  # G
            np.median(white_samples[2::3]),  # R
        ]
    )

    black_bgr = np.array(
        [
            np.median(black_samples[0::3]),  # B
            np.median(black_samples[1::3]),  # G
            np.median(black_samples[2::3]),  # R
        ]
    )

    # 線形変換の係数を計算
    # black_bgr → 0、white_bgr → 255 になるように変換
    range_vals = white_bgr - black_bgr
    range_vals = np.where(range_vals < 1, 1, range_vals)  # ゼロ除算を防ぐ

    # 画像全体を正規化
    corrected = image.astype(float)
    corrected = (corrected - black_bgr) * (255.0 / range_vals)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    # 可視化情報をまとめて返す
    viz_info = {
        "grid": grid_info,
        "white_cells": white_cells,
        "black_cells": black_cells,
    }

    return corrected, viz_info, white_bgr, black_bgr


def calculate_marker_rotation(corners):
    """
    ArUcoマーカーの回転角度を計算する

    Args:
        corners: マーカーの角の座標 (1, 4, 2) の形状で、4つの角の座標を持つ

    Returns:
        float: 回転角度（度）
    """
    if corners is None or len(corners) == 0:
        return 0.0

    # 最初のマーカーの角を取得
    pts = corners[0].reshape(4, 2)

    # ArUcoマーカーの角の順序: 左上、右上、右下、左下
    # 上辺のベクトルから角度を計算（左上→右上）
    top_left = pts[0]
    top_right = pts[1]

    # ベクトルの角度を計算
    dx = top_right[0] - top_left[0]
    dy = top_right[1] - top_left[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def correct_rotation(image, angle_deg):
    """
    画像を最も近い直角（0°, 90°, 180°, 270°）に補正する

    Args:
        image: 入力画像
        angle_deg: 現在の角度（度）

    Returns:
        tuple: (回転後の画像, 適用した回転量)
    """
    # 最も近い90度の倍数を見つける
    normalized_angle = angle_deg % 360

    # 各基準角度との差を計算
    angles = [0, 90, 180, 270]
    differences = [abs(normalized_angle - a) for a in angles]
    # 360度境界も考慮
    differences.append(abs(normalized_angle - 360))

    # 最小の差を持つ角度を選択
    min_diff_idx = differences.index(min(differences))
    if min_diff_idx == 4:  # 360度の場合は0度と同じ
        target_angle = 0
    else:
        target_angle = angles[min_diff_idx]

    # 必要な回転量を計算
    rotation_needed = target_angle - normalized_angle

    # 回転量が小さい場合は補正しない（閾値: 1度）
    if abs(rotation_needed) < 1.0:
        return image, 0.0

    # 画像の中心を取得
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 回転行列を作成（OpenCVは反時計回りが正なので、符号を反転）
    rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_needed, 1.0)

    # 回転後の画像サイズを計算（画像が切れないように）
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 回転行列の平行移動成分を調整
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # 画像を回転
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return rotated, rotation_needed


def draw_debug_grid(image, viz_info):
    """
    デバッグ用のグリッドとサンプルポイントを画像に描画する

    Args:
        image: 描画対象の画像
        viz_info: auto_white_balanceから返される可視化情報

    Returns:
        描画後の画像
    """
    if viz_info is None:
        return image

    debug_frame = image.copy()
    grid = viz_info["grid"]
    white_cells = viz_info["white_cells"]
    black_cells = viz_info["black_cells"]

    # 6×6グリッド線を描画
    x, y, w, h = grid["x"], grid["y"], grid["w"], grid["h"]
    grid_size = grid["grid_size"]
    cell_w = grid["cell_w"]
    cell_h = grid["cell_h"]

    # 縦線を描画（緑色、太さ2）
    for i in range(grid_size + 1):
        line_x = int(x + i * cell_w)
        cv2.line(debug_frame, (line_x, y), (line_x, y + h), (0, 255, 0), 2)

    # 横線を描画（緑色、太さ2）
    for i in range(grid_size + 1):
        line_y = int(y + i * cell_h)
        cv2.line(debug_frame, (x, line_y), (x + w, line_y), (0, 255, 0), 2)

    # 白色サンプルポイントを描画（黄色の円）
    for cx, cy in white_cells:
        cv2.circle(debug_frame, (cx, cy), 5, (0, 255, 255), -1)  # 黄色
        cv2.circle(debug_frame, (cx, cy), 6, (0, 0, 0), 2)  # 黒い外枠

    # 黒色サンプルポイントを描画（青色の円）
    for cx, cy in black_cells:
        cv2.circle(debug_frame, (cx, cy), 5, (255, 0, 0), -1)  # 青色
        cv2.circle(debug_frame, (cx, cy), 6, (255, 255, 255), 2)  # 白い外枠

    # 凡例を表示（背景付き）
    legend_y = max(y - 50, 30)  # 画面上端に近すぎないように調整
    cv2.rectangle(
        debug_frame, (x - 5, legend_y - 5), (x + 250, legend_y + 50), (0, 0, 0), -1
    )
    cv2.putText(
        debug_frame,
        f"White: {len(white_cells)} cells",
        (x, legend_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        debug_frame,
        f"Black: {len(black_cells)} cells",
        (x, legend_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )

    return debug_frame


def perspective_transform_from_marker(
    image: np.ndarray,
    corners: np.ndarray,
    marker_size_mm=50,
    output_dpi=300,
    draw_corners=False,
):
    """
    ArUcoマーカーの四隅の座標を使って透視変換（台形補正）を行う

    Args:
        image: 入力画像（BGR形式）
        corners: ArUcoマーカーの角の座標
        marker_size_mm: マーカーの実際のサイズ（mm）、デフォルト50mm
        output_dpi: 出力画像の解像度（DPI）、デフォルト300
        draw_corners: True の場合、変換後に元画像の4隅にマーカーを描画

    Returns:
        tuple: (変換後の画像, 変換行列, 出力サイズ(width, height), 変換後の4隅座標)
               変換後の4隅座標は [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] の形式 (左上、右上、右下、左下)
               マーカーが検出されない場合は (None, None, None, None)
    """
    if corners is None or len(corners) == 0:
        return None, None, None, None

    # 最初のマーカーの四隅の座標を取得
    # ArUcoマーカーの角の順序: 左上、右上、右下、左下
    src_points = corners[0].reshape(4, 2).astype(np.float32)

    # マーカーの実サイズから出力画像のピクセルサイズを計算
    # DPI (dots per inch) から mm あたりのピクセル数を計算
    # 1 inch = 25.4 mm
    pixels_per_mm = output_dpi / 25.4
    marker_size_pixels = int(marker_size_mm * pixels_per_mm)

    # 変換先の座標（正方形のマーカー）
    # 左上、右上、右下、左下の順
    dst_points = np.array(
        [
            [0, 0],
            [marker_size_pixels, 0],
            [marker_size_pixels, marker_size_pixels],
            [0, marker_size_pixels],
        ],
        dtype=np.float32,
    )

    # 透視変換行列を計算
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 画像全体を変換するために、画像の四隅も変換してみる
    h, w = image.shape[:2]
    image_corners = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
    ).reshape(-1, 1, 2)

    # 画像の四隅を変換
    transformed_corners = cv2.perspectiveTransform(image_corners, matrix)

    # 変換後の画像サイズを計算（全ての変換後の点を含む最小矩形）
    x_coords = transformed_corners[:, 0, 0]
    y_coords = transformed_corners[:, 0, 1]

    min_x = int(np.floor(np.min(x_coords)))
    max_x = int(np.ceil(np.max(x_coords)))
    min_y = int(np.floor(np.min(y_coords)))
    max_y = int(np.ceil(np.max(y_coords)))

    # 出力サイズ
    output_width = max_x - min_x
    output_height = max_y - min_y

    # オフセット行列を作成（負の座標を補正）
    offset_matrix = np.array(
        [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32
    )

    # 最終的な変換行列 = オフセット行列 × 透視変換行列
    final_matrix = offset_matrix @ matrix
    final_matrix = cast(np.ndarray, final_matrix)

    # 透視変換を適用
    transformed = cv2.warpPerspective(
        image,
        final_matrix,
        (output_width, output_height),
        flags=cv2.INTER_NEAREST,  # 補間を行わない
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # 元画像の4隅の座標を計算
    # 画像サイズと一致させるため、外側の境界座標を使用
    h_orig, w_orig = image.shape[:2]
    original_corners = np.array(
        [
            [0, 0],
            [w_orig, 0],
            [w_orig, h_orig],
            [0, h_orig],
        ],  # 左上  # 右上  # 右下  # 左下
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    # 変換後の座標を計算
    transformed_corner_points = cv2.perspectiveTransform(original_corners, final_matrix)

    # 座標をリスト形式に変換 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    # transformed_corner_points は shape (N,1,2) の配列なので (N,2) に整形してから展開する
    pts2 = transformed_corner_points.reshape(-1, 2)
    corner_coords = [(int(float(x)), int(float(y))) for x, y in pts2]

    if draw_corners:
        # 変換後の4隅にマーカーを描画する
        for i, corner in enumerate(transformed_corner_points):
            x, y = int(corner[0][0]), int(corner[0][1])
            cv2.circle(transformed, (x, y), 15, (255, 0, 0), -1)  # 塗りつぶし

    return transformed, final_matrix, (output_width, output_height), corner_coords

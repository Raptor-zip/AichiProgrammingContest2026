"""
画像処理関連の関数
- ホワイトバランス補正
- 回転補正
- デバッグ画像の描画
"""

import cv2
import numpy as np


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
        'x': x, 'y': y, 'w': w, 'h': h,
        'grid_size': grid_size,
        'cell_w': cell_w, 'cell_h': cell_h
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

    white_bgr = np.array([
        np.median(white_samples[0::3]),  # B
        np.median(white_samples[1::3]),  # G
        np.median(white_samples[2::3])   # R
    ])

    black_bgr = np.array([
        np.median(black_samples[0::3]),  # B
        np.median(black_samples[1::3]),  # G
        np.median(black_samples[2::3])   # R
    ])

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
        'grid': grid_info,
        'white_cells': white_cells,
        'black_cells': black_cells
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
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))

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
    grid = viz_info['grid']
    white_cells = viz_info['white_cells']
    black_cells = viz_info['black_cells']

    # 6×6グリッド線を描画
    x, y, w, h = grid['x'], grid['y'], grid['w'], grid['h']
    grid_size = grid['grid_size']
    cell_w = grid['cell_w']
    cell_h = grid['cell_h']

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
    cv2.rectangle(debug_frame, (x - 5, legend_y - 5), (x + 250, legend_y + 50), (0, 0, 0), -1)
    cv2.putText(debug_frame, f"White: {len(white_cells)} cells",
               (x, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(debug_frame, f"Black: {len(black_cells)} cells",
               (x, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return debug_frame

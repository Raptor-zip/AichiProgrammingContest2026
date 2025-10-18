import os
import sys
import json

# Ensure Qt uses PySide6's plugins rather than OpenCV's bundled plugins which can cause
# "Could not load the Qt platform plugin 'xcb'" errors. We set QT_PLUGIN_PATH to PySide6
# package plugins directory when possible.
try:
    from PySide6 import QtCore, QtWidgets, QtGui
    import PySide6
    pyside_pkg_dir = os.path.dirname(PySide6.__file__)
    pyside_plugins = os.path.join(pyside_pkg_dir, 'plugins')
    # Prepend to QT_PLUGIN_PATH so Qt finds PySide6 plugins first
    existing = os.environ.get('QT_PLUGIN_PATH', '')
    if pyside_plugins and pyside_plugins not in existing:
        os.environ['QT_PLUGIN_PATH'] = pyside_plugins + \
            (os.pathsep + existing if existing else '')
except Exception:
    # If PySide6 import fails, re-raise so the error is visible
    raise
import cv2
import cv2.aruco as aruco
import pytesseract
from PIL import Image
import numpy as np
from datetime import datetime


class ToastNotification(QtWidgets.QLabel):
    """
    自動的に消えるトースト通知ウィジェット
    """
    def __init__(self, message, parent=None, duration=2000):
        super().__init__(message, parent)
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(50, 50, 50, 220);
                color: white;
                padding: 15px 20px;
                border-radius: 8px;
                font-size: 14px;
            }
        """)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMaximumWidth(400)

        # 親ウィジェットの中央に配置
        if parent:
            self.setParent(parent)
            self.adjustSize()
            parent_rect = parent.rect()
            x = (parent_rect.width() - self.width()) // 2
            y = parent_rect.height() - self.height() - 50
            self.move(x, y)

        # 一定時間後に自動で消える
        QtCore.QTimer.singleShot(duration, self.fade_out)

    def fade_out(self):
        """フェードアウトして削除"""
        self.hide()
        self.deleteLater()


class SubjectSettingsDialog(QtWidgets.QDialog):
    """
    ArUcoマーカーIDと教科名を対応付けて管理するダイアログ。
    テーブル形式でID-教科名のペアを表示・編集し、JSONファイルに保存する。
    """
    def __init__(self, current_mappings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("教科設定")
        self.resize(500, 400)

        # current_mappings は {id: subject_name} の辞書
        self.mappings = current_mappings.copy() if current_mappings else {}

        layout = QtWidgets.QVBoxLayout(self)

        # 説明ラベル
        info_label = QtWidgets.QLabel(
            "ArUcoマーカーIDと教科名を対応付けます。\n"
            "追加・編集後、「保存」ボタンで設定を保存してください。"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # テーブルウィジェット
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["マーカーID", "教科名"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        # テーブルにデータを読み込む
        self.load_table_data()

        # ボタン群
        btn_layout = QtWidgets.QHBoxLayout()

        add_btn = QtWidgets.QPushButton("行を追加")
        add_btn.clicked.connect(self.add_row)
        btn_layout.addWidget(add_btn)

        remove_btn = QtWidgets.QPushButton("選択行を削除")
        remove_btn.clicked.connect(self.remove_row)
        btn_layout.addWidget(remove_btn)

        btn_layout.addStretch()

        save_btn = QtWidgets.QPushButton("保存")
        save_btn.clicked.connect(self.save_and_accept)
        btn_layout.addWidget(save_btn)

        cancel_btn = QtWidgets.QPushButton("キャンセル")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def load_table_data(self):
        """現在のマッピングをテーブルに表示"""
        self.table.setRowCount(0)
        for marker_id, subject in sorted(self.mappings.items(), key=lambda x: int(x[0])):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(marker_id)))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(subject))

    def add_row(self):
        """新しい行を追加"""
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(""))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))
        self.table.setCurrentCell(row, 0)

    def remove_row(self):
        """選択された行を削除"""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)

    def save_and_accept(self):
        """テーブルの内容をマッピング辞書に保存してダイアログを閉じる"""
        new_mappings = {}
        for row in range(self.table.rowCount()):
            id_item = self.table.item(row, 0)
            subject_item = self.table.item(row, 1)
            if id_item and subject_item:
                marker_id = id_item.text().strip()
                subject = subject_item.text().strip()
                if marker_id and subject:
                    new_mappings[marker_id] = subject
        self.mappings = new_mappings
        self.accept()

    def get_mappings(self):
        """現在のマッピングを取得"""
        return self.mappings


class OCRWorker(QtCore.QThread):
    """
    OCR をバックグラウンドで実行するためのスレッドクラス。
    UI スレッドをブロックしないように、QThread を継承して OCR を行い、完了時に
    `finished` シグナルで結果の文字列を送出します。

    コンストラクタは以下を受け付けます:
      - frame: OpenCV の BGR numpy 配列（カメラのフレーム）
      - image_path: ファイルパスから読み込む場合はそのパス
    frame と image_path のどちらかが指定されていれば動作します。
    """
    finished = QtCore.Signal(str)

    def __init__(self, frame=None, image_path=None, parent=None):
        super().__init__(parent)
        # BGR フレームを受け取る場合はコピーして保持する
        # もしくは保存済みファイルのパスを受け取る
        self.frame = frame.copy() if frame is not None else None
        self.image_path = image_path

    def run(self):
        """QThread のエントリポイント。ここで pytesseract を呼び出す。"""
        try:
            # フレームが渡されていれば numpy -> PIL の変換を行う
            if self.frame is not None:
                pil_image = Image.fromarray(
                    cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
            elif self.image_path is not None:
                # 画像ファイルから直接読み込む
                pil_image = Image.open(self.image_path)
            else:
                raise ValueError("No image provided to OCRWorker")

            # 日本語＋英語で OCR を試みる
            text = pytesseract.image_to_string(pil_image, lang='jpn+eng')
            text = text.strip()
            if not text:
                text = "(no text detected)"
        except Exception as e:
            # OCR 中に問題があればエラーメッセージを返す
            text = f"[OCR error] {e}"
        # 結果をシグナルで送出して UI スレッド側で受け取れるようにする
        self.finished.emit(text)


class CameraWindow(QtWidgets.QMainWindow):
    def __init__(self, debug_mode=False):
        super().__init__()
        self.setWindowTitle("Aruco + OCR Camera")
        self.resize(1000, 700)

        # デバッグモード
        self.debug_mode = debug_mode

        # captures directory
        self.captures_dir = os.path.join(os.path.dirname(__file__), 'captures')
        os.makedirs(self.captures_dir, exist_ok=True)

        # subject mappings JSON file
        self.settings_file = os.path.join(os.path.dirname(__file__), 'subject_mappings.json')
        self.subject_mappings = self.load_subject_mappings()

        # Video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Cannot open camera")
            sys.exit(1)

        # カメラの画質を最大に設定
        # 解像度を最大に設定（一般的なWebカメラは1920x1080または1280x720）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # その他の画質パラメータを設定
        # FOURCC コーデックを MJPEG に設定（高画質）
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        # 可能な場合はフレームレートも設定（30fps）
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # ArUco setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, params)

        # ArUco 信頼度フィルターの閾値（調整可能）
        # - aruco_area_ratio_threshold: 画像面積に対するマーカー面積の比率（小さすぎるものを除外）
        # - aruco_fill_threshold: マーカー凸包に対する実際のポリゴン面積の充填率（歪み判定）
        self.aruco_area_ratio_threshold = 0.001  # 画像面積の割合（例: 0.001 -> 0.1%）
        self.aruco_fill_threshold = 0.6  # bounding rect に対する面積の充填率

        # ----- UI セットアップ -----
        # メインウィジェットとレイアウトを作る
        # (ビデオ表示ラベル、コントロールボタン群、OCR 出力)
        # UI 部分は後ほど update_frame() で画像を QLabel に流し込みます
        # -----
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # カメラ映像を表示する QLabel
        self.video_label = QtWidgets.QLabel()
        # 最小サイズを設定しておく（ウィンドウの縮小で潰れすぎないようにする）
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        # 教科設定ボタン
        self.settings_btn = QtWidgets.QPushButton("教科設定")
        self.settings_btn.clicked.connect(self.open_subject_settings)
        controls.addWidget(self.settings_btn)

        # ホワイトバランス補正トグルボタン
        self.wb_toggle_btn = QtWidgets.QPushButton("✓ 補正ON")
        self.wb_toggle_btn.setCheckable(True)
        self.wb_toggle_btn.setChecked(True)
        self.wb_toggle_btn.clicked.connect(self.toggle_white_balance)
        self.wb_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 5px 10px;
            }
            QPushButton:checked {
                background-color: #4CAF50;
            }
            QPushButton:!checked {
                background-color: #757575;
            }
        """)
        controls.addWidget(self.wb_toggle_btn)

        # ArUco 検出をトリガーに自動撮影するための単発タイマー
        # マーカーを検出したらこのタイマーを start して一定時間（capture_delay_ms）後に撮影する
        self.capture_delay_ms = 800  # ミリ秒
        self.aruco_auto_timer = QtCore.QTimer(self)
        self.aruco_auto_timer.setSingleShot(True)
        self.aruco_auto_timer.timeout.connect(self.take_picture)
        # 前フレームの検出状態を保持して、状態遷移でタイマーを開始/停止する
        self._last_aruco_detected = False

        # ホワイトバランス補正のON/OFF切り替えフラグ
        self.white_balance_enabled = True

        spacer = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        controls.addItem(spacer)

        self.quit_btn = QtWidgets.QPushButton("終了")
        self.quit_btn.clicked.connect(self.close)
        controls.addWidget(self.quit_btn)

        # 撮影後に一時停止したライブフィードを再開するボタン
        self.resume_btn = QtWidgets.QPushButton("撮影再開")
        self.resume_btn.clicked.connect(self.resume_camera)
        controls.addWidget(self.resume_btn)

        # OCR の結果などを表示するテキスト領域
        self.ocr_output = QtWidgets.QTextEdit()
        self.ocr_output.setReadOnly(True)
        self.ocr_output.setMaximumHeight(120)
        layout.addWidget(self.ocr_output)

        # カメラからフレームを定期的に取得して表示するためのタイマー
        # 大体 30ms ごと（約33fps）で update_frame を呼ぶ
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33fps

        # remove automatic OCR timer: OCR should run after taking a picture
        # (the OCR button will still be available for manual runs)

        self.current_frame = None

    def load_subject_mappings(self):
        """JSONファイルから教科マッピングを読み込む"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "警告", f"設定ファイルの読み込みに失敗しました: {e}")
                return {}
        return {}

    def save_subject_mappings(self):
        """教科マッピングをJSONファイルに保存"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.subject_mappings, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "エラー", f"設定ファイルの保存に失敗しました: {e}")
            return False

    def open_subject_settings(self):
        """教科設定ダイアログを開く"""
        dialog = SubjectSettingsDialog(self.subject_mappings, self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self.subject_mappings = dialog.get_mappings()
            if self.save_subject_mappings():
                # トースト通知で保存完了を表示（2秒後に自動で消える）
                toast = ToastNotification("教科設定を保存しました", self, duration=4000)
                toast.show()

    def toggle_white_balance(self):
        """ホワイトバランス補正のON/OFFを切り替え"""
        self.white_balance_enabled = self.wb_toggle_btn.isChecked()
        if self.white_balance_enabled:
            self.wb_toggle_btn.setText("✓ 補正ON")
            toast = ToastNotification("ホワイトバランス補正: ON", self, duration=2000)
        else:
            self.wb_toggle_btn.setText("補正OFF")
            toast = ToastNotification("ホワイトバランス補正: OFF", self, duration=2000)
        toast.show()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # keep original BGR for saving/ocr
        self.current_frame = frame.copy()

        # ArUco マーカー検出: グレースケールで検出を行い、検出があればマーカー候補をフィルタ
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        # corners/ids は None の場合や ndarray/list の場合があるため安全に処理する
        filtered_corners = []
        filtered_ids = []
        if ids is not None and len(ids) > 0:
            # image area used for normalization
            img_h, img_w = frame.shape[:2]
            img_area = float(img_h * img_w)
            for ci, cid in zip(corners, ids.flatten() if hasattr(ids, 'flatten') else ids):
                # corners の形状は (1,4,2) や (4,2) のことがある
                pts = None
                try:
                    pts = ci.reshape(-1, 2)
                except Exception:
                    # fallback: convert to numpy array
                    pts = cv2.UMat(ci).get().reshape(-1, 2)

                # polygon 面積を計算
                area = abs(cv2.contourArea(pts))
                # bounding rect と面積の比（詰まり具合）
                x, y, w, h = cv2.boundingRect(pts.astype(int))
                rect_area = float(w * h) if w > 0 and h > 0 else 0.0
                fill_ratio = (area / rect_area) if rect_area > 0 else 0.0
                area_ratio = area / img_area if img_area > 0 else 0.0

                # 単純な信頼度判定: 面積が十分であり（画面に対して小さすぎない）、
                # バウンディング矩形に比較してポリゴンが極端に細長/歪んでいないこと
                if area_ratio >= self.aruco_area_ratio_threshold and fill_ratio >= self.aruco_fill_threshold:
                    filtered_corners.append(pts.reshape(1, -1, 2))
                    filtered_ids.append([cid])

        # aruco_detected はフィルタ後の結果を見る
        aruco_detected = len(filtered_ids) > 0
        if aruco_detected:
            # 表示用に OpenCV の drawDetectedMarkers が期待する形に戻す
            try:
                fc = [c.astype(float) for c in filtered_corners]
                fid = cv2.UMat(cv2.UMat(np.array(filtered_ids))
                               ).get() if False else None
            except Exception:
                # 最小限: use filtered_corners and filtered_ids directly
                pass
            # 直接描画できる形にする
            try:
                # filtered_corners は list of (1,4,2) になっているのでそのまま渡す
                aruco.drawDetectedMarkers(
                    frame, filtered_corners, np.array(filtered_ids))
            except Exception:
                # fallback: draw original markers for visualization if conversion fails
                try:
                    aruco.drawDetectedMarkers(frame, corners, ids)
                except Exception:
                    pass

        # マーカーの検出状態に応じて自動撮影タイマーを制御する
        # - マーカーが新たに検出されたら 単発タイマーで自動撮影を行う
        # - マーカーが消えたら 保留中の自動撮影をキャンセルする
        if aruco_detected and not self._last_aruco_detected and self.timer.isActive():
            if not self.aruco_auto_timer.isActive():
                self.aruco_auto_timer.start(self.capture_delay_ms)
        elif aruco_detected:
            pass
        else:
            if self.aruco_auto_timer.isActive():
                self.aruco_auto_timer.stop()

        # 次フレーム用に検出状態を保持
        self._last_aruco_detected = aruco_detected

        # convert to RGB QImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line,
                            QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        # scale to label size
        pix = pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

    def auto_white_balance(self, image, corners):
        """
        ArUcoマーカーを6×6グリッドに分割し、各グリッドの白色と黒色の中央値を基準に
        画像全体のホワイトバランスを自動調整する（精度向上版）
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

    def calculate_marker_rotation(self, corners):
        """
        ArUcoマーカーの回転角度を計算する
        cornersは (1, 4, 2) の形状で、4つの角の座標を持つ
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

    def correct_rotation(self, image, angle_deg):
        """
        画像を最も近い直角（0°, 90°, 180°, 270°）に補正する
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

    def take_picture(self):
        if self.current_frame is None:
            return

        # 検出されたマーカーIDを取得
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        # マーカーが検出されていない場合
        if ids is None or len(ids) == 0:
            QtWidgets.QMessageBox.warning(
                self, "警告", "ArUcoマーカーが検出されていません。")
            return

        # 最初に検出されたマーカーIDを使用
        marker_id = str(ids[0][0])

        # マーカーIDに対応する教科名を取得
        subject_name = self.subject_mappings.get(marker_id, "未分類")

        # 教科ごとのフォルダを作成
        subject_dir = os.path.join(self.captures_dir, subject_name)
        os.makedirs(subject_dir, exist_ok=True)

        # まず回転補正を実行
        marker_angle = self.calculate_marker_rotation(corners)
        rotated_frame, rotation_applied = self.correct_rotation(self.current_frame, marker_angle)

        # 回転後の画像に対してホワイトバランス補正を適用
        if self.white_balance_enabled:
            # 回転後の画像でマーカーを再検出
            gray_rotated = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
            corners_rotated, ids_rotated, _ = self.detector.detectMarkers(gray_rotated)

            if corners_rotated is not None and len(corners_rotated) > 0:
                corrected_frame, viz_info, white_bgr, black_bgr = self.auto_white_balance(rotated_frame, corners_rotated)
            else:
                # マーカーが検出できない場合は回転後の画像をそのまま使用
                corrected_frame = rotated_frame
                viz_info, white_bgr, black_bgr = None, None, None
        else:
            corrected_frame = rotated_frame
            viz_info, white_bgr, black_bgr = None, None, None

        # ファイル名を生成して保存
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(subject_dir, f'capture_{ts}.png')

        # 最高品質で保存（PNG形式、圧縮レベル0-9、0が最高品質）
        cv2.imwrite(filename, corrected_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # デバッグモードの場合、グリッド付きの画像も保存
        if self.debug_mode and viz_info is not None:
            debug_frame = corrected_frame.copy()
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

            # デバッグ画像を保存
            debug_filename = os.path.join(subject_dir, f'capture_{ts}_debug.png')
            cv2.imwrite(debug_filename, debug_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # show the saved image in the video_label
        try:
            # load with QImage for display
            image = QtGui.QImage(filename)
            pix = QtGui.QPixmap.fromImage(image)
            pix = pix.scaled(self.video_label.size(),
                             QtCore.Qt.KeepAspectRatio)
            self.video_label.setPixmap(pix)
        except Exception:
            rgb = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line,
                                QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)
            pix = pix.scaled(self.video_label.size(),
                             QtCore.Qt.KeepAspectRatio)
            self.video_label.setPixmap(pix)

        # トースト通知で保存完了を表示（2秒後に自動で消える）
        rotation_info = f"\n回転補正: {rotation_applied:.1f}度" if abs(rotation_applied) >= 1.0 else ""
        toast_msg = f"教科: {subject_name}\nマーカーID: {marker_id}{rotation_info}\n保存完了"
        toast = ToastNotification(toast_msg, self, duration=4000)
        toast.show()

        # run OCR on the saved image (in background)
        worker = OCRWorker(frame=None, image_path=filename, parent=self)
        worker.finished.connect(self.on_ocr_result)
        worker.start()

        # stop updating from camera until user resumes
        if self.timer.isActive():
            self.timer.stop()

    def run_ocr(self):
        # Run OCR on the currently displayed frame (prefers current_frame if camera is active)
        if self.current_frame is not None and self.timer.isActive():
            src_frame = self.current_frame.copy()
            worker = OCRWorker(frame=src_frame, parent=self)
            worker.finished.connect(self.on_ocr_result)
            worker.start()
            return

        # If camera is paused (showing a saved image), we don't have current_frame to use.
        # In that case, attempt to OCR the last saved file if available by checking captures dir.
        files = sorted([f for f in os.listdir(self.captures_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not files:
            QtWidgets.QMessageBox.information(
                self, "OCR", "No captured image to run OCR on.")
            return
        latest = os.path.join(self.captures_dir, files[-1])
        worker = OCRWorker(image_path=latest, parent=self)
        worker.finished.connect(self.on_ocr_result)
        worker.start()

    def resume_camera(self):
        # restart live feed
        if not self.timer.isActive():
            self.timer.start(30)

    def on_ocr_result(self, text):
        self.ocr_output.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] {text}")

    def closeEvent(self, event):
        self.timer.stop()
        # stop ocr timer if present (older versions may have created it)
        if getattr(self, 'ocr_timer', None):
            try:
                self.ocr_timer.stop()
            except Exception:
                pass
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)

    # コマンドライン引数からデバッグモードを取得
    debug_mode = '--debug' in sys.argv or '-d' in sys.argv

    win = CameraWindow(debug_mode=debug_mode)

    if debug_mode:
        print("デバッグモードで起動しました。グリッド付き画像も保存されます。")

    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

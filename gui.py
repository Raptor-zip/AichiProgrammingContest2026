import os
import sys

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aruco + OCR Camera")
        self.resize(1000, 700)

        # captures directory
        self.captures_dir = os.path.join(os.path.dirname(__file__), 'captures')
        os.makedirs(self.captures_dir, exist_ok=True)

        # Video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Cannot open camera")
            sys.exit(1)

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

        # 撮影ボタン。Aruco が検出されていない場合は無効にする（後で update_frame で制御）
        self.capture_btn = QtWidgets.QPushButton("Take Picture")
        self.capture_btn.clicked.connect(self.take_picture)
        controls.addWidget(self.capture_btn)

        # 初期状態では撮影ボタンを無効化（Aruco 検出があって初めて撮れるようにする）
        self.capture_btn.setEnabled(False)

        # ArUco 検出をトリガーに自動撮影するための単発タイマー
        # マーカーを検出したらこのタイマーを start して一定時間（capture_delay_ms）後に撮影する
        self.capture_delay_ms = 800  # ミリ秒
        self.aruco_auto_timer = QtCore.QTimer(self)
        self.aruco_auto_timer.setSingleShot(True)
        self.aruco_auto_timer.timeout.connect(self.take_picture)
        # 前フレームの検出状態を保持して、状態遷移でタイマーを開始/停止する
        self._last_aruco_detected = False

        # 手動で OCR を実行するボタン（ライブ中のフレーム、または最後に保存した画像に対して実行）
        self.ocr_btn = QtWidgets.QPushButton("Run OCR Now")
        self.ocr_btn.clicked.connect(self.run_ocr)
        controls.addWidget(self.ocr_btn)

        spacer = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        controls.addItem(spacer)

        self.quit_btn = QtWidgets.QPushButton("Quit")
        self.quit_btn.clicked.connect(self.close)
        controls.addWidget(self.quit_btn)

        # 撮影後に一時停止したライブフィードを再開するボタン
        self.resume_btn = QtWidgets.QPushButton("Resume Camera")
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

        # マーカーの検出状態に応じて撮影ボタンと自動撮影タイマーを制御する
        # - マーカーが新たに検出されたら capture_btn を有効化し、単発タイマーで自動撮影を行う
        # - マーカーが消えたら capture_btn は無効化し、保留中の自動撮影をキャンセルする
        if aruco_detected and not self._last_aruco_detected and self.timer.isActive():
            self.capture_btn.setEnabled(True)
            if not self.aruco_auto_timer.isActive():
                self.aruco_auto_timer.start(self.capture_delay_ms)
        elif aruco_detected:
            # 連続検出状態ではボタンは有効のままにする
            self.capture_btn.setEnabled(True)
        else:
            # マーカーがない場合は撮影不可にして、もし自動タイマーが動いていれば止める
            self.capture_btn.setEnabled(False)
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

    def take_picture(self):
        if self.current_frame is None:
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.captures_dir, f'capture_{ts}.png')
        cv2.imwrite(filename, self.current_frame)
        # show the saved image in the video_label
        try:
            # load with QImage for display
            image = QtGui.QImage(filename)
            pix = QtGui.QPixmap.fromImage(image)
            pix = pix.scaled(self.video_label.size(),
                             QtCore.Qt.KeepAspectRatio)
            self.video_label.setPixmap(pix)
        except Exception:
            # fallback: convert from numpy frame
            rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line,
                                QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)
            pix = pix.scaled(self.video_label.size(),
                             QtCore.Qt.KeepAspectRatio)
            self.video_label.setPixmap(pix)

        QtWidgets.QMessageBox.information(
            self, "Saved", f"Saved to {filename}")

        # run OCR on the saved image (in background)
        self.ocr_btn.setEnabled(False)
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
            self.ocr_btn.setEnabled(False)
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
        self.ocr_btn.setEnabled(False)
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
        self.ocr_btn.setEnabled(True)

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
    win = CameraWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

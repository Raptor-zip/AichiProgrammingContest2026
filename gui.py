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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aruco + OCR Camera")
        self.resize(1000, 700)

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

        # ArUco 検出をトリガーに自動撮影するための単発タイマー
        # マーカーを検出したらこのタイマーを start して一定時間（capture_delay_ms）後に撮影する
        self.capture_delay_ms = 800  # ミリ秒
        self.aruco_auto_timer = QtCore.QTimer(self)
        self.aruco_auto_timer.setSingleShot(True)
        self.aruco_auto_timer.timeout.connect(self.take_picture)
        # 前フレームの検出状態を保持して、状態遷移でタイマーを開始/停止する
        self._last_aruco_detected = False

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

        # ファイル名を生成して保存
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(subject_dir, f'capture_{ts}.png')

        # 最高品質で保存（PNG形式、圧縮レベル0-9、0が最高品質）
        cv2.imwrite(filename, self.current_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

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

        # トースト通知で保存完了を表示（2秒後に自動で消える）
        toast_msg = f"教科: {subject_name}\nマーカーID: {marker_id}\n保存完了"
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
    win = CameraWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

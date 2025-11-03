import json
import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING
import cv2
import cv2.aruco as aruco
import numpy as np
import requests

if TYPE_CHECKING:
    import yomitoku.schemas

# Ensure Qt uses PySide6's plugins rather than OpenCV's bundled plugins which can cause
# "Could not load the Qt platform plugin 'xcb'" errors. We set QT_PLUGIN_PATH to PySide6
# package plugins directory when possible.
try:
    import PySide6
    from PySide6 import QtCore, QtGui, QtWidgets

    pyside_pkg_dir = os.path.dirname(PySide6.__file__)
    pyside_plugins = os.path.join(pyside_pkg_dir, "plugins")
    # Prepend to QT_PLUGIN_PATH so Qt finds PySide6 plugins first
    existing = os.environ.get("QT_PLUGIN_PATH", "")
    if pyside_plugins and pyside_plugins not in existing:
        os.environ["QT_PLUGIN_PATH"] = pyside_plugins + (
            os.pathsep + existing if existing else ""
        )
except Exception:
    # If PySide6 import fails, re-raise so the error is visible
    raise

from chatgpt import AIProcessingDialog
from config_loader import get_config
from image_processing import (
    auto_white_balance,
    calculate_marker_rotation,
    correct_rotation,
    draw_debug_grid,
    perspective_transform_from_marker,
)
from ocr_worker import YomiTokuWorker
from ui_components import SubjectSettingsDialog, ToastNotification


class CameraWindow(QtWidgets.QMainWindow):
    def __init__(self, debug_mode=False):
        super().__init__()

        # 設定を読み込む
        self.config = get_config()

        self.setWindowTitle("Aruco + OCR Camera")
        self.resize(self.config.get_window_width(),
                    self.config.get_window_height())

        # ウィンドウアイコンを設定
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))

        # モダンなデザインのスタイルシートを適用
        self.setStyleSheet(
            """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a2e, stop:1 #16213e);
            }
            QLabel#videoLabel {
                background-color: #0f3460;
                border-radius: 12px;
                border: 2px solid #533483;
                padding: 8px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #533483, stop:1 #3d2564);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: normal;
                min-width: 100px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6b4397, stop:1 #533483);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3d2564, stop:1 #2d1a4c);
                padding-top: 14px;
            }
            QPushButton#settingsButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e94560, stop:1 #c42847);
            }
            QPushButton#settingsButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff5577, stop:1 #e94560);
            }
            QPushButton#quitButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #757575, stop:1 #5a5a5a);
            }
            QPushButton#quitButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #8a8a8a, stop:1 #707070);
            }
            QPushButton#wbToggleOn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #388E3C);
            }
            QPushButton#wbToggleOn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #66BB6A, stop:1 #4CAF50);
            }
            QPushButton#wbToggleOff {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #757575, stop:1 #5a5a5a);
            }
            QPushButton#wbToggleOff:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #8a8a8a, stop:1 #707070);
            }
            QPushButton#resumeButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
            }
            QPushButton#resumeButton:hover {
                background: #4CAF50;
            }

            QPushButton#aiButton {
                background: #9C27B0;
            }
            QPushButton#aiButton:hover {
                background: #7B1FA2;
            }
            QPushButton#aiButton:disabled {
                background: #666666;
                color: #999999;
            }
            QTextEdit {
                background-color: #16213e;
                color: #e0e0e0;
                border: 2px solid #533483;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                selection-background-color: #533483;
            }
            QTextEdit:focus {
                border: 2px solid #6b4397;
            }
        """
        )

        # デバッグモード
        self.debug_mode = debug_mode

        # captures directory
        self.captures_dir = os.path.join(
            os.path.dirname(__file__), self.config.get_captures_dir()
        )
        os.makedirs(self.captures_dir, exist_ok=True)

        # subject mappings JSON file
        self.settings_file = os.path.join(
            os.path.dirname(__file__), self.config.get_subject_mappings_file()
        )
        self.subject_mappings = self.load_subject_mappings()

        # Video capture: try network MJPEG stream first, but verify we can actually read a frame.
        # If the stream can't provide frames, fall back to the local camera (index 0).
        def try_open_capture(source, tries=3):
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                return None
            # quick read-check: attempt to read one frame (with small flush attempts)
            for _ in range(tries):
                ret, _ = cap.read()
                if ret:
                    return cap
            # no frames read -> treat as failure
            try:
                cap.release()
            except Exception:
                pass
            return None

        # 設定からカメラタイプとURLを取得
        self.cap = try_open_capture(
            self.config.get_network_video_url(),
            tries=self.config.get_network_retry_count(),
        )
        self.cap_type = "network"
        if self.cap is None:
            # try the default local camera
            self.cap = try_open_capture(
                self.config.get_local_device_index(),
                tries=self.config.get_network_retry_count(),
            )
            self.cap_type = "local"

        if self.cap is None:
            # show a user-friendly error and stop initialization
            QtWidgets.QMessageBox.critical(
                self,
                "エラー",
                "カメラを開くことができませんでした。ネットワークカメラとローカルカメラの両方を確認してください。",
                QtWidgets.QMessageBox.StandardButton.Ok,
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            # raise an exception so caller can handle it (or exit in main)
            raise RuntimeError("Failed to open any camera source")

        # Set buffer size to 1 to always get the latest frame and prevent latency buildup
        # This is critical when stream FPS > processing FPS (e.g., 60fps stream with 33fps timer)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.get_buffer_size())

        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        # self.cap.set(cv2.CAP_PROP_FPS, 30)

        # ArUco setup
        # 設定からArUco辞書タイプを取得
        dict_type_name = self.config.get_aruco_dict_type()
        dict_type = getattr(aruco, dict_type_name, aruco.DICT_4X4_50)
        self.aruco_dict = aruco.getPredefinedDictionary(dict_type)
        params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, params)

        # ArUco 信頼度フィルターの閾値（設定ファイルから読み込み）
        # - aruco_area_ratio_threshold: 画像面積に対するマーカー面積の比率（小さすぎるものを除外）
        # - aruco_fill_threshold: マーカー凸包に対する実際のポリゴン面積の充填率（歪み判定）
        self.aruco_area_ratio_threshold = self.config.get_aruco_area_ratio_threshold()
        self.aruco_fill_threshold = self.config.get_aruco_fill_threshold()

        # ----- UI セットアップ -----
        # メインウィジェットとレイアウトを作る
        # (ビデオ表示ラベル、コントロールボタン群、OCR 出力)
        # UI 部分は後ほど update_frame() で画像を QLabel に流し込みます
        # -----
        # メインのキャプチャ画面を保持しておき、AI画面と切り替えられるようにする
        self.camera_central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.camera_central)

        # カメラ映像を表示する QLabel
        self.video_label = QtWidgets.QLabel()
        # 最小サイズを設定しておく（ウィンドウの縮小で潰れすぎないようにする）
        self.video_label.setMinimumSize(
            self.config.get_video_label_min_width(),
            self.config.get_video_label_min_height(),
        )
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(12)
        layout.addLayout(controls)

        # 教科設定ボタン
        self.settings_btn = QtWidgets.QPushButton("⚙️ 教科設定")
        self.settings_btn.setObjectName("settingsButton")
        self.settings_btn.clicked.connect(self.open_subject_settings)
        self.settings_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        controls.addWidget(self.settings_btn)

        # ホワイトバランス補正トグルボタン
        wb_default = self.config.get_white_balance_enabled_by_default()
        self.wb_toggle_btn = QtWidgets.QPushButton(
            "補正ON" if wb_default else "補正OFF"
        )
        self.wb_toggle_btn.setCheckable(True)
        self.wb_toggle_btn.setChecked(wb_default)
        self.wb_toggle_btn.clicked.connect(self.toggle_white_balance)
        self.settings_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        controls.addWidget(self.wb_toggle_btn)

        # ArUco 検出をトリガーに自動撮影するための単発タイマー
        # マーカーを検出したらこのタイマーを start して一定時間（capture_delay_ms）後に撮影する
        self.capture_delay_ms = self.config.get_aruco_auto_capture_delay_ms()
        self.aruco_auto_timer = QtCore.QTimer(self)
        self.aruco_auto_timer.setSingleShot(True)
        self.aruco_auto_timer.timeout.connect(self.take_picture)
        # 前フレームの検出状態を保持して、状態遷移でタイマーを開始/停止する
        self._last_aruco_detected = False

        # ホワイトバランス補正のON/OFF切り替えフラグ
        self.white_balance_enabled = wb_default

        spacer = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )

        controls.addItem(spacer)

        # 撮影後に一時停止したライブフィードを再開するボタン
        self.resume_btn = QtWidgets.QPushButton("📷 撮影再開")
        self.resume_btn.setObjectName("resumeButton")
        self.resume_btn.clicked.connect(self.resume_camera)
        self.resume_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        controls.addWidget(self.resume_btn)

        # OCR の結果などを表示するテキスト領域
        self.ocr_output = QtWidgets.QTextEdit()
        self.ocr_output.setReadOnly(True)
        self.ocr_output.setMaximumHeight(
            self.config.get_ocr_output_max_height())
        self.ocr_output.setPlaceholderText("OCR結果がここに表示されます...")
        layout.addWidget(self.ocr_output)

        # --- AI画面を作成 (画面遷移用) ---
        self.ai_page = QtWidgets.QWidget()
        ai_layout = QtWidgets.QVBoxLayout(self.ai_page)
        ai_layout.setContentsMargins(20, 20, 20, 20)
        ai_layout.setSpacing(12)

        ai_title = QtWidgets.QLabel("📚 過去の撮影履歴")
        ai_title.setStyleSheet(
            "font-size:18px; color: #e0e0e0; font-weight: bold;")
        ai_layout.addWidget(ai_title)

        # 水平分割: 左側にリスト、右側に詳細表示
        ai_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # 左側: 撮影履歴リスト
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 教科フィルター
        filter_layout = QtWidgets.QHBoxLayout()
        filter_label = QtWidgets.QLabel("教科で絞り込み:")
        filter_label.setStyleSheet("color: #e0e0e0;")
        filter_layout.addWidget(filter_label)

        self.subject_filter = QtWidgets.QComboBox()
        self.subject_filter.setStyleSheet("""
            QComboBox {
                background-color: #16213e;
                color: #e0e0e0;
                border: 2px solid #533483;
                border-radius: 4px;
                padding: 4px;
            }
            QComboBox:hover {
                border: 2px solid #6b4397;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #16213e;
                color: #e0e0e0;
                selection-background-color: #533483;
            }
        """)
        self.subject_filter.currentTextChanged.connect(
            self.filter_captures_by_subject)
        filter_layout.addWidget(self.subject_filter)
        filter_layout.addStretch()
        left_layout.addLayout(filter_layout)

        # 撮影履歴リスト
        self.capture_list = QtWidgets.QListWidget()
        self.capture_list.setStyleSheet("""
            QListWidget {
                background-color: #16213e;
                color: #e0e0e0;
                border: 2px solid #533483;
                border-radius: 8px;
                padding: 4px;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #533483;
            }
            QListWidget::item:selected {
                background-color: #533483;
            }
            QListWidget::item:hover {
                background-color: #3d2564;
            }
        """)
        self.capture_list.itemSelectionChanged.connect(
            self.on_capture_selected)
        left_layout.addWidget(self.capture_list)

        ai_splitter.addWidget(left_widget)

        # 右側: 詳細表示
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 画像プレビュー
        self.preview_label = QtWidgets.QLabel("画像を選択してください")
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #0f3460;
                border: 2px solid #533483;
                border-radius: 8px;
                color: #808080;
                min-height: 300px;
            }
        """)
        self.preview_label.setScaledContents(False)
        right_layout.addWidget(self.preview_label, stretch=3)

        # OCRテキスト表示
        self.ai_text_display = QtWidgets.QTextEdit()
        self.ai_text_display.setReadOnly(True)
        self.ai_text_display.setPlaceholderText("OCR結果がここに表示されます")
        right_layout.addWidget(self.ai_text_display, stretch=2)

        # AI処理ボタン
        ai_process_detail_btn = QtWidgets.QPushButton("🤖 AI処理を実行")
        ai_process_detail_btn.setObjectName("aiButton")
        ai_process_detail_btn.clicked.connect(self.open_ai_processing)
        ai_process_detail_btn.setCursor(
            QtCore.Qt.CursorShape.PointingHandCursor)
        right_layout.addWidget(ai_process_detail_btn)

        ai_splitter.addWidget(right_widget)
        ai_splitter.setStretchFactor(0, 1)
        ai_splitter.setStretchFactor(1, 2)

        ai_layout.addWidget(ai_splitter)

        # 中央ウィジェットは QStackedWidget を使って画面遷移を行う
        # （setCentralWidget でウィジェットを入れ替えると古いウィジェットが
        #  削除されてしまい、C++オブジェクトが既に削除される問題が発生するため）

        # メインコンテナ（上部にモード切替ボタン、下部にスタック）
        main_container = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 上部: モード切替バー
        mode_bar = QtWidgets.QWidget()
        mode_bar.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #533483, stop:1 #3d2564);
                border-bottom: 2px solid #6b4397;
            }
        """)
        mode_bar_layout = QtWidgets.QHBoxLayout(mode_bar)
        mode_bar_layout.setContentsMargins(10, 5, 10, 5)
        mode_bar_layout.setSpacing(10)

        # モード切替ボタン
        self.camera_mode_btn = QtWidgets.QPushButton("📷 撮影モード")
        self.camera_mode_btn.setCheckable(True)
        self.camera_mode_btn.setChecked(True)
        self.camera_mode_btn.clicked.connect(self.show_camera_page)
        self.camera_mode_btn.setCursor(
            QtCore.Qt.CursorShape.PointingHandCursor)
        self.camera_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #1976D2;
                border: 2px solid #64B5F6;
            }
            QPushButton:hover {
                background-color: #42A5F5;
            }
        """)
        mode_bar_layout.addWidget(self.camera_mode_btn)

        self.ai_mode_btn = QtWidgets.QPushButton("📚 AIモード")
        self.ai_mode_btn.setCheckable(True)
        self.ai_mode_btn.setChecked(False)
        self.ai_mode_btn.clicked.connect(self.show_ai_page)
        self.ai_mode_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.ai_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #7B1FA2;
                border: 2px solid #CE93D8;
            }
            QPushButton:hover {
                background-color: #AB47BC;
            }
        """)
        mode_bar_layout.addWidget(self.ai_mode_btn)
        # ボタンサイズを統一（高さと最小幅）
        uniform_height = 36
        uniform_min_width = 120
        try:
            self.camera_mode_btn.setFixedHeight(uniform_height)
            self.camera_mode_btn.setMinimumWidth(uniform_min_width)
            self.ai_mode_btn.setFixedHeight(uniform_height)
            self.ai_mode_btn.setMinimumWidth(uniform_min_width)
        except Exception:
            pass

        mode_bar_layout.addStretch()

        main_layout.addWidget(mode_bar)

        # スタック
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.camera_central)
        self.stack.addWidget(self.ai_page)
        main_layout.addWidget(self.stack)

        self.setCentralWidget(main_container)

        # 初期はカメラ画面を表示
        self.stack.setCurrentWidget(self.camera_central)

        # カメラからフレームを定期的に取得して表示するためのタイマー
        # 大体 30ms ごと（約33fps）で update_frame を呼ぶ
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self.config.get_frame_interval_ms())

        # remove automatic OCR timer: OCR should run after taking a picture
        # (the OCR button will still be available for manual runs)

        self.current_frame = None

        # Flag to pause camera feed display (but keep reading frames to maintain stream sync)
        self.camera_paused = False
        self.paused_display_frame = None

        # 最後のOCR結果を保存（AI処理用）
        self.last_ocr_text = ""
        self.last_subject_name = ""

        self.ocr_timer: QtCore.QTimer = QtCore.QTimer(self)

    def load_subject_mappings(self):
        """JSONファイルから教科マッピングを読み込む"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "警告",
                    f"設定ファイルの読み込みに失敗しました: {e}",
                    QtWidgets.QMessageBox.StandardButton.Ok,
                    QtWidgets.QMessageBox.StandardButton.Ok,
                )
                return {}
        return {}

    def save_subject_mappings(self):
        """教科マッピングをJSONファイルに保存"""
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.subject_mappings, f,
                          ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "エラー",
                f"設定ファイルの保存に失敗しました: {e}",
                QtWidgets.QMessageBox.StandardButton.Ok,
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            return False

    def open_subject_settings(self):
        """教科設定ダイアログを開く"""
        dialog = SubjectSettingsDialog(self.subject_mappings, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.subject_mappings = dialog.get_mappings()
            if self.save_subject_mappings():
                toast = ToastNotification("教科設定を保存しました", self, duration=4000)
                toast.show()

    def toggle_white_balance(self):
        """ホワイトバランス補正のON/OFFを切り替え"""
        self.white_balance_enabled = self.wb_toggle_btn.isChecked()
        if self.white_balance_enabled:
            self.wb_toggle_btn.setText("✓ 補正ON")
            self.wb_toggle_btn.setObjectName("wbToggleOn")
            toast = ToastNotification("ホワイトバランス補正: ON", self, duration=2000)
        else:
            self.wb_toggle_btn.setText("補正OFF")
            self.wb_toggle_btn.setObjectName("wbToggleOff")
            toast = ToastNotification("ホワイトバランス補正: OFF", self, duration=2000)
        # スタイルを再適用
        self.wb_toggle_btn.setStyle(self.wb_toggle_btn.style())
        toast.show()

    def update_frame(self):
        try:
            if self.cap is None:
                return
            ret, frame = self.cap.read()
            if not ret:
                return

            # Grab extra frames to flush buffer if stream FPS > timer FPS
            # This ensures we're always processing the most recent frame
            # and prevents latency buildup
            for _ in range(2):  # flush up to 2 old frames
                ret_flush, _ = self.cap.read()
                if not ret_flush:
                    break

        except Exception as e:
            # FFmpeg/MJPEG stream errors (e.g., "Stream ends prematurely", "overread")
            # These are often non-fatal, so we log and continue
            print(f"Warning: Frame read error: {e}")
            return

        # print(f"Captured frame size: {frame.shape[1]}x{frame.shape[0]}")

        # keep original BGR for saving/ocr
        self.current_frame = frame.copy()

        # If camera is paused, show the paused frame instead but keep reading to maintain stream sync
        if self.camera_paused:
            if self.paused_display_frame is not None:
                # Display the paused frame. The QLabel may have been deleted
                # (e.g. due to central widget swap), so guard against RuntimeError.
                try:
                    rgb = cv2.cvtColor(
                        self.paused_display_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    bytes_per_line = ch * w
                    qimg = QtGui.QImage(
                        rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888
                    )
                    pix = QtGui.QPixmap.fromImage(qimg)
                    try:
                        label_size = self.video_label.size()
                    except RuntimeError:
                        return
                    if not pix.isNull():
                        pix = pix.scaled(
                            label_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio
                        )
                        try:
                            self.video_label.setPixmap(pix)
                        except RuntimeError:
                            return
                except RuntimeError:
                    # Underlying Qt object was deleted; nothing to do
                    return
            return

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
            for ci, cid in zip(
                corners, ids.flatten() if hasattr(ids, "flatten") else ids
            ):
                # corners の形状は (1,4,2) や (4,2) のことがある
                pts = ci.reshape(-1, 2)

                # polygon 面積を計算
                area = abs(cv2.contourArea(pts))
                # bounding rect と面積の比（詰まり具合）
                x, y, w, h = cv2.boundingRect(pts.astype(int))
                rect_area = float(w * h) if w > 0 and h > 0 else 0.0
                fill_ratio = (area / rect_area) if rect_area > 0 else 0.0
                area_ratio = area / img_area if img_area > 0 else 0.0

                # 単純な信頼度判定: 面積が十分であり（画面に対して小さすぎない）、
                # バウンディング矩形に比較してポリゴンが極端に細長/歪んでいないこと
                if (
                    area_ratio >= self.aruco_area_ratio_threshold
                    and fill_ratio >= self.aruco_fill_threshold
                ):
                    filtered_corners.append(pts.reshape(1, -1, 2))
                    filtered_ids.append([cid])

        # aruco_detected はフィルタ後の結果を見る
        aruco_detected = len(filtered_ids) > 0
        if aruco_detected:
            # 表示用に OpenCV の drawDetectedMarkers が期待する形に戻す
            try:
                fc = [c.astype(float) for c in filtered_corners]
                fid = (
                    cv2.UMat(cv2.UMat(np.array(filtered_ids))
                             ).get() if False else None
                )
            except Exception:
                # 最小限: use filtered_corners and filtered_ids directly
                pass
            # 直接描画できる形にする
            try:
                # filtered_corners は list of (1,4,2) になっているのでそのまま渡す
                aruco.drawDetectedMarkers(
                    frame, filtered_corners, np.array(filtered_ids)
                )
            except Exception:
                # fallback: draw original markers for visualization if conversion fails
                try:
                    aruco.drawDetectedMarkers(frame, corners, ids)
                except Exception:
                    pass

        # マーカーの検出状態に応じて自動撮影タイマーを制御する
        # - マーカーが新たに検出されたら 単発タイマーで自動撮影を行う
        # - マーカーが消えたら 保留中の自動撮影をキャンセルする
        # Note: only start auto-capture if camera is not paused
        if aruco_detected and not self._last_aruco_detected and not self.camera_paused:
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
        qimg = QtGui.QImage(
            rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888
        )
        pix = QtGui.QPixmap.fromImage(qimg)
        # scale to label size — ensure label still exists
        try:
            label_size = self.video_label.size()
        except RuntimeError:
            # QLabel deleted (central widget swapped) — skip updating
            return
        if not pix.isNull():
            try:
                pix = pix.scaled(
                    label_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                self.video_label.setPixmap(pix)
            except RuntimeError:
                return

    def take_picture(self):
        if self.cap_type == "network":
            url = self.config.get_network_photo_url()
            response = requests.get(url)
            if response.status_code != 200:
                print("画像を取得できませんでした")
                return
            # バイト列をNumPy配列に変換
            img_array = np.frombuffer(response.content, dtype=np.uint8)

            # OpenCVでデコード
            self.current_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            # ローカルカメラの場合は現在のフレームを使用
            if self.current_frame is None:
                print("現在のフレームがありません")
                return

        if self.current_frame is None:
            return

        # 検出されたマーカーIDを取得
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        # マーカーが検出されていない場合
        if ids is None or len(ids) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "警告",
                "ArUcoマーカーが検出されていません。",
                QtWidgets.QMessageBox.StandardButton.Ok,
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            return

        # 最初に検出されたマーカーIDを使用
        marker_id = str(ids[0][0])

        # マーカーIDに対応する教科名を取得
        subject_name = self.subject_mappings.get(marker_id, "未分類")

        # 教科名を保存（AI処理用）
        self.last_subject_name = subject_name

        # 教科ごとのフォルダを作成
        subject_dir = os.path.join(self.captures_dir, subject_name)
        os.makedirs(subject_dir, exist_ok=True)

        # タイムスタンプを生成（全てのファイルで共通）
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ステップ0: 元の画像を保存（デバッグモード）
        if self.debug_mode:
            original_filename = os.path.join(
                subject_dir, f"capture_{ts}_0_original.jpg"
            )
            cv2.imwrite(original_filename, self.current_frame)

        # ステップ1: 台形補正（透視変換）を適用
        # マーカーの四隅の座標から画像全体を正面から見た状態に変換
        (
            perspective_frame,
            transform_matrix,
            output_size,
            corner_coords,
        ) = perspective_transform_from_marker(
            self.current_frame,
            np.asarray(corners),
            marker_size_mm=self.config.get_aruco_marker_size_mm(),
            output_dpi=self.config.get_aruco_output_dpi(),
        )

        # 台形補正が成功した場合はその画像を使用、失敗した場合は元の画像を使用
        if perspective_frame is not None:
            processing_frame = perspective_frame
            perspective_applied = True
            # デバッグモード: 台形補正後の画像を保存
            if self.debug_mode:
                perspective_filename = os.path.join(
                    subject_dir, f"capture_{ts}_1_perspective.jpg"
                )
                cv2.imwrite(perspective_filename, perspective_frame)
        else:
            processing_frame = self.current_frame.copy()
            perspective_applied = False

        # ステップ2: トリミングをやめ、検出された四角形を描画し、ハフ変換で直線も描画する
        # (処理用のフレームはそのまま使用し、描画はコピー上で行う)
        overlay = processing_frame.copy()

        # エッジ検出と輪郭検出による用紙の検出（トリミングは行わない）
        gray_trim = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
        kernel = tuple(self.config.get_gaussian_blur_kernel())
        blur = cv2.GaussianBlur(gray_trim, kernel, 0)
        edges = cv2.Canny(
            blur, self.config.get_canny_threshold1(), self.config.get_canny_threshold2()
        )

        # 輪郭検出
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        paper_corners = None
        for cnt in contours:
            approx = cv2.approxPolyDP(
                cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                paper_corners = approx.reshape(4, 2)
                break

        # 検出された四角形を描画
        if paper_corners is not None:
            pts = paper_corners.astype(int)
            # 線を描く
            cv2.polylines(overlay, [pts], isClosed=True,
                          color=(0, 255, 0), thickness=3)
            # 四隅に小さい円を描画
            for x, y in pts:
                cv2.circle(overlay, (int(x), int(y)), 6, (0, 255, 0), -1)

        # ハフ変換で直線検出
        try:
            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=self.config.get_hough_threshold(),
                minLineLength=self.config.get_hough_min_line_length(),
                maxLineGap=self.config.get_hough_max_line_gap(),
            )
        except Exception:
            lines = None

        if lines is not None:
            lines = np.asarray(lines, dtype=np.int32)
            for l in lines:
                x1, y1, x2, y2 = l[0]
                # 角度を計算（度単位）。atan2 の結果はラジアン。
                dy = y2 - y1
                dx = x2 - x1
                angle_rad = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle_rad)
                # 正規化して 0..180 の範囲にする（絶対角度）
                abs_angle = abs(angle_deg)
                if abs_angle > 180:
                    abs_angle = abs_angle % 180

                # 色分け: -5..5 度 (ほぼ水平) -> 赤, 85..95 度 (ほぼ垂直) -> 緑, それ以外 -> 灰
                # 角度は signed だがほとんど水平判定は abs(angle) <= 5 として扱う
                color = (192, 192, 192)  # デフォルト灰 (BGR)
                # 水平付近（-5〜5度）
                if -5.0 <= angle_deg <= 5.0:
                    color = (0, 0, 255)  # 赤 (B,G,R)
                # 垂直付近（85〜95度 または -95〜-85）
                elif 85.0 <= abs_angle <= 95.0:
                    color = (0, 255, 0)  # 緑

                # cv2.line(overlay, (x1, y1), (x2, y2), color, 2)

        # デバッグモード: 検出結果（四角 + 直線）を保存
        if self.debug_mode:
            detected_filename = os.path.join(
                subject_dir, f"capture_{ts}_2_detected.jpg"
            )
            cv2.imwrite(detected_filename, overlay)

        # 描画入りの画像をそのまま次ステップに渡す（トリミングはしない）
        processing_frame = overlay

        # ステップ3: 回転補正を実行（トリミング後の画像に対して）
        # トリミング後の画像でマーカーを再検出
        gray_trimmed = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
        corners_trimmed, ids_trimmed, _ = self.detector.detectMarkers(
            gray_trimmed)

        if corners_trimmed is not None and len(corners_trimmed) > 0:
            marker_angle = calculate_marker_rotation(corners_trimmed)
            rotated_frame, rotation_applied = correct_rotation(
                processing_frame, marker_angle
            )
        else:
            # マーカーが検出できない場合は回転補正をスキップ
            rotated_frame = processing_frame
            rotation_applied = 0.0

        if self.debug_mode:
            rotated_filename = os.path.join(
                subject_dir, f"capture_{ts}_3_rotated.jpg")
            cv2.imwrite(rotated_filename, rotated_frame)

        # ステップ4: 回転後の画像に対してホワイトバランス補正を適用
        if self.white_balance_enabled:
            # 回転後の画像でマーカーを再検出
            gray_rotated = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
            corners_rotated, ids_rotated, _ = self.detector.detectMarkers(
                gray_rotated)

            if corners_rotated is not None and len(corners_rotated) > 0:
                corrected_frame, viz_info, white_bgr, black_bgr = auto_white_balance(
                    rotated_frame, corners_rotated
                )
            else:
                # マーカーが検出できない場合は回転後の画像をそのまま使用
                corrected_frame = rotated_frame
                viz_info, white_bgr, black_bgr = None, None, None
        else:
            corrected_frame = rotated_frame
            viz_info, white_bgr, black_bgr = None, None, None

        if self.debug_mode:
            wb_filename = os.path.join(
                subject_dir, f"capture_{ts}_4_white_balance.jpg")
            cv2.imwrite(wb_filename, corrected_frame)

        # ファイル名を生成して最終画像を保存
        filename = os.path.join(subject_dir, f"capture_{ts}.png")
        cv2.imwrite(filename, corrected_frame)

        # デバッグモードの場合、グリッド付きの画像も保存
        if self.debug_mode and viz_info is not None:
            debug_frame = draw_debug_grid(corrected_frame, viz_info)
            debug_filename = os.path.join(
                subject_dir, f"capture_{ts}_5_grid.png")
            cv2.imwrite(debug_filename, debug_frame)

        # show the saved image in the video_label
        # Try to load with QImage first. However, QImage may return a null image
        # if the image is too large (Qt allocation limits). Detect that case and
        # fall back to OpenCV-based conversion which is more robust here.
        image = QtGui.QImage(filename)
        if image.isNull():
            # Fallback: convert with OpenCV -> QImage from buffer
            try:
                rgb = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QtGui.QImage(
                    rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888
                )
                pix = QtGui.QPixmap.fromImage(qimg)
            except Exception:
                pix = QtGui.QPixmap()
        else:
            pix = QtGui.QPixmap.fromImage(image)

        # Ensure pixmap is valid before scaling/setting to avoid QPixmap::scaled null warnings
        if not pix.isNull():
            pix = pix.scaled(
                self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio
            )
            self.video_label.setPixmap(pix)
            # Store the paused display frame
            self.paused_display_frame = corrected_frame.copy()
        else:
            # As a last resort, show nothing but log a warning
            print(
                "Warning: failed to create a valid QPixmap for display (image may be too large)."
            )

        perspective_info = "\n台形補正: 適用" if perspective_applied else ""
        rotation_info = (
            f"\n回転補正: {rotation_applied:.1f}度"
            if abs(rotation_applied) >= 1.0
            else ""
        )
        toast_msg = f"教科: {subject_name}\nマーカーID: {marker_id}{perspective_info}{rotation_info}\n保存完了"
        toast = ToastNotification(toast_msg, self, duration=4000)
        toast.show()

        # YomiTokuの処理を非同期で実行
        # タイムスタンプと教科ディレクトリを保存して、ワーカー完了時に使用
        self.current_ts = ts
        self.current_subject_dir = subject_dir
        self.current_corrected_frame = corrected_frame.copy()

        yomitoku_worker = YomiTokuWorker(frame=corrected_frame, parent=self)
        yomitoku_worker.finished.connect(self.on_yomitoku_result)
        yomitoku_worker.error.connect(self.on_yomitoku_error)
        yomitoku_worker.start()

        # pause camera feed by setting a flag instead of stopping timer
        # stopping/starting the timer causes MJPEG stream sync issues
        self.camera_paused = True

    def on_yomitoku_result(
        self, results: "yomitoku.schemas.OCRSchema", ocr_vis, layout_vis
    ):
        """YomiTokuの処理が完了した時のコールバック"""
        ts = self.current_ts
        subject_dir = self.current_subject_dir
        corrected_frame = self.current_corrected_frame

        # HTML形式で解析結果をエクスポート（results が存在する場合のみ）
        if results is not None:
            try:
                print(type(results))

                json_filename = os.path.join(
                    subject_dir, f"capture_{ts}_analysis.json")

                results.to_json(json_filename, img=corrected_frame)

                # wordsから各contentを改行区切りで結合
                ocr_text = "\n".join(word.content for word in results.words)
                print(ocr_text)

                self.last_ocr_text = ocr_text
            except Exception as e:
                print(f"Warning: failed to export analysis to HTML: {e}")

        # 可視化画像を保存（存在する場合のみ）
        if ocr_vis is not None:
            try:
                ocr_filename = os.path.join(
                    subject_dir, f"capture_{ts}_ocr_vis.jpg")
                cv2.imwrite(ocr_filename, ocr_vis)
            except Exception as e:
                print(f"Warning: failed to save ocr_vis: {e}")
        if layout_vis is not None:
            try:
                layout_filename = os.path.join(
                    subject_dir, f"capture_{ts}_layout_vis.jpg"
                )
                cv2.imwrite(layout_filename, layout_vis)
            except Exception as e:
                print(f"Warning: failed to save layout_vis: {e}")

    def on_yomitoku_error(self, error_msg):
        """YomiTokuの処理でエラーが発生した時のコールバック"""
        print(f"Warning: YomiToku processing failed: {error_msg}")

    def resume_camera(self):
        # resume live feed by clearing the paused flag
        self.camera_paused = False
        self.paused_display_frame = None

    def open_ai_processing(self):
        """AI処理ダイアログを開く"""
        if not self.last_ocr_text.strip():
            QtWidgets.QMessageBox.information(
                self,
                "情報",
                "処理するテキストがありません。\n先に画像を撮影してOCRを実行してください。",
            )
            return

        # AI処理ダイアログを表示
        dialog = AIProcessingDialog(
            self, self.last_ocr_text, self.last_subject_name)
        dialog.exec()

    def show_ai_page(self):
        """カメラ画面からAI画面へ遷移する。captures フォルダから履歴を読み込む。"""
        # カメラ表示を一時停止（フレーム読み取りは続ける）
        self.camera_paused = True
        # スタック内のページを切り替える
        try:
            self.stack.setCurrentWidget(self.ai_page)
            # ボタンの状態を更新
            self.camera_mode_btn.setChecked(False)
            self.ai_mode_btn.setChecked(True)
            # captures フォルダから履歴を読み込んで表示
            self.load_capture_history()
        except Exception as e:
            print(f"Warning: failed to switch to AI page: {e}")
            return

    def show_camera_page(self):
        """AI画面からカメラ画面へ戻す。"""
        # スタック内のページを切り替える
        try:
            self.stack.setCurrentWidget(self.camera_central)
            # ボタンの状態を更新
            self.camera_mode_btn.setChecked(True)
            self.ai_mode_btn.setChecked(False)
        except Exception as e:
            print(f"Warning: failed to switch to camera page: {e}")
            return
        # カメラ表示を再開
        self.camera_paused = False

    def load_capture_history(self):
        """captures フォルダから撮影履歴を読み込んでリスト表示する。"""
        self.capture_list.clear()
        self.subject_filter.clear()

        # 全教科を取得
        subjects = set(["すべて"])
        capture_items = []

        if not os.path.exists(self.captures_dir):
            return

        # 教科ごとのフォルダをスキャン
        for subject_name in os.listdir(self.captures_dir):
            subject_path = os.path.join(self.captures_dir, subject_name)
            if not os.path.isdir(subject_path):
                continue

            subjects.add(subject_name)

            # 各教科フォルダ内の画像ファイルをスキャン
            for filename in os.listdir(subject_path):
                if filename.endswith('.png') and not any(x in filename for x in ['_ocr_vis', '_layout_vis', '_grid', '_original', '_perspective', '_detected', '_rotated', '_white_balance']):
                    # タイムスタンプを抽出 (capture_20231103_123456.png)
                    if filename.startswith('capture_') and len(filename) > 20:
                        timestamp_str = filename[8:23]  # 20231103_123456
                        try:
                            # タイムスタンプをパース
                            dt = datetime.strptime(
                                timestamp_str, "%Y%m%d_%H%M%S")

                            # JSONファイルの存在確認
                            json_path = os.path.join(
                                subject_path, f"capture_{timestamp_str}_analysis.json")
                            has_ocr = os.path.exists(json_path)

                            capture_items.append({
                                'subject': subject_name,
                                'timestamp': dt,
                                'timestamp_str': timestamp_str,
                                'image_path': os.path.join(subject_path, filename),
                                'json_path': json_path if has_ocr else None,
                                'display_text': f"{subject_name} - {dt.strftime('%Y/%m/%d %H:%M:%S')}"
                            })
                        except ValueError:
                            continue

        # タイムスタンプでソート（新しい順）
        capture_items.sort(key=lambda x: x['timestamp'], reverse=True)

        # 教科フィルターを設定
        self.subject_filter.addItems(sorted(subjects))
        self.subject_filter.setCurrentText("すべて")

        # リストに追加（全データを保持）
        self.all_capture_items = capture_items
        self.filter_captures_by_subject("すべて")

    def filter_captures_by_subject(self, subject):
        """教科で撮影履歴をフィルタリング。"""
        self.capture_list.clear()

        if not hasattr(self, 'all_capture_items'):
            return

        for item in self.all_capture_items:
            if subject == "すべて" or item['subject'] == subject:
                list_item = QtWidgets.QListWidgetItem(item['display_text'])
                list_item.setData(QtCore.Qt.ItemDataRole.UserRole, item)
                self.capture_list.addItem(list_item)

    def on_capture_selected(self):
        """リストで選択された撮影履歴の詳細を表示。"""
        current_item = self.capture_list.currentItem()
        if not current_item:
            return

        item_data = current_item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not item_data:
            return

        # 画像を表示
        image_path = item_data['image_path']
        if os.path.exists(image_path):
            pixmap = QtGui.QPixmap(image_path)
            if not pixmap.isNull():
                # プレビューラベルのサイズに合わせてスケーリング
                scaled_pixmap = pixmap.scaled(
                    self.preview_label.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
                self.preview_label.setPixmap(scaled_pixmap)
            else:
                self.preview_label.setText("画像の読み込みに失敗しました")
        else:
            self.preview_label.setText("画像ファイルが見つかりません")

        # OCR結果を表示
        json_path = item_data.get('json_path')
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # words から content を抽出
                    if 'words' in data:
                        ocr_text = '\n'.join(word.get('content', '')
                                             for word in data['words'])
                        self.ai_text_display.setPlainText(ocr_text)
                        # AI処理用に保存
                        self.last_ocr_text = ocr_text
                        self.last_subject_name = item_data['subject']
                    else:
                        self.ai_text_display.setPlainText("OCR結果が見つかりません")
            except Exception as e:
                self.ai_text_display.setPlainText(f"OCR結果の読み込みエラー: {e}")
        else:
            self.ai_text_display.setPlainText("OCR結果がありません")
            self.last_ocr_text = ""

    def closeEvent(self, event):
        self.timer.stop()
        # stop ocr timer if present (older versions may have created it)
        if getattr(self, "ocr_timer", None):
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
    debug_mode = "--debug" in sys.argv or "-d" in sys.argv

    win = CameraWindow(debug_mode=debug_mode)

    if debug_mode:
        print("デバッグモードで起動しました。グリッド付き画像も保存されます。")

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

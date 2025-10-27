"""
UI関連のコンポーネント
- トースト通知
- 教科設定ダイアログ
"""

from PySide6 import QtCore, QtWidgets, QtGui


class ToastNotification(QtWidgets.QLabel):
    """
    自動的に消えるトースト通知ウィジェット
    """

    def __init__(self, message, parent=None, duration=2000):
        super().__init__(message, parent)
        self.setStyleSheet(
            """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(83, 52, 131, 240), stop:1 rgba(233, 69, 96, 240));
                color: white;
                padding: 18px 28px;
                border-radius: 12px;
                font-size: 15px;
                font-weight: normal;
            }
        """
        )
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMaximumWidth(450)
        # シャドウエフェクトを追加
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QtGui.QColor("black"))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

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
        self.resize(600, 500)

        # モダンなスタイルを適用
        self.setStyleSheet(
            """
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a2e, stop:1 #16213e);
            }
            QLabel {
                color: #e0e0e0;
                font-size: 14px;
                padding: 8px;
            }
            QTableWidget {
                background-color: #16213e;
                alternate-background-color: #1a2540;
                color: #e0e0e0;
                border: 2px solid #533483;
                border-radius: 8px;
                gridline-color: #533483;
                font-size: 13px;
                selection-background-color: #533483;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background-color: #6b4397;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #533483, stop:1 #3d2564);
                color: white;
                padding: 10px;
                border: none;
                border-right: 1px solid #16213e;
                font-weight: normal;
                font-size: 14px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #533483, stop:1 #3d2564);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: normal;
                min-width: 80px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6b4397, stop:1 #533483);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3d2564, stop:1 #2d1a4c);
            }
            QPushButton#addButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #388E3C);
            }
            QPushButton#addButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #66BB6A, stop:1 #4CAF50);
            }
            QPushButton#removeButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e94560, stop:1 #c42847);
            }
            QPushButton#removeButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff5577, stop:1 #e94560);
            }
            QPushButton#saveButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                min-width: 100px;
            }
            QPushButton#saveButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #42A5F5, stop:1 #2196F3);
            }
        """
        )

        # current_mappings は {id: subject_name} の辞書
        self.mappings = current_mappings.copy() if current_mappings else {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

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
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(True)
        layout.addWidget(self.table)

        # テーブルにデータを読み込む
        self.load_table_data()

        # ボタン群
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(12)

        add_btn = QtWidgets.QPushButton("➕ 行を追加")
        add_btn.setObjectName("addButton")
        add_btn.clicked.connect(self.add_row)
        add_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        btn_layout.addWidget(add_btn)

        remove_btn = QtWidgets.QPushButton("🗑️ 選択行を削除")
        remove_btn.setObjectName("removeButton")
        remove_btn.clicked.connect(self.remove_row)
        remove_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        btn_layout.addWidget(remove_btn)

        btn_layout.addStretch()

        cancel_btn = QtWidgets.QPushButton("キャンセル")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        btn_layout.addWidget(cancel_btn)

        save_btn = QtWidgets.QPushButton("💾 保存")
        save_btn.setObjectName("saveButton")
        save_btn.clicked.connect(self.save_and_accept)
        save_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def load_table_data(self):
        """現在のマッピングをテーブルに表示"""
        self.table.setRowCount(0)
        for marker_id, subject in sorted(
            self.mappings.items(), key=lambda x: int(x[0])
        ):
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

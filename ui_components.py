"""
UI関連のコンポーネント
- トースト通知
- 教科設定ダイアログ
"""

from PySide6 import QtCore, QtWidgets


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

"""
OCR処理を非同期で実行するワーカークラス
"""

import cv2
import pytesseract
from PIL import Image
from PySide6 import QtCore


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

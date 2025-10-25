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
        self.frame = frame.copy( ) if frame is not None else None
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

            # pil_imageをファイルに保存する
            pil_image.save("debug_ocr_input.jpg")

            import re
            import pandas as pd

            # 信頼度しきい値（0-100）
            confidence_threshold = 50

            # 日本語OCR（信頼度付き）
            data_jpn = pytesseract.image_to_data(
                pil_image, lang='jpn', output_type=pytesseract.Output.DICT)

            # 信頼度が閾値以上の単語のみを抽出
            filtered_text_jpn = []
            for i, conf in enumerate(data_jpn['conf']):
                if int(conf) >= confidence_threshold:
                    text_word = data_jpn['text'][i].strip()
                    if text_word:  # 空文字でない場合のみ追加
                        filtered_text_jpn.append(text_word)

            text_jpn = ' '.join(filtered_text_jpn)

            # 英語OCR（信頼度付き）
            data_eng = pytesseract.image_to_data(
                pil_image, lang='eng', output_type=pytesseract.Output.DICT)

            # 信頼度が閾値以上の単語のみを抽出
            filtered_text_eng = []
            for i, conf in enumerate(data_eng['conf']):
                if int(conf) >= confidence_threshold:
                    text_word = data_eng['text'][i].strip()
                    if text_word:  # 空文字でない場合のみ追加
                        filtered_text_eng.append(text_word)

            text_eng = ' '.join(filtered_text_eng)

            # 連続した空白を削除してクリーンアップ
            text_jpn = re.sub(r'\s+', ' ', text_jpn.strip())
            text_eng = re.sub(r'\s+', ' ', text_eng.strip())

            # 両方の結果を結合（信頼度情報も含める）
            jpn_count = len(filtered_text_jpn)
            eng_count = len(filtered_text_eng)
            text = f"日本語[{jpn_count}語]:{text_jpn} 英語[{eng_count}語]:{text_eng}".strip()

            # 再度連続した空白を削除
            text = re.sub(r'\s+', ' ', text)

            if not text:
                text = "(no text detected)"
        except Exception as e:
            # OCR 中に問題があればエラーメッセージを返す
            text = f"[OCR error] {e}"
        # 結果をシグナルで送出して UI スレッド側で受け取れるようにする
        self.finished.emit(text)

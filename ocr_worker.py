"""
OCR処理を非同期で実行するワーカークラス
"""

import cv2
from PIL import Image
from PySide6 import QtCore


class YomiTokuWorker(QtCore.QThread):
    """YomiToku DocumentAnalyzerを非同期実行するワーカー"""

    finished = QtCore.Signal(object, object, object)  # results, ocr_vis, layout_vis
    error = QtCore.Signal(str)

    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.frame = frame.copy( ) if frame is not None else None

    def run(self):
        if self.frame is None:
            self.error.emit("フレームがありません")
            return

        try:
            from yomitoku import DocumentAnalyzer

            # Try CUDA first, fallback to CPU
            analyzer = None
            try:
                analyzer = DocumentAnalyzer(visualize=True, device="cuda")
            except Exception:
                try:
                    analyzer = DocumentAnalyzer(visualize=True, device="cpu")
                except Exception as e:
                    self.error.emit(f"DocumentAnalyzerの初期化に失敗: {e}")
                    return

            if analyzer is not None:
                try:
                    results, ocr_vis, layout_vis = analyzer(self.frame)
                    self.finished.emit(results, ocr_vis, layout_vis)
                except Exception as e:
                    self.error.emit(f"DocumentAnalyzerの実行に失敗: {e}")
            else:
                self.error.emit("DocumentAnalyzerが利用できません")
        except Exception as e:
            self.error.emit(f"予期しないエラー: {e}")

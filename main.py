import cv2
import cv2.aruco as aruco
import pytesseract
from PIL import Image
import time

def main():
    cap = cv2.VideoCapture(0)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # OCR用のタイマー
    last_ocr_time = time.time()
    ocr_interval = 2.0  # 2秒間隔

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_any = False

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            detected_any = True
            # マーカー枠とIDを描画（辞書番号も表示）
            frame = aruco.drawDetectedMarkers(frame, corners, ids)

        if not detected_any:
            cv2.putText(frame, "No marker detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 2秒に1回OCRを実行
        current_time = time.time()
        if current_time - last_ocr_time >= ocr_interval:
            # OpenCVのフレームをPIL Imageに変換してOCRを実行
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            try:
                # 日本語と英語を検出
                text = pytesseract.image_to_string(pil_image, lang='jpn+eng')
                text = text.strip()
                
                if text:
                    print(f"[OCR] 検出されたテキスト: {text}")
                else:
                    print("[OCR] テキストが検出されませんでした")
            except Exception as e:
                print(f"[OCR] エラー: {e}")
            
            last_ocr_time = current_time

        cv2.imshow("Aruco Detection (try all dicts)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

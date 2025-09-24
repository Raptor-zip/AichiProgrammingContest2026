import cv2
import cv2.aruco as aruco

def main():
    cap = cv2.VideoCapture(0)

    parameters = aruco.DetectorParameters()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_any = False

        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            detected_any = True
            # マーカー枠とIDを描画（辞書番号も表示）
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            # cv2.putText(frame, f"Dict: {d}", (10, 30),
                        # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if not detected_any:
            cv2.putText(frame, "No marker detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Aruco Detection (try all dicts)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

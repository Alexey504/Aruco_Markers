import cv2
from pioneer_sdk import Camera
from pyzbar.pyzbar import decode
from pioneer_sdk import Pioneer
import time
import numpy as np
import json

# Подключение камеры дрона
ip = "10.1.100.109:8554"
endpoint = "pioneer_stream"

stream = cv2.VideoCapture(f'rtsp://{ip}/{endpoint}')

while not stream.isOpened():
    stream = cv2.VideoCapture(f'rtsp://{ip}/{endpoint}')
    time.sleep(1)

# Подключение дрона
# drone = Pioneer(ip="10.1.100.109", mavlink_port=5656)

if __name__ == "__main__":

    # Множество для распознанных кодов
    qr_set = set()
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = stream.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Детекция Qr кода
        for decoded in decode(gray):
            qr = decoded.data.decode("utf-8")
            qr_set.add(qr)
            points = np.array(list(map(lambda p: [p.x, p.y], decoded.polygon)))
            # Настройка положения надписи
            x1, y1 = points[0].astype(int)
            x2, y2 = points[2].astype(int)
            pos = ((x1 + x2) // 2, (y1 + y2) // 2)
            frame = cv2.polylines(frame, [points.astype(int)], True, (0, 255, 0), 8)
            # Отображение текста
            cv2.putText(frame,
                        qr,
                        pos,
                        font, 1,
                        (0, 0, 0),
                        2,)

        cv2.imshow("qr_reader", frame)

        if cv2.waitKey(1) == 27:  # Выход
            break
    # drone.land()
    # time.sleep(10)
    # drone.disarm()
    # drone.close_connection()
    cv2.destroyAllWindows()  # Close all opened openCV windows

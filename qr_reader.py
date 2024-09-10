import cv2
from pioneer_sdk import Camera
from pioneer_sdk import Pioneer


def drone_moving(x: float, y: float, z: float, yaw=0) -> None:
    """Передвижение дрона к заданным координатам"""

    drone.arm()
    drone.takeoff()
    drone.go_to_local_point(x, y, z, yaw)


# Подключение дрона
drone = Pioneer(ip='127.0.0.1', mavlink_port=8000)
# Создание детектора
qcd = cv2.QRCodeDetector()
# Множество для распознанных кодов
qr_set = set()


if __name__ == "__main__":
    # Подключение камеры
    camera = Camera(ip='127.0.0.1', port=18000)
    drone_moving(-0.05, 3.36, 1)  # перемещение дрона
    font = cv2.FONT_HERSHEY_SIMPLEX

    # проверка наличия позиции дрона
    while True:
        drone_pos = drone.get_local_position_lps()
        if drone_pos is not None:
            break

    while True:
        frame = camera.get_cv_frame()
        if frame is not None:
            # Детекция Qr кода
            ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)
            if ret_qr:
                # Настройка положения надписи
                x1, y1 = points[0][0].astype(int)
                x2, y2 = points[0][2].astype(int)
                pos = ((x1 + x2) // 4, (y1 + y2) // 4)
                for s, p in zip(decoded_info, points):
                    # Проверка на повторное считывание Qr кода
                    if s and s not in qr_set:
                        print(s)
                        qr_set.add(s)
                        color = (0, 255, 0)
                    elif s and s in qr_set:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    # Отрисовка рамки (опционально)
                    frame = cv2.polylines(frame, [p.astype(int)], True, color, 8)
                    x_, y_ = points[0][2].astype(int)
                    # Отображение текста
                    cv2.putText(frame,
                                s,
                                pos,
                                font, 1,
                                (255, 255, 255),
                                2,)
            cv2.imshow("video", frame)

        if cv2.waitKey(1) == 27:  # Выход
            break

    cv2.destroyAllWindows()  # Close all opened openCV windows

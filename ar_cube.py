import cv2
import numpy as np
from pioneer_sdk import Camera
from pioneer_sdk import Pioneer


def load_coefficients(path):
    """Загрузка матрицы камеры и матрицы искажений"""

    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    camera_matrix = cv_file.getNode("mtx").mat()
    dist_coeffs = cv_file.getNode("dist").mat()

    cv_file.release()
    return camera_matrix, dist_coeffs


# Подключение к дрону
drone = Pioneer(ip='127.0.0.1', mavlink_port=8000)
# Словарь аруко маркеров
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# Параметры для детекции маркеров
aruco_params = cv2.aruco.DetectorParameters()
# Детектор аруко маркеров
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
# Загрузка матрицы камеры и матрицы искажений
camera_matrix, dist_coeffs = load_coefficients("out_camera_data.yml")
# Длина стороны маркера в метрах
size_of_marker = 0.05
s = size_of_marker
axis = np.float32([[-s, -s, 0], [-s, s, 0], [s, s, 0], [s, -s, 0],
                   [-s, -s, s], [-s, s, s],
                   [s, s, s], [s, -s, s]])


def draw_cube(img, rvec, tvec, camera_matrix, dist_coeffs):
    """Добавление куба на экран"""

    # Проекция 3D точек в 2D изображение
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    # Отрисовка передних граней куба
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 2)
    # Отрисовка задних граней куба
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 2)
    # Отрисовка остальных рёбер
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 2)

    return img


def drone_moving(x: float, y: float, z: float, yaw=0) -> None:
    """Передвижение дрона к заданным координатам"""

    drone.arm()
    drone.takeoff()
    drone.go_to_local_point(x, y, z, yaw)


if __name__ == "__main__":

    # Подключение камеры
    camera = Camera(ip='127.0.0.1', port=18000)
    cnt = 0
    drone_moving(-0.95, 0.95, 1)  # перемещение дрона

    while True:
        frame = camera.get_cv_frame()

        # Преобразование изображения в серый формат
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Поиск маркеров ArUco
        corners, ids, rejected = aruco_detector.detectMarkers(frame)

        if ids is not None:
            # Оценка позы маркеров
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
            cnt += 0.1
            for rvec, tvec in zip(rvecs, tvecs):
                # Отображение осей координат на маркере (опционально)
                # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                # Отображение 3D объекта (куба)
                frame = draw_cube(frame, np.array([rvec[0][0], rvec[0][1] + cnt, rvec[0][2]]),
                                  tvec, camera_matrix, dist_coeffs)
        # Отображение изображения
        cv2.imshow('3D Object on ArUco Marker', frame)

        # Выход
        if cv2.waitKey(1) == 27:
            break

    # Освобождение ресурсов
    cv2.destroyAllWindows()

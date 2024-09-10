import cv2
from pioneer_sdk import Camera
from pioneer_sdk import Pioneer
import numpy as np
import numpy.typing as npt

# Словарь аруко маркеров
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# Параметры для детекции маркеров
aruco_params = cv2.aruco.DetectorParameters()
# Детектор аруко маркеров
# Требуется версия opencv 4.7.0
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def load_img(path):
    """Загрузка изображения"""

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return image


# Словарь изображений для уникальных id
img_dict = {
    199: load_img('img/cat2.png'),
    200: load_img('img/cat3.png'),
}

# Словарь для хранения загруженных изображений
loaded_imgs = {}


def drone_moving(x: float, y: float, z: float, yaw=0) -> None:
    """Передвижение дрона к заданным координатам"""

    drone = Pioneer(ip='127.0.0.1', mavlink_port=8000)
    drone.arm()
    drone.takeoff()
    drone.go_to_local_point(x, y, z, yaw)


def transform_image(frame: npt.NDArray[float], crns: tuple, m_ids: npt.NDArray[float]) -> npt.NDArray[float]:
    """
    Возвращает переданное изображение в перспективе.

    Args:
        frame (npt.NDArray[float]): Экран.
        crns (tuple): Углы всех маркеров на экране.
        m_ids (npt.NDArray[float]): Id всех маркеров.

    Returns:
        npt.NDArray[float]: Изображение в перспективе.
    """

    new_t_img = np.zeros((frame.shape[0], frame.shape[1], 4))
    m_ids = m_ids.ravel()
    # Перебор всех маркеров
    for i, corn in enumerate(zip(crns, m_ids)):
        # Добавление изображения
        img = loaded_imgs.setdefault(m_ids[i], img_dict.get(m_ids[i]))
        img_rows, img_cols, img_ch = img.shape
        # Углы маркеров
        pts1 = np.float32([[img_cols, img_rows], [0, img_rows],
                           [0, 0], [img_cols, 0]])
        pts2 = np.float32(crns[i].reshape(4, 2))
        # Матрица перспективы
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # Перевод изображения в перспективу
        t_img = cv2.warpPerspective(img, M, (frame.shape[1], frame.shape[0]))
        # Добавление изображения на экран
        new_t_img += t_img

    return new_t_img


def picture_on_marker(frame: npt.NDArray[float], trans_img: npt.NDArray[float]) -> npt.NDArray[float]:
    """
    Наложения картинки на экран.

    Args:
        frame (npt.NDArray[float]): Экран.
        trans_img (tuple): Изображение в перспективе.

    Returns:
        npt.NDArray[float]: Экран с изображениями.
    """

    # Настройка каналов
    alpha_channel = trans_img[:, :, 3] / 255
    overlay_colors = trans_img[:, :, :3]
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
    h, w = trans_img.shape[:2]
    # Наложение маски
    background_subsection = frame[0:h, 0:w]
    composite = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask
    frame[0:h, 0:w] = composite

    return frame


if __name__ == "__main__":

    # Подключение каммеры
    camera = Camera(ip='127.0.0.1', port=18000)
    # Передвижение дрона в заданные координаты
    drone_moving(-0.47, 1.2, 1)

    while True:
        cv_frame = camera.get_cv_frame()

        # Детекция маркеров
        corners, ids, rejected = aruco_detector.detectMarkers(cv_frame)
        # cv2.aruco.drawDetectedMarkers(cv_frame, corners, ids)

        if corners:
            # изменения изображения в перспективу
            transformed_img = transform_image(cv_frame, corners, ids)
            # Размещение картинки на экране
            frame_with_picture = picture_on_marker(cv_frame, transformed_img)

            cv2.imshow("video", frame_with_picture)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

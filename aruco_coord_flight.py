import sys
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import numpy as np
from pioneer_sdk import Camera
from pioneer_sdk import Pioneer
import threading
import queue
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


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

# Коэффициенты размеров ширины и высоты экрана
scale_x = 2.65 / 640
scale_y = 2.45 / 480

# Координаты маркеров по индексам
dict_aruco_pos = {}
# Начальные данные
x_data, y_data = [], []
# список для координат маркеров
point_list = []
# Очередь для потоков
data_queue = queue.Queue()
# Инициализация данных для графика
x_coords = []
y_coords = []
labels = []


def update(frame):
    """Функция для обновления данных графика"""

    if not data_queue.empty():
        # Новые координаты
        m_id, lst_coord = data_queue.get(block=True)
        new_x, new_y = lst_coord

        # Добавляем данные в списки
        x_coords.append(new_x)
        y_coords.append(new_y)
        labels.append(m_id)

        # Очищаем предыдущие данные с графика
        ax.clear()

        # Разделяем четные и нечетные маркеры
        even_x = [x_coords[i] for i in range(len(x_coords)) if labels[i] % 2 == 0]
        even_y = [y_coords[i] for i in range(len(y_coords)) if labels[i] % 2 == 0]
        odd_x = [x_coords[i] for i in range(len(x_coords)) if labels[i] % 2 != 0]
        odd_y = [y_coords[i] for i in range(len(y_coords)) if labels[i] % 2 != 0]

        # Отображаем точки на графике
        ax.scatter(even_x, even_y, color='blue', label='Четные маркеры')
        ax.scatter(odd_x, odd_y, color='red', label='Нечетные маркеры')

        # Координаты для каждой точки
        for i in range(len(x_coords)):
            ax.text(x_coords[i], y_coords[i] + 0.2, f'({x_coords[i]:.1f}, {y_coords[i]:.1f})', fontsize=9)

        # 1 вариант
        # Разделение по четным и нечетным маркерам
        if even_x and odd_x:
            even_points = np.array(list(zip(even_x, even_y)))
            odd_points = np.array(list(zip(odd_x, odd_y)))
            X = np.vstack((even_points, odd_points))
            y = np.hstack((np.zeros(even_points.shape[0]), np.ones(odd_points.shape[0])))

            model = SVC(kernel='rbf', C=1, gamma=0.5)
            model.fit(X, y)
            xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))

            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Отображаем границу решений
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='Paired')

        # 2 вариант
        # Разделение по группам (по умолчанию - пары)
        # if len(x_coords) >= 2:  # Должно быть достаточно точек для построения модели
        #     # k-NN классификатор
        #     X = np.column_stack((x_coords, y_coords))
        #     y = np.array(labels)
        #     knn = KNeighborsClassifier(n_neighbors=2)  # Используем 2 ближайших соседа
        #     knn.fit(X, y)
        #
        #     #  Сетка для построения границы решений
        #     xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
        #     Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        #     Z = Z.reshape(xx.shape)
        #
        #     # Отображаем границу решений
        #     ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

        # Настройки графика
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.legend()
        ax.set_title('Aruco coordinates')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')


def start_matplotlib():
    """Отображение координат на графике"""

    global ax
    # Создаем фигуру и оси
    fig, ax = plt.subplots()
    # Настройка анимации
    ani = FuncAnimation(fig, update, frames=100, interval=1000)

    plt.show()


def move_to_marker(corners_: tuple) -> (bool, int, int):
    """
    Передвижение дрона

        Args:
        corners_ (tuple): Углы всех маркеров на экране.

    Returns:
        (bool, sign y, sign yaw): Наличие маркера, знак скорости y, знак скорости поворота.
    """
    # Скорости дрона
    vx, vy, vz, vyaw = 0, 0, 0, 0

    # Определяем движение для дрона
    # Координаты центра экрана
    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2
    # Рассчитываем центр маркера
    cX = int((corners_[0][0][0][0] + corners_[0][0][1][0] + corners_[0][0][2][0] + corners_[0][0][3][0]) / 4)
    cY = int((corners_[0][0][0][1] + corners_[0][0][1][1] + corners_[0][0][2][1] + corners_[0][0][3][1]) / 4)

    # Движение дрона
    # Центр экрана должен совпасть с центром маркера
    if cX < frame_center_x - 20:
        vyaw = -0.4
    elif cX > frame_center_x + 20:
        vyaw = 0.4
    if cY < frame_center_y - 20:
        vy = 0.3
    elif cY > frame_center_y + 20:
        vy = -0.3
    drone.set_manual_speed(vx, vy, vz, vyaw)
    direct_y, direct_yaw = np.sign(vy), np.sign(vyaw)

    # Проверка центров
    if abs(cX - frame_center_x) <= 10 and abs(cY - frame_center_y) <= 10:
        print('point')
        drone.set_manual_speed(0, 0, 0, 0)
        return True, 1, 1

    return False, direct_y, direct_yaw


def check_boarder(pos: tuple) -> bool:
    """Проверка крайних положений дрона"""

    pos_x, pos_y, _ = pos
    if abs(pos_x) < 4.9 and abs(pos_y) < 4.9:
        return True
    return False


def save_position(corners_s: tuple, ids_s: npt.NDArray[float]) -> None:
    """
    Сохранение координат маркеров

    Args:
        corners_s (tuple): Углы всех маркеров на экране.
        ids_s (npt.NDArray[float]): Номера маркеров.
    """

    # проверка наличия позиции дрона
    while True:
        drone_pos = drone.get_local_position_lps()
        if drone_pos is not None:
            break

    for i, (corner, id_) in enumerate(zip(corners_s, ids_s)):

        # Рассчитываем центр маркера
        cX = int((corner[0][0][0] + corner[0][1][0] + corner[0][2][0] + corner[0][3][0]) / 4)
        cY = int((corner[0][0][1] + corner[0][1][1] + corner[0][2][1] + corner[0][3][1]) / 4)

        # Перенос начала координат экрана
        width, height = 320, 240
        cX = cX - width
        cY = (height * 2 - cY)

        while True:
            a = drone.get_yaw()
            if a is not None:
                break

        # Перевод из локальной системы координат в глобальную
        a = np.radians(a)
        # Матрица позиции дрона в глобальной системе координат
        vec_drone = np.matrix([drone_pos[0], drone_pos[1]])
        # Матрица позиции маркера в локальной системы координат
        vec_marker_camera = np.matrix([cX, cY])
        # Матрица масштаба
        scale_matrix = np.matrix([[scale_x, 0],
                                  [0, scale_y]])
        # Матрица поворота
        angle_matrix = np.matrix([[np.cos(a), -np.sin(a)],
                                 [np.sin(a), np.cos(a)]])
        # Глобальные координаты маркера
        coords = vec_drone + vec_marker_camera * scale_matrix * angle_matrix

        # Округление позиции
        round_pos = np.round(coords, 2).tolist()

        # Сохранение позиции
        if id_ not in dict_aruco_pos:
            dict_aruco_pos[id_] = [round_pos[0]]
            point_list.append(round_pos[0])
            data_queue.put((id_, round_pos[0]))


if __name__ == "__main__":

    # Подключение каммеры
    camera = Camera(ip='127.0.0.1', port=18000)

    # Процесс для работы графика
    graph = threading.Thread(target=start_matplotlib)
    graph.daemon = True
    graph.start()

    # Запуск дрона
    drone.arm()
    drone.takeoff()

    # Настройка движения дрона
    ids_markers = set()
    angle = 0
    last_y, last_yaw = 1, 1
    check = True

    while True:
        frame = camera.get_cv_frame()

        if frame is not None:
            # Детекция маркеров
            corners, ids, rejected = aruco_detector.detectMarkers(frame)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Проверка обнаруженного маркера или ранее проверенного маркера
            if corners and not ids_markers.issuperset(set(ids.flatten().tolist())):
                id_list = ids.flatten().tolist()
                # Передвижение дрона
                reached, last_y, last_yaw = move_to_marker(corners)
                if reached:
                    print('Marker found')
                    angle = 0
                    # Добавление маркеров в просмотренные
                    ids_markers.update(id_list)
                    # Сохранение координат маркера
                    save_position(corners, id_list)
            else:
                drone_pos = drone.get_local_position_lps()
                if drone_pos is not None:
                    check = check_boarder(drone_pos)
                if check:
                    # Движение для поиска маркеров
                    drone.set_manual_speed(0, 0.2, 0, last_yaw * 0.3)
                    angle += 0.2
                    if angle >= 25:
                        last_yaw *= -1
                        angle = 0
                else:
                    drone.set_manual_speed(0, 0.2, 0, last_yaw * 0.4)

        cv2.imshow("video", frame)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()  # Close all opened openCV windows
    sys.exit()

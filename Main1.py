import sys
import os
import cv2
import face_recognition
import numpy as np
from scipy.spatial import distance as dist
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from collections import Counter
import logging
import json


# Настройка логирования
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

file_handler = FlushFileHandler('main1_log.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        file_handler,
                        logging.StreamHandler(sys.stdout)
                    ])

logging.info("Программа запущена")


class GroupManager:
    """Класс для управления группами лиц."""
    def __init__(self, file_path='groups.json'):
        self.file_path = file_path
        self.groups = {}
        self.load_from_file()  # Загружаем данные из файла при инициализации

    def create_group(self, group_name):
        """Создание новой группы."""
        if group_name in self.groups:
            logging.warning(f"Группа {group_name} уже существует.")
        else:
            self.groups[group_name] = []
            logging.info(f"Группа {group_name} создана.")
            self.save_to_file()  # Сохранение изменений

    def delete_group(self, group_name):
        """Удаление группы."""
        if group_name in self.groups:
            del self.groups[group_name]
            logging.info(f"Группа {group_name} удалена.")
            self.save_to_file()  # Сохранение изменений
        else:
            logging.warning(f"Группа {group_name} не найдена.")

    def rename_group(self, old_name, new_name):
        """Переименование группы."""
        if old_name in self.groups:
            self.groups[new_name] = self.groups.pop(old_name)
            logging.info(f"Группа {old_name} переименована в {new_name}.")
            self.save_to_file()  # Сохранение изменений
        else:
            logging.warning(f"Группа {old_name} не найдена.")

    def add_face_to_group(self, face_name, group_name):
        """Добавление лица в группу."""
        if group_name in self.groups:
            self.groups[group_name].append(face_name)
            logging.info(f"Лицо {face_name} добавлено в группу {group_name}.")
            self.save_to_file()  # Сохранение изменений
        else:
            logging.warning(f"Группа {group_name} не найдена.")

    def get_group_of_face(self, face_name):
        """Получение группы, к которой принадлежит лицо."""
        for group_name, members in self.groups.items():
            if face_name in members:
                return group_name
        return None

    def save_to_file(self):
        """Сохранение групп в JSON файл."""
        try:
            with open(self.file_path, 'w') as file:
                json.dump(self.groups, file)
            logging.info("Группы успешно сохранены в файл.")
        except IOError as e:
            logging.error(f"Ошибка при сохранении в файл: {e}")

    def load_from_file(self):
        """Загрузка групп из JSON файла."""
        try:
            with open(self.file_path, 'r') as file:
                self.groups = json.load(file)
            logging.info("Группы успешно загружены из файла.")
        except FileNotFoundError:
            logging.warning("Файл не найден, загрузка пропущена.")
        except json.JSONDecodeError as e:
            logging.error(f"Ошибка при чтении JSON файла: {e}")
        except IOError as e:
            logging.error(f"Ошибка при загрузке из файла: {e}")


def load_known_face_encodings(folder_path):
    known_encodings = {}
    image_extensions = ['.jpg', '.jpeg', '.png']

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            known_image = face_recognition.load_image_file(file_path)
            encoding = face_recognition.face_encodings(known_image)
            if encoding:
                known_encodings[filename] = encoding[0]
                logging.info(f"Загружена кодировка для {filename}")
            else:
                logging.warning(f"Не удалось получить кодировку для {filename}")
    return known_encodings


def recognize_faces(unknown_encodings, known_encodings, group_manager):
    recognized_faces = []
    for unknown_encoding in unknown_encodings:
        name_found = "Unknown"
        for name, known_encoding in known_encodings.items():
            compare = face_recognition.compare_faces([known_encoding], unknown_encoding)
            if compare[0]:
                group = group_manager.get_group_of_face(name)
                if group:
                    name_found = f"{name} (Группа: {group})"
                else:
                    name_found = name
                logging.info(f"Распознано лицо: {name_found}")
                break
        recognized_faces.append(name_found)
    return recognized_faces


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def detect_head_movement(prev_landmarks, current_landmarks):
    if prev_landmarks is None:
        return False
    nose_prev = np.array(prev_landmarks['nose_bridge'][0])
    nose_curr = np.array(current_landmarks['nose_bridge'][0])
    distance = np.linalg.norm(nose_curr - nose_prev)
    logging.info(f"Движение головы: {distance > 2}")
    return distance > 2


def analyze_face_texture(img, face_location):
    top, right, bottom, left = face_location
    face_region = img[top:bottom, left:right]
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    logging.info(f"Текстура лица: {'живая' if laplacian_var > 100 else 'не живая'}")
    return laplacian_var > 100


class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("Ошибка: Камера не обнаружена.")
            logging.error("Ошибка: Камера не обнаружена.")
            return

        self.group_manager = GroupManager()
        self.known_faces = load_known_face_encodings('Images')

        self.frame_timer = QTimer(self)
        self.search_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_frame)
        self.search_timer.timeout.connect(self.finish_face_search)

        self.face_results = []
        self.blink_counter = 0
        self.movement_detected = False
        self.prev_landmarks = None
        self.texture_analysis_result = False
        self.EYE_AR_THRESH = 0.2
        self.MIN_BLINKS = 1

    def initUI(self):
        self.setWindowTitle('Face Recognition App')

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel('Нажмите "Начать", чтобы искать лица...', self)
        self.status_label.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton('Начать поиск', self)
        self.start_button.clicked.connect(self.start_face_search)

        self.quit_button = QPushButton('Выход', self)
        self.quit_button.clicked.connect(self.close_application)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.quit_button)

        self.setLayout(self.layout)

    def start_face_search(self):
        self.status_label.setText("Поиск лиц в процессе...")
        logging.info("Поиск лиц начат.")
        self.face_results.clear()
        self.blink_counter = 0
        self.movement_detected = False
        self.prev_landmarks = None
        self.texture_analysis_result = False
        self.frame_timer.start(30)
        self.search_timer.start(5000)

    def finish_face_search(self):
        self.frame_timer.stop()
        self.search_timer.stop()

        filtered_faces = [face for face in self.face_results if face != "Unknown"]

        if filtered_faces:
            most_common_face = Counter(filtered_faces).most_common(1)[0][0]
            if self.blink_counter < self.MIN_BLINKS and not self.movement_detected or not self.texture_analysis_result:
                self.status_label.setText(f'Распознано: {most_common_face}, но лицо не является живым.')
                logging.info(f'Распознано: {most_common_face}, но лицо не является живым.')
            else:
                self.status_label.setText(f'Распознано: {most_common_face}, лицо живое.')
                logging.info(f'Распознано: {most_common_face}, лицо живое.')
        else:
            self.status_label.setText('Распознано: Unknown или лицо не является живым')
            logging.info('Распознано: Unknown или лицо не является живым')

    @pyqtSlot()
    def update_frame(self):
        success, img = self.cap.read()

        if not success:
            self.status_label.setText("Ошибка при получении изображения")
            logging.error("Ошибка при получении изображения")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_landmarks_list = face_recognition.face_landmarks(img_rgb)
        face_locations = face_recognition.face_locations(img_rgb)

        if face_landmarks_list:
            for i, face_landmarks in enumerate(face_landmarks_list):
                if detect_head_movement(self.prev_landmarks, face_landmarks):
                    self.movement_detected = True
                self.prev_landmarks = face_landmarks

                left_eye = face_landmarks['left_eye']
                right_eye = face_landmarks['right_eye']

                left_eye_ratio = eye_aspect_ratio(left_eye)
                right_eye_ratio = eye_aspect_ratio(right_eye)

                avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

                if avg_eye_ratio < self.EYE_AR_THRESH:
                    self.blink_counter += 1

                self.texture_analysis_result = analyze_face_texture(img_rgb, face_locations[i])

        recognize = face_recognition.face_encodings(img_rgb)

        if recognize:
            results = recognize_faces(recognize, self.known_faces, self.group_manager)
            self.face_results.extend(results)
        else:
            self.face_results.append("Unknown")

        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_qt_format)
        self.video_label.setPixmap(pixmap)

    def close_application(self):
        self.frame_timer.stop()
        self.search_timer.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Приложение закрыто")
        self.close()
        logging.shutdown()


def main():
    app = QApplication(sys.argv)
    face_recognition_app = FaceRecognitionApp()
    face_recognition_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

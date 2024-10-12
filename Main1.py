import sys
import os
import cv2
import face_recognition
import numpy as np
from scipy.spatial import distance as dist
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


# Функция для расчета EAR (отношение сторон глаз)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Функция для анализа текстуры с различными порогами
def analyze_face_texture(img, face_location, texture_threshold):
    top, right, bottom, left = face_location
    face_region = img[top:bottom, left:right]
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    return laplacian_var > texture_threshold

# Функция для анализа бликов с различными порогами
def analyze_reflection(img, face_location, reflection_threshold):
    top, right, bottom, left = face_location
    face_region = img[top:bottom, left:right]
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    reflection_score = cv2.mean(gray_face)[0]
    return reflection_score < reflection_threshold

# Функция для детекции морганий с различными порогами EAR
def detect_blink(eye_landmarks, ear_threshold):
    ear = eye_aspect_ratio(eye_landmarks)
    return ear < ear_threshold

# Функция для детекции движения головы с различными порогами смещения
def detect_head_movement(prev_landmarks, current_landmarks, movement_threshold):
    if prev_landmarks is None:
        return False
    nose_prev = np.array(prev_landmarks['nose_bridge'][0])
    nose_curr = np.array(current_landmarks['nose_bridge'][0])
    distance = np.linalg.norm(nose_curr - nose_prev)
    return distance > movement_threshold

# Функция для определения "живого" лица с уникальными уровнями проверки живости
def is_live_face(img, face_location, face_landmarks, prev_landmarks=None, liveness_level=1):
    texture_result = reflection_result = blink_detected = head_movement_detected = True
    
    if liveness_level == 1:
        texture_result = analyze_face_texture(img, face_location, texture_threshold=50)  # Мягкий порог для текстуры
        logging.info(f"Результат анализа текстуры на уровне 1: {'живое' if texture_result else 'поддельное'}")
        
    if liveness_level == 2:
        texture_result = analyze_face_texture(img, face_location, texture_threshold=100)  # Строгий порог для текстуры
        logging.info(f"Результат анализа текстуры на уровне 2: {'живое' if texture_result else 'поддельное'}")
    
    if liveness_level == 3:
        reflection_result = analyze_reflection(img, face_location, reflection_threshold=180)  # Мягкий порог для бликов
        logging.info(f"Результат анализа бликов на уровне 3: {'естественное' if reflection_result else 'поддельное'}")
        
    if liveness_level == 4:
        reflection_result = analyze_reflection(img, face_location, reflection_threshold=150)  # Строгий порог для бликов
        logging.info(f"Результат анализа бликов на уровне 4: {'естественное' if reflection_result else 'поддельное'}")
    
    if liveness_level == 5:
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        blink_detected = detect_blink(left_eye, ear_threshold=0.3) or detect_blink(right_eye, ear_threshold=0.3)  # Мягкий порог EAR
        logging.info(f"Моргание на уровне 5: {'обнаружено' if blink_detected else 'не обнаружено'}")
    
    if liveness_level == 6:
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        blink_detected = detect_blink(left_eye, ear_threshold=0.2) or detect_blink(right_eye, ear_threshold=0.2)  # Строгий порог EAR
        logging.info(f"Моргание на уровне 6: {'обнаружено' if blink_detected else 'не обнаружено'}")
    
    if liveness_level == 7:
        head_movement_detected = detect_head_movement(prev_landmarks, face_landmarks, movement_threshold=3)  # Мягкий порог движения
        logging.info(f"Движение головы на уровне 7: {'обнаружено' if head_movement_detected else 'не обнаружено'}")
    
    if liveness_level == 8:
        head_movement_detected = detect_head_movement(prev_landmarks, face_landmarks, movement_threshold=2)  # Строгий порог движения
        logging.info(f"Движение головы на уровне 8: {'обнаружено' if head_movement_detected else 'не обнаружено'}")
    
    if liveness_level == 9:
        # Комплексная проверка с мягкими порогами
        texture_result = analyze_face_texture(img, face_location, texture_threshold=50)
        reflection_result = analyze_reflection(img, face_location, reflection_threshold=180)
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        blink_detected = detect_blink(left_eye, ear_threshold=0.3) or detect_blink(right_eye, ear_threshold=0.3)
        head_movement_detected = detect_head_movement(prev_landmarks, face_landmarks, movement_threshold=3)
    
    if liveness_level == 10:
        # Комплексная проверка с жесткими порогами
        texture_result = analyze_face_texture(img, face_location, texture_threshold=100)
        reflection_result = analyze_reflection(img, face_location, reflection_threshold=150)
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        blink_detected = detect_blink(left_eye, ear_threshold=0.2) or detect_blink(right_eye, ear_threshold=0.2)
        head_movement_detected = detect_head_movement(prev_landmarks, face_landmarks, movement_threshold=2)
    
    if not (texture_result and reflection_result and blink_detected and head_movement_detected):
        logging.info(f"Лицо на уровне {liveness_level} не является живым.")
        return False
    
    logging.info(f"Лицо распознано как живое на уровне {liveness_level}.")
    return True


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Ошибка: Камера не обнаружена.")
        return

    known_faces = load_known_face_encodings('Images')
    group_manager = GroupManager()

    prev_landmarks = None
    liveness_level = 9

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Ошибка при получении изображения с камеры.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

        if face_landmarks_list:
            for i, face_landmarks in enumerate(face_landmarks_list):
                face_location = face_locations[i]

                # Проверка на живость лица с учетом уровня
                is_live = is_live_face(rgb_frame, face_location, face_landmarks, prev_landmarks, liveness_level)
                prev_landmarks = face_landmarks

                if is_live:
                    logging.info("Лицо является живым.")
                else:
                    logging.info("Лицо не является живым.")
        
        # Отображение видеопотока
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("Video", frame)

        logging.info(f"Текущий уровень проверки живости: {liveness_level}")

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Программа завершена.")

if __name__ == "__main__":
    main()
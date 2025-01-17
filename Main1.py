import sys
import os
import cv2
import face_recognition
import numpy as np
from scipy.spatial import distance as dist
from collections import Counter
import logging
import logging.handlers
import json
import platform
import subprocess

# Custom logging handler that ensures log messages are written and flushed to file immediately
class FlushFileHandler(logging.handlers.RotatingFileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

# Sets up logging to both a rotating file handler and console with detailed format
def setup_logging(log_file='main1_log.log'):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s')
    file_handler = FlushFileHandler(log_file, maxBytes=1024*1024*5, backupCount=5)  # Rotate at 5MB, keep 5 backups
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
    logging.info("Logging system initialized with rotating file handler.")

setup_logging()

def lock1():
    subprocess.run(['./lock1'], capture_output=True, text=True)

def lock2():
    subprocess.run(['./lock2'], capture_output=True, text=True)

class GroupManager:
    """Class for managing groups of faces with enhanced logging functionality."""
    
    def __init__(self, file_path='groups.json'):
        self.file_path = file_path
        self.groups = {}
        self.load_from_file()  # Load group data on initialization

    def create_group(self, group_name):
        """Creates a new group if it doesn't exist."""
        if group_name in self.groups:
            logging.warning(f"Group '{group_name}' already exists.")
        else:
            self.groups[group_name] = []
            logging.info(f"Group '{group_name}' created.")
            self.save_to_file()  # Save changes to file

    def delete_group(self, group_name):
        """Deletes an existing group by name if found."""
        if group_name in self.groups:
            del self.groups[group_name]
            logging.info(f"Group '{group_name}' deleted.")
            self.save_to_file()
        else:
            logging.warning(f"Group '{group_name}' not found.")

    def rename_group(self, old_name, new_name):
        """Renames an existing group if found."""
        if old_name in self.groups:
            self.groups[new_name] = self.groups.pop(old_name)
            logging.info(f"Group '{old_name}' renamed to '{new_name}'.")
            self.save_to_file()
        else:
            logging.warning(f"Group '{old_name}' not found.")

    def add_face_to_group(self, face_name, group_name):
        """Adds a face to a specified group if the group exists."""
        if group_name in self.groups:
            self.groups[group_name].append(face_name)
            logging.info(f"Face '{face_name}' added to group '{group_name}'.")
            self.save_to_file()
        else:
            logging.warning(f"Group '{group_name}' not found.")
            
    def remove_face(self, face_name):
        """Removes a face from all groups."""
        found = False
        for group_name, members in self.groups.items():
            if face_name in members:
                members.remove(face_name)
                found = True
                logging.info(f"Face '{face_name}' removed from group '{group_name}'.")
        if found:
            self.save_to_file()
        else:
            logging.warning(f"Face '{face_name}' not found in any group.")

    def get_group_of_face(self, face_name):
        """Returns the group to which a specific face belongs, if any."""
        for group_name, members in self.groups.items():
            if face_name in members:
                return group_name
        return None

    def save_to_file(self):
        """Saves all groups to a JSON file."""
        try:
            with open(self.file_path, 'w') as file:
                json.dump(self.groups, file)
            logging.info("Groups successfully saved to file.")
        except IOError as e:
            logging.error(f"Error saving to file: {e}")

    def load_from_file(self):
        """Loads groups from a JSON file, if available."""
        try:
            with open(self.file_path, 'r') as file:
                self.groups = json.load(file)
            logging.info("Groups successfully loaded from file.")
        except FileNotFoundError:
            logging.warning("File not found, skipping load.")
        except json.JSONDecodeError as e:
            logging.error(f"Error reading JSON file: {e}")
        except IOError as e:
            logging.error(f"Error loading from file: {e}")

    def set_liveness_level(self, level):
        """Sets the global liveness level variable."""
        global liveness_level
        liveness_level = level
        logging.info(f"Liveness level set to {liveness_level}")


def load_known_face_encodings(folder_path):
    """
    Loads face encodings for each image in a given folder.

    Parameters:
    folder_path (str): Path to the folder containing face images.

    Returns:
    dict: A dictionary with filenames as keys and face encodings as values.
    """
    known_encodings = {}
    image_extensions = ['.jpg', '.jpeg', '.png']

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            known_image = face_recognition.load_image_file(file_path)
            encoding = face_recognition.face_encodings(known_image)
            if encoding:
                known_encodings[filename] = encoding[0]
                logging.info(f"Encoding loaded for '{filename}'")
            else:
                logging.warning(f"Failed to obtain encoding for '{filename}'")
    return known_encodings


def recognize_faces(unknown_encodings, known_encodings, group_manager):
    """
    Recognizes faces by comparing unknown face encodings to known encodings.

    Parameters:
    unknown_encodings (list): List of face encodings to recognize.
    known_encodings (dict): Dictionary of known face encodings.
    group_manager (GroupManager): Instance for managing and retrieving group data.

    Returns:
    list: List of recognized face names and groups.
    """
    recognized_faces = []
    for unknown_encoding in unknown_encodings:
        name_found = "Unknown"
        for name, known_encoding in known_encodings.items():
            compare = face_recognition.compare_faces([known_encoding], unknown_encoding)
            if compare[0]:
                group = group_manager.get_group_of_face(name)
                name_found = f"{name} (Group: {group})" if group else name
                logging.info(f"Recognized face: {name_found}")
                break
        recognized_faces.append(name_found)
    return recognized_faces


# Calculates Eye Aspect Ratio (EAR) used for detecting blinks
def eye_aspect_ratio(eye):
    """
    Calculates the Eye Aspect Ratio (EAR) for blink detection.

    Parameters:
    eye (list): List of landmark coordinates for one eye.

    Returns:
    float: EAR value for the eye.
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Analyzes face texture for liveness detection based on threshold
def analyze_face_texture(img, face_location, texture_threshold):
    """
    Analyzes face texture to check for liveness.

    Parameters:
    img (ndarray): Image containing the face.
    face_location (tuple): Coordinates of the face in the image.
    texture_threshold (int): Variance threshold for Laplacian filter.

    Returns:
    bool: True if texture indicates liveness, False otherwise.
    """
    top, right, bottom, left = face_location
    face_region = img[top:bottom, left:right]
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    return laplacian_var > texture_threshold

# Analyzes face reflections to determine natural or artificial reflection levels
def analyze_reflection(img, face_location, reflection_threshold):
    """
    Checks reflection on the face to assess liveness.

    Parameters:
    img (ndarray): Image containing the face.
    face_location (tuple): Coordinates of the face in the image.
    reflection_threshold (int): Threshold for reflection detection.

    Returns:
    bool: True if reflection level is natural, False otherwise.
    """
    top, right, bottom, left = face_location
    face_region = img[top:bottom, left:right]
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    reflection_score = cv2.mean(gray_face)[0]
    return reflection_score < reflection_threshold

# Detects blinks based on Eye Aspect Ratio (EAR) threshold
def detect_blink(eye_landmarks, ear_threshold):
    """
    Detects a blink based on the Eye Aspect Ratio (EAR).

    Parameters:
    eye_landmarks (list): Coordinates of eye landmarks.
    ear_threshold (float): EAR threshold below which a blink is detected.

    Returns:
    bool: True if a blink is detected, False otherwise.
    """
    ear = eye_aspect_ratio(eye_landmarks)
    return ear < ear_threshold

# Detects head movement by comparing nose landmark displacement
def detect_head_movement(prev_landmarks, current_landmarks, movement_threshold):
    """
    Detects head movement based on nose position shift.

    Parameters:
    prev_landmarks (dict): Previous landmarks including nose bridge.
    current_landmarks (dict): Current landmarks including nose bridge.
    movement_threshold (float): Displacement threshold for movement.

    Returns:
    bool: True if head movement is detected, False otherwise.
    """
    if prev_landmarks is None:
        return False
    nose_prev = np.array(prev_landmarks['nose_bridge'][0])
    nose_curr = np.array(current_landmarks['nose_bridge'][0])
    distance_moved = np.linalg.norm(nose_curr - nose_prev)
    return distance_moved > movement_threshold


# Функция для определения "живого" лица с уникальными уровнями проверки живости
def is_live_face(img, face_location, face_landmarks, prev_landmarks=None, liveness_level=1):
    """
    Determines whether the given face is live based on various liveness tests, depending on the specified level.

    Parameters:
    img (ndarray): The image containing the face.
    face_location (tuple): Coordinates of the face in the image.
    face_landmarks (dict): Landmarks for the detected face (e.g., eyes, nose, etc.).
    prev_landmarks (dict, optional): Previous frame's landmarks for detecting head movement. Default is None.
    liveness_level (int): Level of liveness testing. Higher levels involve more complex checks.

    Returns:
    bool: True if the face is recognized as live, False otherwise.
    """
    # Initialize results for all liveness checks as True
    texture_result = reflection_result = blink_detected = head_movement_detected = True
    
    # Level 0: No checks, just assume the face is live
    if liveness_level == 0:
        logging.info("Level 0: No liveness check performed.")
        return True  # Return True as no checks are being done
    
    # Level 1: Soft threshold for texture analysis
    if liveness_level == 1:
        texture_result = analyze_face_texture(img, face_location, texture_threshold=50)  # Soft threshold for texture
        logging.info(f"Texture analysis result at level 1: {'live' if texture_result else 'fake'}")
        
    # Level 2: Strict threshold for texture analysis
    if liveness_level == 2:
        texture_result = analyze_face_texture(img, face_location, texture_threshold=100)  # Strict threshold for texture
        logging.info(f"Texture analysis result at level 2: {'live' if texture_result else 'fake'}")
    
    # Level 3: Soft threshold for reflection analysis
    if liveness_level == 3:
        reflection_result = analyze_reflection(img, face_location, reflection_threshold=180)  # Soft threshold for reflections
        logging.info(f"Reflection analysis result at level 3: {'natural' if reflection_result else 'fake'}")
        
    # Level 4: Strict threshold for reflection analysis
    if liveness_level == 4:
        reflection_result = analyze_reflection(img, face_location, reflection_threshold=150)  # Strict threshold for reflections
        logging.info(f"Reflection analysis result at level 4: {'natural' if reflection_result else 'fake'}")
    
    # Level 5: Blink detection with a soft EAR (Eye Aspect Ratio) threshold
    if liveness_level == 5:
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        blink_detected = detect_blink(left_eye, ear_threshold=0.3) or detect_blink(right_eye, ear_threshold=0.3)  # Soft EAR threshold
        logging.info(f"Blink detection result at level 5: {'detected' if blink_detected else 'not detected'}")
    
    # Level 6: Blink detection with a strict EAR threshold
    if liveness_level == 6:
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        blink_detected = detect_blink(left_eye, ear_threshold=0.2) or detect_blink(right_eye, ear_threshold=0.2)  # Strict EAR threshold
        logging.info(f"Blink detection result at level 6: {'detected' if blink_detected else 'not detected'}")
    
    # Level 7: Head movement detection with a soft movement threshold
    if liveness_level == 7:
        head_movement_detected = detect_head_movement(prev_landmarks, face_landmarks, movement_threshold=3)  # Soft movement threshold
        logging.info(f"Head movement detection result at level 7: {'detected' if head_movement_detected else 'not detected'}")
    
    # Level 8: Head movement detection with a strict movement threshold
    if liveness_level == 8:
        head_movement_detected = detect_head_movement(prev_landmarks, face_landmarks, movement_threshold=2)  # Strict movement threshold
        logging.info(f"Head movement detection result at level 8: {'detected' if head_movement_detected else 'not detected'}")
    
    # Level 9: Comprehensive check with soft thresholds for texture, reflection, blink, and head movement
    if liveness_level == 9:
        texture_result = analyze_face_texture(img, face_location, texture_threshold=50)  # Soft threshold for texture
        reflection_result = analyze_reflection(img, face_location, reflection_threshold=180)  # Soft threshold for reflections
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        blink_detected = detect_blink(left_eye, ear_threshold=0.3) or detect_blink(right_eye, ear_threshold=0.3)  # Soft EAR threshold
        head_movement_detected = detect_head_movement(prev_landmarks, face_landmarks, movement_threshold=3)  # Soft movement threshold
        
        # Logging for level 9
        logging.info("Level 9: Comprehensive check with soft thresholds")
        logging.info(f"Texture analysis result at level 9: {'live' if texture_result else 'fake'}")
        logging.info(f"Reflection analysis result at level 9: {'natural' if reflection_result else 'fake'}")
        logging.info(f"Blink detection result at level 9: {'detected' if blink_detected else 'not detected'}")
        logging.info(f"Head movement detection result at level 9: {'detected' if head_movement_detected else 'not detected'}")
    
    # Level 10: Comprehensive check with strict thresholds for texture, reflection, blink, and head movement
    if liveness_level == 10:
        texture_result = analyze_face_texture(img, face_location, texture_threshold=100)  # Strict threshold for texture
        reflection_result = analyze_reflection(img, face_location, reflection_threshold=150)  # Strict threshold for reflections
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        blink_detected = detect_blink(left_eye, ear_threshold=0.2) or detect_blink(right_eye, ear_threshold=0.2)  # Strict EAR threshold
        head_movement_detected = detect_head_movement(prev_landmarks, face_landmarks, movement_threshold=2)  # Strict movement threshold
        
        # Logging for level 10
        logging.info("Level 10: Comprehensive check with strict thresholds")
        logging.info(f"Texture analysis result at level 10: {'live' if texture_result else 'fake'}")
        logging.info(f"Reflection analysis result at level 10: {'natural' if reflection_result else 'fake'}")
        logging.info(f"Blink detection result at level 10: {'detected' if blink_detected else 'not detected'}")
        logging.info(f"Head movement detection result at level 10: {'detected' if head_movement_detected else 'not detected'}")
    
    # If any of the liveness checks fail, the face is considered not live
    if not (texture_result and reflection_result and blink_detected and head_movement_detected):
        logging.info(f"Face at level {liveness_level} is not live.")
        return False
    
    logging.info(f"Face recognized as live at level {liveness_level}.")
    return True


def main():
    """
    Main function to run live face recognition and liveness detection.
    Captures video from the webcam, processes frames, and performs recognition and liveness checks.
    Handles headless mode (no GUI) for Ubuntu server environments.
    """
    
    global liveness_level
    
    # Check the operating system
    current_platform = platform.system()

    # Try opening the camera with id 0 first, if it fails, try other ids
    cap = None
    for camera_id in range(0, 5):  # Try camera ids from 0 to 4
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            logging.info(f"Camera with ID {camera_id} successfully opened.")  # Log which camera ID worked
            break
        else:
            logging.warning(f"Camera with ID {camera_id} could not be opened.")  # Log failure to open specific camera

    if not cap or not cap.isOpened():
        logging.error("Unable to access any camera. Please check your connection and try again.")  # If no camera is found
        return

    # Load known faces and their encodings from the 'Images' directory
    known_faces = load_known_face_encodings('Images')
    group_manager = GroupManager()

    prev_landmarks = None  # Variable to store the previous face landmarks for movement analysis
    liveness_level = 0  # Set a threshold for liveness detection (higher means stricter criteria)

    # Check if the environment is headless (no display available)
    is_headless = current_platform == 'Linux' and not os.environ.get('DISPLAY')
    
    frame_count = 0

    while True:
        ret, frame = cap.read()  # Capture each frame from the video stream
        if not ret:
            logging.error("Failed to capture an image from the camera. Please check the camera connection.")  # Clear error message for frame capture failure
            break
        
        if frame_count % 10 != 0: # Skipping every second frame
            frame_count += 1
            continue
        
        frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the captured frame from BGR to RGB color space
        face_locations = face_recognition.face_locations(rgb_frame)  # Detect faces in the frame
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)  # Detect landmarks for each face

        if face_landmarks_list:
            for i, face_landmarks in enumerate(face_landmarks_list):
                face_location = face_locations[i]

                # Perform liveness detection
                is_live = is_live_face(rgb_frame, face_location, face_landmarks, prev_landmarks, liveness_level)
                prev_landmarks = face_landmarks  # Update previous landmarks for the next frame

                if is_live:
                    unknown_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]
                    recognized_faces = recognize_faces([unknown_encoding], known_faces, group_manager)

                    for recognized_face in recognized_faces:
                        logging.info(f"Live face detected at position {face_location}. Recognized face: {recognized_face}.")
                        print(recognized_face)
                        if recognized_face == 'person1.jpg':
                            lock1()
                            #pass
                        else:
                            lock2()
                            #pass
                else:
                    logging.info(f"Non-live face detected at position {face_location}.")
        
        # Draw rectangles around detected faces on the frame
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Draw green rectangles

        # If not in headless mode, display the frame
        if not is_headless:
            cv2.imshow("Live Face Recognition", frame)

        # Log the frame (or save to a file) if in headless mode for verification purposes
        if is_headless:
            logging.info("Frame captured, but not displayed due to headless mode.")

        # Exit on pressing 'q' or if in headless mode (simulate exit if needed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting the application as 'q' was pressed.")  # Inform the user when they exit the program
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close OpenCV windows if any
    logging.info("Program execution finished successfully.")  # Log the program's completion

if __name__ == "__main__":
    # Set up logging configuration for better clarity and cross-platform compatibility
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

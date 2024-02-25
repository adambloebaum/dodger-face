import cv2
import dlib
import numpy as np
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_initialize import Base, Face

# Database connection
engine = create_engine('sqlite:///face_encodings.db')
Session = sessionmaker(bind=engine)
session = Session()

face_rec_model = dlib.face_recognition_model_v1('facetracking/dlib_face_recognition_resnet_model_v1.dat')
cnn_face_detector = dlib.cnn_face_detection_model_v1('facetracking/mmod_human_face_detector.dat')
shape_predictor = dlib.shape_predictor('facetracking/shape_predictor_68_face_landmarks.dat')

def load_known_encodings(session):
    known_encodings = []
    known_ids = []
    known_names = []
    faces = session.query(Face).all()

    for face in faces:
        # Load and deserialize the encoding
        known_encodings.append(json.loads(face.encoding))
        known_ids.append(face.id)
        known_names.append(face.name)

    return known_encodings, known_names

# Extract every Nth frame from the video
def extract_frames(video_path, frame_skip=50):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

# Detect faces
def detect_faces(frames, cnn_face_detector):
    detected_faces = []

    for frame in frames:
        # convert the OpenCV BGR image to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = cnn_face_detector(rgb_image, 1)

        for face in faces:
            x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
            detected_faces.append(rgb_image[y:y+h, x:x+w])

    return detected_faces

# Extract face encodings
def extract_face_encodings(face):
    # Convert the OpenCV BGR image to RGB
    rgb_image = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Initialize the Dlib face detector and shape predictor
    face_detector = dlib.get_frontal_face_detector()

    # Detect faces
    detected_faces = face_detector(rgb_image, 1)
    if len(detected_faces) == 0:
        # No faces found in the image, return None or an appropriate value
        return None

    # Get the landmarks/parts for the face
    shape = shape_predictor(rgb_image, detected_faces[0])

    # Align and crop the face using the landmarks
    face_chip = dlib.get_face_chip(rgb_image, shape)

    # Compute the face descriptor
    face_descriptor = face_rec_model.compute_face_descriptor(face_chip)

    # Convert the descriptor to a numpy array, then to a list, and serialize to JSON
    encoding = np.array(face_descriptor).tolist()
    return json.dumps(encoding)

# Identify face in video
def identify_faces_in_video(video_path, known_encodings, known_names, known_ids):
    frames = extract_frames(video_path)
    for frame in frames:
        detected_faces = detect_faces([frame], cnn_face_detector)
        for face in detected_faces:
            face_encoding_json = extract_face_encodings(face)
            if face_encoding_json is not None:
                encoding = json.loads(face_encoding_json)

                # Calculate distances to known encodings
                distances = np.linalg.norm([encoding] - known_encodings, axis=1)
                best_match_index = np.argmin(distances)

                if distances[best_match_index] < 0.6:
                    name = known_names[best_match_index]
                    id = known_ids[best_match_index]
                    print(f"Identified: {name} - {id}")
                else:
                    print("Unknown face detected")
            else:
                print("No face found in this frame.")


known_encodings, known_names, known_ids = load_known_encodings(session)

identify_faces_in_video(".mp4", known_encodings, known_names, known_ids)

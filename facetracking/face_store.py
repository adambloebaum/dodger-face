import cv2
import numpy as np
import os
import glob
import dlib
import json
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from db_initialize import Base, Face

# Database connection
engine = create_engine('sqlite:///face_encodings_db.db')
Session = sessionmaker(bind=engine)
session = Session()

# Load in models and predictor
cnn_face_detector = dlib.cnn_face_detection_model_v1(r'mmod_human_face_detector.dat')
face_rec_model = dlib.face_recognition_model_v1(r'dlib_face_recognition_resnet_model_v1.dat')
shape_predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')

# Assumed filename format of LastName_FirstName_IDnum
video_directory = r"face_scans"

def parse_filename(filename):
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    parts = name.split('_')
    if len(parts) < 3:
        return None, None
    last_name, first_name, id = parts[0], parts[1], parts[2]
    full_name = f"{first_name} {last_name}"
    return full_name, id

# Extract every Nth frame from the video
def extract_frames(video_path, frame_skip=20):
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
        # Convert the OpenCV BGR image to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = cnn_face_detector(rgb_image, 1)

        for face in faces:
            x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
            detected_faces.append(rgb_image[y:y+h, x:x+w])

    return detected_faces

# Apply distortions to faces for robustness
def distort_faces(faces, rotate=True, add_noise=True, flip=True):
    distorted_faces = []

    for face in faces:
        if rotate:
            # Rotate the face by a random angle between -30 and 30 degrees
            (h, w) = face.shape[:2]
            center = (w // 2, h // 2)
            angle = np.random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            face = cv2.warpAffine(face, M, (w, h))

        if add_noise:
            # Add random noise to the face
            noise = np.random.normal(0, 15, face.shape).astype(np.uint8)
            face = cv2.add(face, noise)

        if flip:
            # Horizontally flip the face
            face = cv2.flip(face, 1)

        distorted_faces.append(face)

    return distorted_faces

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

def store_face_encoding(session, name, id, encoding):
    new_face = Face(name=name, id=id, encoding=encoding)
    session.add(new_face)
    session.commit()

# Process each video
def process_video(video_path):
    name, id = parse_filename(video_path)
    if name is None or id is None:
        print(f"Invalid filename format for {video_path}. Skipping.")
        return

    print(f"Beginning video processing for {name} - {id}")
    
    print("Starting frame extraction...")
    frames = extract_frames(video_path)
    print(len(frames), "frames extracted")
    
    print("Starting face detection...")
    faces = detect_faces(frames, cnn_face_detector)
    print(len(faces), "faces detected")

    print("Starting face augmentation...")
    rotated_faces = distort_faces(faces, rotate=True, add_noise=False, flip=False)
    noisy_faces = distort_faces(faces, rotate=False, add_noise=True, flip=False)
    flipped_faces = distort_faces(faces, rotate=False, add_noise=False, flip=True)
    augmented_faces = rotated_faces + noisy_faces + flipped_faces
    combined_faces = faces + augmented_faces
    print(len(augmented_faces), "face generated")
    
    print("Starting encoding extraction...")
    processed_faces = 0
    for face in combined_faces:
        encoding = extract_face_encodings(face)
        if encoding is not None:
            store_face_encoding(session, name, id, encoding)
            processed_faces += 1
    
    print(f"Successfully processed {processed_faces} face encodings to the database for {name} - {id}.")

# Loop through and process each video in the directory
for video_file in glob.glob(os.path.join(video_directory, '*.mov')):
    process_video(video_file)

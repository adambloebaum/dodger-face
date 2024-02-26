import os
import cv2
import json
import dlib
import numpy as np
import pandas as pd
from initialize_database import Base, Face, Encoding

def parse_filename(filename):
    """
    Extracts the full name from the given filename.

    The filename is expected to have at least two parts separated by an underscore.
    For example: 'John_Doe.mp4'

    Parameters:
    filename (str): The name of the file.

    Returns:
    str: The full name extracted from the file, or None if not enough parts are present.
    """
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    parts = name.split('_')
    if len(parts) < 2:
        return "No Name"
    first_name, last_name = parts[0].capitalize(), parts[1].capitalize()
    full_name = f"{first_name} {last_name}"
    return full_name

def get_or_create_face(session, name):
    face = session.query(Face).filter_by(name=name).first()
    if face is None:
        face = Face(name=name)
        session.add(face)
        session.commit()
    return face.id

# set really high currently without more computing power
def extract_frames(video_path, frame_skip=3000):
    """
    Extracts frames from a video, skipping a specified number of frames.

    Parameters:
    video_path (str): Path to the video file.
    frame_skip (int): Number of frames to skip between each extracted frame.

    Returns:
    list: A list of extracted frames.
    """
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

def detect_faces(frames, cnn_face_detector):
    """
    Detects faces in the given frames.

    Parameters:
    frames (list): List of frames from a video.
    cnn_face_detector: Dlib's CNN face detection model.

    Returns:
    list: A list of cropped images of detected faces.
    """
    detected_faces = []

    for frame in frames:
        # Convert the OpenCV BGR image to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = cnn_face_detector(rgb_image, 1)

        for face in faces:
            print('Face detected!')
            x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
            detected_faces.append(rgb_image[y:y+h, x:x+w])

    return detected_faces

def distort_faces(faces, rotate=True, add_noise=True, flip=True):
    """
    Applies distortions to a list of face images for data augmentation.

    Parameters:
    faces (list): List of face images.
    rotate (bool): Whether to apply random rotation.
    add_noise (bool): Whether to add random noise.
    flip (bool): Whether to horizontally flip the image.

    Returns:
    list: A list of distorted face images.
    """
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

def extract_face_encodings(face, shape_predictor, face_rec_model):
    """
    Extracts face encodings from a single face image.

    Parameters:
    face: The face image.
    shape_predictor: Dlib's shape predictor for face landmark detection.
    face_rec_model: Dlib's face recognition model.

    Returns:
    str: A JSON string of the face encoding, or None if no face is detected.
    """
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

def store_face_encoding(session, face_id, encoding):
    """
    Stores a face encoding in the database.

    Parameters:
    session: The SQLAlchemy session for database interaction.
    face_id (int): The ID of the face associated with the encoding.
    encoding (str): The face encoding in JSON format.
    """
    new_encoding = Encoding(face_id=face_id, encoding=encoding)
    session.add(new_encoding)
    session.commit()

def intake_video(video_path, cnn_face_detector, shape_predictor, face_rec_model, session):
    """
    Processes a video file, detects faces, and stores their encodings in the database.

    Parameters:
    video_path (str): Path to the video file.
    cnn_face_detector: Dlib's CNN face detection model.
    shape_predictor: Dlib's shape predictor for face landmarks.
    face_rec_model: Dlib's face recognition model.
    session: The SQLAlchemy session for database interaction.
    """
    name = parse_filename(video_path)
    if name is None:
        print(f"Invalid filename format for {video_path}. Skipping.")
        return

    face_id = get_or_create_face(session, name)
    print(f"Beginning video processing for {name}...")
    
    frames = extract_frames(video_path)
    print(len(frames), "frames extracted")
    
    faces = detect_faces(frames, cnn_face_detector)
    print(len(faces), "faces detected")

    rotated_faces = distort_faces(faces, rotate=True, add_noise=False, flip=False)
    noisy_faces = distort_faces(faces, rotate=False, add_noise=True, flip=False)
    flipped_faces = distort_faces(faces, rotate=False, add_noise=False, flip=True)
    augmented_faces = rotated_faces + noisy_faces + flipped_faces
    combined_faces = faces + augmented_faces
    print(len(augmented_faces), "faces generated")
    
    processed_faces = 0
    for face in combined_faces:
        encoding = extract_face_encodings(face, shape_predictor, face_rec_model)
        if encoding is not None:
            store_face_encoding(session, face_id, encoding)
            processed_faces += 1
    
    print(f"Successfully uploaded {processed_faces} face encodings to the database for {name}")

def load_known_encodings(session):
    """
    Loads known face encodings and names from the database.

    Parameters:
    session: The SQLAlchemy session for database interaction.

    Returns:
    tuple: Two lists, one of known face encodings and another of corresponding names.
    """
    known_encodings = []
    known_names = []

    try:
        # Perform a join query to get encodings with corresponding names
        results = session.query(Face.name, Encoding.encoding).join(Encoding, Face.id == Encoding.face_id).all()

        for name, encoding in results:
            # Deserialize the encoding from JSON format
            encoding = json.loads(encoding)

            # Append the encoding and corresponding name to the lists
            known_encodings.append(encoding)
            known_names.append(name)
    except Exception as e:
        print(f"Error loading known encodings from database: {e}")

    return known_encodings, known_names

# frame skip currently set higher to 300 for test videos
def identify_faces_in_video(video_path, known_encodings, known_names, cnn_face_detector, shape_predictor, face_rec_model, session, frame_skip=300):
    """
    Identifies faces in a video file using known face encodings.

    This function processes a given video, detects faces frame by frame (with skipping),
    and identifies these faces by comparing them with a set of known encodings. It keeps track
    of each identified face, counting appearances and calculating average confidence.

    Parameters:
    video_path (str): Path to the video file to be processed.
    known_encodings (list): A list of known face encodings for comparison.
    known_names (list): A list of names corresponding to the known encodings.
    cnn_face_detector: Dlib's CNN face detection model.
    shape_predictor: Dlib's shape predictor for face landmark detection.
    face_rec_model: Dlib's face recognition model.
    session: SQLAlchemy database session for any needed database interactions.
    frame_skip (int): Number of frames to skip between each processed frame. Default is 300.

    Returns:
    dict: A dictionary summarizing the appearance count and average confidence for each identified face.
    
    The function first opens the video file and iterates through it, skipping a set number of frames
    as specified. For each processed frame, it detects faces and then extracts their encodings.
    These encodings are compared with a list of known encodings to find the best match. If a match
    is found with a confidence higher than a threshold (0.6 in this case), the function records the
    appearance of the face. Finally, it provides a summary of all faces identified in the video along
    with their appearance counts and average confidence levels.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    progress_interval = total_frames * 0.10  # 10% of total frames

    face_appearances = {}  # Dictionary to hold face appearance count and total confidence
    known_encodings_np = np.array(known_encodings)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            # Process frame
            detected_faces = detect_faces([frame], cnn_face_detector)
            print(f"Frame {frame_count}: Detected {len(detected_faces)} faces")  # Debugging line

            for face in detected_faces:
                encoding_json = extract_face_encodings(face, shape_predictor, face_rec_model)
                if encoding_json is not None:
                    encoding = np.array(json.loads(encoding_json))
                    distances = np.linalg.norm(encoding - known_encodings_np, axis=1)

                    # Debugging: Print the best match and its distance
                    best_match_index = np.argmin(distances)
                    confidence = 1 - distances[best_match_index]
                    print(f" - Best match: {known_names[best_match_index]} with confidence {confidence:.2f}")

                    if confidence > 0.6:
                        name = known_names[best_match_index]
                        if name not in face_appearances:
                            face_appearances[name] = {'count': 0, 'total_confidence': 0}
                        face_appearances[name]['count'] += 1
                        face_appearances[name]['total_confidence'] += confidence
                else:
                    print(" - Face encoding failed")  # Debugging line

        frame_count += 1
        if frame_count % progress_interval < 1:
            print(f"Processing progress: {int(frame_count / total_frames * 100)}%")

    cap.release()

    # Summarize results
    summary = {}
    for name, data in face_appearances.items():
        appearance_time = data['count'] / total_frames * 100
        avg_confidence = data['total_confidence'] / data['count']
        summary[name] = {'appearance_time_percent': appearance_time, 'average_confidence': avg_confidence}

    print("Summary of video analysis:")
    for name, info in summary.items():
        print(f"{name} appeared for {info['appearance_time_percent']:.2f}% of the video with an average confidence of {info['average_confidence']:.2f}")

    return summary

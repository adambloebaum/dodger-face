from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.exc import SQLAlchemyError
import logging
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

Base = declarative_base()

class Face(Base):
    __tablename__ = 'faces'
    id = Column(Integer, primary_key=True)  # Unique ID for each person
    name = Column(String, unique=True)  # Name of the person
    encodings = relationship("Encoding", back_populates="face")  # Relationship to encodings

class Encoding(Base):
    __tablename__ = 'encodings'
    id = Column(Integer, primary_key=True)  # Unique ID for each encoding
    face_id = Column(Integer, ForeignKey('faces.id'))  # Link to the face
    encoding = Column(Text)  # The encoding data
    face = relationship("Face", back_populates="encodings")  # Relationship to face

def initialize_database():
    try:
        # Determine the directory of the current script
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # Create database engine
        engine = create_engine(f'sqlite:///{dir_path}/face_encodings_db.db')
        Base.metadata.create_all(engine)
        logging.info("Database initialized successfully.")
        return engine
    except SQLAlchemyError as e:
        logging.error("Error occurred during engine creation or table initialization: %s", e)
        raise

# Function to load and update the configuration
def load_and_update_config(config_path):
    # Read the JSON configuration file
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Update the connection string
    db_filename = 'face_encodings_db.db'
    db_path = os.path.join(script_dir, db_filename)
    config['database']['connection_string'] = f"sqlite:///{db_path}"

    # Update the dat strings
    face_rec_model = 'dat/dlib_face_recognition_resnet_model_v1.dat'
    cnn_face_detector = 'dat/mmod_human_face_detector.dat'
    shape_predictor = 'dat/shape_predictor_68_face_landmarks.dat'
    face_rec_model_path = os.path.join(script_dir, face_rec_model)
    cnn_face_detector_path = os.path.join(script_dir, cnn_face_detector)
    shape_predictor_path = os.path.join(script_dir, shape_predictor)
    config['models']['face_recognition_model'] = face_rec_model_path
    config['models']['cnn_face_detector'] = cnn_face_detector_path
    config['models']['shape_predictor'] = shape_predictor_path

    # Write the updated configuration back to the JSON file
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

    return config

if __name__ == "__main__":
    # Determine the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Construct the path to config.json
    config_file_path = os.path.join(dir_path, 'config.json')
    # Load and update configuration from the JSON file
    config = load_and_update_config(config_file_path)

    print(config['database']['connection_string'])

    # Initialize the database when this script is run
    initialize_database()

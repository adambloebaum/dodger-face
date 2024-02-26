import os
import glob
import dlib
import json
import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, exc
from functions import intake_video, load_config

def main():
    logging.basicConfig(level=logging.INFO)

    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    config = load_config(config_path)

    try:
        # Database connection
        engine = create_engine(config['database']['connection_string'])
        Session = sessionmaker(bind=engine)
        session = Session()

        # Load in models and predictor
        face_rec_model = dlib.face_recognition_model_v1(config['models']['face_recognition_model'])
        cnn_face_detector = dlib.cnn_face_detection_model_v1(config['models']['cnn_face_detector'])
        shape_predictor = dlib.shape_predictor(config['models']['shape_predictor'])

        script_dir = os.path.dirname(os.path.abspath(__file__))
        video_dir = os.path.join(script_dir, 'trimmed_videos')

        # Iterate through each video file and process it
        for video_file in glob.glob(os.path.join(video_dir, '*.mp4')):
            intake_video(video_file, cnn_face_detector, shape_predictor, face_rec_model, session)

    except exc.SQLAlchemyError as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        session.close()
        logging.info("Database session closed.")

if __name__ == "__main__":
    main()

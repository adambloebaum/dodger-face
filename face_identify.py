import json
import os
import glob
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import dlib
from joblib import load
from functions import load_known_encodings, identify_faces_in_video, load_config

def main():
    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.json')
    config = load_config(config_path)

    # Setup database
    engine = create_engine(config['database']['connection_string'])
    Session = sessionmaker(bind=engine)
    session = Session()

    # Load in models and predictor
    face_rec_model = dlib.face_recognition_model_v1(config['models']['face_recognition_model'])
    cnn_face_detector = dlib.cnn_face_detection_model_v1(config['models']['cnn_face_detector'])
    shape_predictor = dlib.shape_predictor(config['models']['shape_predictor'])
    pca_model = load('pca_model.joblib')
    nn_model = load('nn_model.joblib')

    # Load known encodings and names from the database
    known_encodings, known_names = load_known_encodings(session)

    # Construct the path to the 'test_videos' folder in one step
    video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_videos')

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the directory where the untrained summaries will be saved
    summary_dir = os.path.join(script_dir, 'summaries')
    # Create the summary directory if it doesn't exist
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    # Iterate through each MP4 file in the 'test_videos' folder
    for video_file in glob.glob(os.path.join(video_dir, '*.mp4')):
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        print(f"Processing video: {base_name}")
        summary = identify_faces_in_video(video_file, known_encodings, known_names, cnn_face_detector, shape_predictor, face_rec_model, pca_model, nn_model, session)
        print(f"Summary for {base_name}:")
        for name, info in summary.items():
            print(f"  {name}: appeared for {info['appearance_time_percent']}% of the video with an average confidence of {info['average_confidence']}")

        json_filename = os.path.join(summary_dir, f'{base_name}_summary.json')
        with open(json_filename, 'w') as json_file:
            json.dump(summary, json_file, indent=4)

if __name__ == "__main__":
    main()

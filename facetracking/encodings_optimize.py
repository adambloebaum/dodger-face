from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from initialize_database import Base, Face, Encoding
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import json
import os
from functions import load_known_encodings

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    config = load_config(config_path)

    # Database connection
    engine = create_engine(config['database']['connection_string'])
    Session = sessionmaker(bind=engine)
    session = Session()

    # Load known encodings
    known_encodings, known_names = load_known_encodings(session)
    known_encodings = np.array(known_encodings)
    print("Encodings loaded successfully")

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=0.95)  # Tune this as needed
    encodings_pca = pca.fit_transform(known_encodings)
    print("PCA completed")

    # Train Nearest Neighbors model
    nn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn_model.fit(encodings_pca)
    print("NN completed")

    # Save models
    joblib.dump(pca, 'facetracking/pca_model.joblib')
    joblib.dump(nn_model, 'facetracking/nn_model.joblib')

    print("PCA and Nearest Neighbors models saved successfully.")

if __name__ == "__main__":
    main()

# dodgerface
Facial recognition system for the LA Dodgers star players

## Introduction
Dodgerface is a facial recognition system designed to identify LA Dodgers stars from interviews and press conferences. Utilizing advanced computer vision and machine learning techniques, it processes video clips, stores facial encodings in a database, and accurately identifies individuals in new video clips. Stars Clayton Kershaw, Shohei Ohtani, Walker Buehler, Mookie Betts, and Freedie Freeman are used in this project.

## Installation
To set up, follow these steps:
1. Install required Python packages.
2. Set up the SQLite database and local filepaths using `initialize_database.py`.

## Usage
1. **Database Setup**: Run `initialize_database.py` to set up the SQLite database and configuration filepaths
2. **Video Trimming**: Use `video_trim.py` to trim raw videos and store them in `trimmed_videos`. *Raw videos have been omitted from the public repository due to filesize*.
3. **Database Upload**: Process trimmed videos with `face_store.py` to upload face encoding data to the database.
4. **Face Identification**: Run `face_identify.py` on videos in `test_videos` to identify faces. Summaries are saved in `summaries` in JSON format.

## Features
- Video trimming and standardization.
- Facial encoding storage and retrieval from a database.
- Face recognition in videos with statistical summaries.

## Dependencies
- Python 3.x
- OpenCV
- Dlib
- SQLAlchemy
- MoviePy
- Pandas
- Numpy

## Database Schema

The project utilizes a relational database with a schema designed to effectively manage and retrieve facial data. The database schema is comprised of two main tables: `faces` and `encodings`. Here's a detailed overview:

### `faces` Table
- **`id`**: An integer that serves as the primary key. It represents a unique identifier for each individual person.
- **`name`**: A string field that stores the name of the person. This is also unique for each entry.

### `encodings` Table
- **`id`**: An integer that serves as the primary key, representing a unique identifier for each facial encoding.
- **`face_id`**: An integer foreign key that links to the `id` in the `faces` table. It signifies which person a particular encoding belongs to.
- **`encoding`**: A text field that contains the facial feature encoding data.

### Relationships
- **Face to Encoding**: This is a one-to-many relationship where each face can have multiple encodings. This relationship is represented in the `Face` class with the `encodings` attribute, which refers to the `Encoding` class. The `back_populates="face"` attribute in the `Encoding` class ensures a bidirectional relationship.
- **Encoding to Face**: This represents a many-to-one relationship. Each encoding is linked to a single face. This relationship is defined in the `Encoding` class with the `face` attribute, referring to the `Face` class. The `back_populates="encodings"` attribute in the `Face` class maintains the connection from the other side.

## Identification Process
1. **Loading Known Encodings**: The system loads known face encodings and corresponding names from the database. This is done using a join query to associate each encoding with the correct name, deserializing the encoding from JSON format.

2. **Video Processing**: For each video file, the system captures frames, skipping a specified number of frames to optimize processing.

3. **Face Detection and Encoding**: Faces are detected in each frame using Dlib's CNN face detection model. These faces are then encoded using Dlib's face recognition model.

4. **Identification and Matching**: The encoded faces are transformed using a PCA model to reduce dimensionality. The Nearest Neighbor model is then used to find the best match from the known encodings. A threshold is set to determine whether a face is known or unknown.

5. **Tracking Appearances**: The system tracks each appearance of a known individual, counting the occurrences and calculating the confidence score based on the distance metric from the Nearest Neighbor model.

## Summarization output
The summarization process involves the following steps:

1. **Appearance Time Calculation**: For each identified individual, the system calculates the percentage of time they appear in the video based on their appearance count and total frames with faces.

2. **Average Confidence**: The average confidence score for each individual is calculated. This score is derived from the confidence scores obtained during the identification process.

3. **Summary Generation**: The system generates a summary that includes the appearance time percentage and average confidence for each identified individual. This summary is provided in a dictionary format, which can then be serialized into JSON for reporting or further analysis.

## Configuration
Configuration settings are located in `config.json`. This includes database connection and paths to facial recognition models.

## Documentation
Detailed documentation of functions is available in `functions.py`, including database operations, video processing, and face detection.

## Examples
Example JSON summaries from facial recognition can be found in the `summaries` directory.

## Video Links

### Raw videos trimmed and processed

Clayton Kershaw: https://www.youtube.com/watch?v=2kN57k_SveU&ab_channel=DodgersHighlights

Walker Beuhler: https://www.youtube.com/watch?v=IWoNVyGEz6Q&ab_channel=DodgerBlue

Mookie Betts: https://www.youtube.com/watch?v=7hDOAUQH2GA&ab_channel=TheDodgersBleedLosPodcast

Freddie Freeman: https://www.youtube.com/watch?v=ajqDo3zuJIo&ab_channel=BallySportsSouth

Shohei Ohtani: https://www.youtube.com/watch?v=9qUCo-K2OhQ&ab_channel=GQSports

### Test videos summarized

Clayton Kershaw & Walker Buehler & reporter: https://www.youtube.com/watch?v=BiBvapeX7Zo&ab_channel=LosAngelesTimes

Shohei Ohtani & Harold Reynolds: https://www.youtube.com/watch?v=2zPRReEUnUg&t=24s&ab_channel=MLB

Freddie Freeman & Mookie Betts & reporter: https://www.youtube.com/watch?v=VzYpn7BV2ys&ab_channel=LosAngelesTimes

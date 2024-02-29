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
- **Encoding to Face**: This represents a many-to-one relationship. Each encoding is linked to a single face. This relationship is defined in the `Encoding` class with the `face` attribute, referring to the `Face` class. The `back_populates="encodings"` attribute in the `Face` class maintains the connection from the other side【51†source】.

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

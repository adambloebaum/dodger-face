# putervis

computer vision

## facetracking

a facial recognition pipeline for storing facial encodings to a database and identifying the face in a video. 3 minute clips from interviews/press conferences by LA dodgers stars Clayton Kershaw, Walker Beuhler, Mookie Betts, Freddie Freeman, and Shohei Ohtani are used to extract face encodings to the database, and similar videos are used to test the effectiveness of the facial recognition system

### database schema

#### `faces`
- `id` (integer): primary key, unique id for each person.
- `name` (string): name of the person, unique.

#### `encodings`
- `id` (integer): primary key, unique id for each encoding.
- `face_id` (integer, foreignkey): foreign key linking to `id` in the `faces` table.
- `encoding` (text): facial feature encoding data.

### relationships

- **Face to Encoding**: one-to-many relationship. each face can have multiple encodings. this relationship is represented in the `Face` class with the `encodings` attribute, which refers to the `Encoding` class. the `back_populates="face"` attribute in the `Encoding` class ensures bidirectional relationship.
- **Encoding to Face**: many-to-one relationship. each encoding is linked to a single face. this relationship is represented in the `Encoding` class with the `face` attribute, which refers to the `Face` class. the `back_populates="encodings"` attribute in the `Face` class maintains the connection from the other side

### optimization

principle component analysis (pca) is configured to maintain 95% of the data variance to enhance computational efficiency without significantly compromising the accuracy of face recognition. a nearest neighbor model is trained on the pca-transformed encodings for rapidly identifying the most similar facial encodings from the database, using the `ball_tree` algorithm. both models are persisted by using joblib for saving and quick loading

### face encoding upload


### video identification




## video links

# raw videos trimmed and processed to database

clayton kershaw: https://www.youtube.com/watch?v=2kN57k_SveU&ab_channel=DodgersHighlights
walker beuhler: https://www.youtube.com/watch?v=IWoNVyGEz6Q&ab_channel=DodgerBlue
mookie betts: https://www.youtube.com/watch?v=7hDOAUQH2GA&ab_channel=TheDodgersBleedLosPodcast
freddie freeman: https://www.youtube.com/watch?v=ajqDo3zuJIo&ab_channel=BallySportsSouth
shohei ohtani: https://www.youtube.com/watch?v=9qUCo-K2OhQ&ab_channel=GQSports

# short videos identified and summarized

mookie betts: https://www.youtube.com/watch?v=13_5FPCngS8&ab_channel=DodgerBlue
freddie freeman: https://www.youtube.com/watch?v=8TIHcCcYXvI&ab_channel=DodgerBlue
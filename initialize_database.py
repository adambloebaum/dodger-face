from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.exc import SQLAlchemyError
import logging

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
        # Create database engine
        engine = create_engine('sqlite:///face_encodings_db.db')
        Base.metadata.create_all(engine)
        logging.info("Database initialized successfully.")
        return engine
    except SQLAlchemyError as e:
        logging.error("Error occurred during engine creation or table initialization: %s", e)
        raise

if __name__ == "__main__":
    # Initialize the database when this script is run
    initialize_database()

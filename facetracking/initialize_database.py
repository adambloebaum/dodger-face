from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.exc import SQLAlchemyError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

Base = declarative_base()

class Face(Base):
    __tablename__ = 'faces'
    name = Column(String, primary_key=True)
    encoding = Column(Text)

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

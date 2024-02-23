from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

Base = declarative_base()

class Face(Base):
    __tablename__ = 'faces'
    id = Column(Integer, primary_key=True)
    name  = Column(String)
    encoding = Column(Text)

try:
    # Create database engine
    engine = create_engine('sqlite:///face_encodings_db.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
except SQLAlchemyError as e:
    print("Error occurred during engine creation or table initialization:", e)

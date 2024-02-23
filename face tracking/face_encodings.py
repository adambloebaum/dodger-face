from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Face(Base):
    __tablename__ = 'faces'
    id = Column(Integer, primary_key=True)
    # store face encodings as a test (JSON seralized)
    encoding = Column(Text)
    # name = Column(String)

engine = create_engine('sqlite:///face_encodings.db')
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)

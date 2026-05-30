from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base import Base


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    cohort_id = Column(Integer, ForeignKey("cohorts.id"), nullable=True)
    demographics = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    cohort = relationship("Cohort", back_populates="students")
    predictions = relationship("Prediction", back_populates="student")

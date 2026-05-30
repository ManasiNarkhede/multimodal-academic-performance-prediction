from sqlalchemy import Column, Integer, DateTime, ForeignKey, Float, Text, String
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    at_risk_probability = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)
    explanation_text = Column(Text, nullable=True)
    text_score = Column(Float, nullable=True)
    tabular_score = Column(Float, nullable=True)
    behavioral_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    student = relationship("Student", back_populates="predictions")
    user = relationship("User", back_populates="predictions")

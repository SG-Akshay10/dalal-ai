import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from .db import Base

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    subscription_tier = Column(String, default="free")
    
    jobs = relationship("AnalysisJob", back_populates="user")

class CompanyNode(Base):
    __tablename__ = "companies"
    symbol = Column(String, primary_key=True, index=True)
    exchange = Column(String)
    name = Column(String)
    sector_id = Column(UUID(as_uuid=True))
    cin = Column(String)
    market_cap = Column(Float)
    
    jobs = relationship("AnalysisJob", back_populates="company")

class AnalysisJob(Base):
    __tablename__ = "jobs"
    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    symbol = Column(String, ForeignKey("companies.symbol"))
    status = Column(String, default="pending")
    progress_pct = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    report_json = Column(JSONB, nullable=True)
    
    user = relationship("User", back_populates="jobs")
    company = relationship("CompanyNode", back_populates="jobs")

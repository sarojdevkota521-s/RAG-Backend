import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InterviewBooking(Base):
    """Interview booking records stored in SQLite3."""

    __tablename__ = "interview_bookings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    date: Mapped[str] = mapped_column(String(50), nullable=False)  # ISO Date e.g. "2026-11-20" or human-readable
    time: Mapped[str] = mapped_column(String(50), nullable=False)  # E.g. "14:00" or "2:00 PM"
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="confirmed")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    def __repr__(self) -> str:
        return f"<InterviewBooking(id={self.id}, name={self.name}, email={self.email}, date={self.date}, time={self.time})>"


from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, EmailStr, ConfigDict


class BookingCreate(BaseModel):
    """Schema for booking an interview."""
    name: str = Field(..., min_length=2, max_length=120, description="Full candidate name")
    email: EmailStr = Field(..., description="Valid candidate email address")
    date: str = Field(..., min_length=4, max_length=50, description="Preferred interview date (e.g., 2026-11-20 or 'next Monday')")
    time: str = Field(..., min_length=2, max_length=50, description="Preferred interview time (e.g., 14:00 or '2:00 PM')")
    notes: Optional[str] = Field(None, max_length=1000, description="Optional notes or position applied for")
    conversation_id: Optional[str] = Field(None, description="Optional associated chat session ID")


class BookingResponse(BaseModel):
    """Schema for returning booking confirmation."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: Optional[str] = None
    name: str
    email: str
    date: str
    time: str
    notes: Optional[str] = None
    status: str
    created_at: datetime


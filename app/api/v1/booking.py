import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.booking import InterviewBooking
from app.schemas.booking import BookingCreate, BookingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bookings", tags=["Interview Bookings"])


@router.post(
    "/",
    response_model=BookingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create or register an interview booking"
)
def create_booking(
    booking_in: BookingCreate,
    db: Session = Depends(get_db)
) -> BookingResponse:
    """Manually book an interview slot and store in SQLite3."""
    booking = InterviewBooking(
        conversation_id=booking_in.conversation_id,
        name=booking_in.name,
        email=booking_in.email,
        date=booking_in.date,
        time=booking_in.time,
        notes=booking_in.notes,
        status="confirmed"
    )
    try:
        db.add(booking)
        db.commit()
        db.refresh(booking)
        return BookingResponse.model_validate(booking)
    except Exception as e:
        db.rollback()
        logger.error("Failed to create booking: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record interview booking in database."
        )


@router.get(
    "/",
    response_model=List[BookingResponse],
    summary="List all interview bookings"
)
def list_bookings(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> List[BookingResponse]:
    """Retrieve all candidate interview bookings stored in SQLite3."""
    bookings = db.query(InterviewBooking).order_by(InterviewBooking.created_at.desc()).offset(skip).limit(limit).all()
    return [BookingResponse.model_validate(b) for b in bookings]


@router.get(
    "/{booking_id}",
    response_model=BookingResponse,
    summary="Get details of a specific interview booking"
)
def get_booking(
    booking_id: str,
    db: Session = Depends(get_db)
) -> BookingResponse:
    """Retrieve details for a specific interview booking by ID."""
    booking = db.query(InterviewBooking).filter(InterviewBooking.id == booking_id).first()
    if not booking:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Booking with ID '{booking_id}' not found."
        )
    return BookingResponse.model_validate(booking)


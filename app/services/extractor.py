import io
import logging
from dataclasses import dataclass
from typing import List, Optional
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """Represents text extracted from a specific page/section."""
    text: str
    page_number: Optional[int] = None


class DocumentExtractor:
    """Service for extracting clean text from uploaded .pdf and .txt files."""

    @staticmethod
    def extract_text_from_txt(content_bytes: bytes) -> List[ExtractedPage]:
        """Extract text from a plain text file, handling multiple encodings."""
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                decoded = content_bytes.decode(encoding)
                cleaned = decoded.strip()
                if not cleaned:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="The uploaded text file is empty."
                    )
                return [ExtractedPage(text=cleaned, page_number=1)]
            except UnicodeDecodeError:
                continue

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to decode text file. Ensure it is encoded in valid UTF-8 or ASCII."
        )

    @staticmethod
    def extract_text_from_pdf(content_bytes: bytes) -> List[ExtractedPage]:
        """Extract text from a PDF file using PyMuPDF (fitz) or pypdf."""
        pages: List[ExtractedPage] = []

        # Try PyMuPDF (fitz) first for fastest and highest-fidelity extraction
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=content_bytes, filetype="pdf")
            for page_index in range(len(doc)):
                page = doc[page_index]
                text = page.get_text()
                if text and text.strip():
                    pages.append(ExtractedPage(text=text.strip(), page_number=page_index + 1))
            doc.close()
        except ImportError:
            # Fallback to pypdf if fitz is not installed
            try:
                import pypdf
                pdf_reader = pypdf.PdfReader(io.BytesIO(content_bytes))
                for page_index, page in enumerate(pdf_reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(ExtractedPage(text=text.strip(), page_number=page_index + 1))
            except Exception as e:
                logger.error("pypdf extraction failed: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to read PDF file: {str(e)}"
                )

        if not pages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No extractable text found in the PDF. Scanned images without OCR are not supported."
            )

        return pages

    @classmethod
    def extract(cls, filename: str, content_bytes: bytes) -> List[ExtractedPage]:
        """Extracts text based on the file extension."""
        lower_name = filename.lower()
        if lower_name.endswith(".txt"):
            return cls.extract_text_from_txt(content_bytes)
        elif lower_name.endswith(".pdf"):
            return cls.extract_text_from_pdf(content_bytes)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Only .pdf and .txt files are allowed."
            )


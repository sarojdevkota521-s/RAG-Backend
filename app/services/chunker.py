import re
import logging
from dataclasses import dataclass
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.schemas.document import ChunkingStrategy
from app.services.extractor import ExtractedPage

logger = logging.getLogger(__name__)


@dataclass
class ChunkItem:
    """Represents an individual chunk prepared for vector embedding and storage."""
    text: str
    chunk_index: int
    page_number: Optional[int]
    char_count: int


class DocumentChunker:
    """
    Applies selectable chunking strategies to extracted text:
    1. recursive_character: Recursive splitting with hierarchical delimiters and overlap.
    2. paragraph_semantic: Cohesive paragraph-based chunking with sliding context window.
    """

    @staticmethod
    def chunk_recursive(
        pages: List[ExtractedPage],
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[ChunkItem]:
        """
        Strategy 1: Recursive Character Splitting.
        Recursively splits text on ['\n\n', '\n', ' ', ''] preserving structure and overlap.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

        chunks: List[ChunkItem] = []
        chunk_counter = 0

        for page in pages:
            split_texts = splitter.split_text(page.text)
            for text in split_texts:
                cleaned = text.strip()
                if cleaned:
                    chunks.append(
                        ChunkItem(
                            text=cleaned,
                            chunk_index=chunk_counter,
                            page_number=page.page_number,
                            char_count=len(cleaned)
                        )
                    )
                    chunk_counter += 1

        return chunks

    @staticmethod
    def chunk_paragraph_semantic(
        pages: List[ExtractedPage],
        target_size: int = 500,
        overlap_size: int = 50
    ) -> List[ChunkItem]:
        """
        Strategy 2: Paragraph / Semantic Chunking.
        Splits on semantic paragraph boundaries (double newlines or bullet lists)
        and groups complete paragraphs together up to target_size with sliding overlap,
        avoiding mid-sentence breaks.
        """
        chunks: List[ChunkItem] = []
        chunk_counter = 0

        for page in pages:
            # Split raw text by paragraph boundaries
            raw_paragraphs = re.split(r'\n\s*\n+', page.text)
            paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

            if not paragraphs:
                continue

            current_chunk_paras: List[str] = []
            current_len = 0

            for para in paragraphs:
                para_len = len(para)

                # If adding this paragraph exceeds target_size and we already have content:
                if current_len + para_len > target_size and current_chunk_paras:
                    chunk_text = "\n\n".join(current_chunk_paras).strip()
                    if chunk_text:
                        chunks.append(
                            ChunkItem(
                                text=chunk_text,
                                chunk_index=chunk_counter,
                                page_number=page.page_number,
                                char_count=len(chunk_text)
                            )
                        )
                        chunk_counter += 1

                    # Provide sliding overlap by keeping the last paragraph if under overlap_size
                    if current_chunk_paras and len(current_chunk_paras[-1]) <= overlap_size:
                        current_chunk_paras = [current_chunk_paras[-1], para]
                        current_len = sum(len(p) for p in current_chunk_paras) + 2
                    else:
                        current_chunk_paras = [para]
                        current_len = para_len
                else:
                    current_chunk_paras.append(para)
                    current_len += para_len + 2  # account for "\n\n"

            # Flush any remaining paragraphs
            if current_chunk_paras:
                chunk_text = "\n\n".join(current_chunk_paras).strip()
                if chunk_text:
                    chunks.append(
                        ChunkItem(
                            text=chunk_text,
                            chunk_index=chunk_counter,
                            page_number=page.page_number,
                            char_count=len(chunk_text)
                        )
                    )
                    chunk_counter += 1

        return chunks

    @classmethod
    def split(
        cls,
        pages: List[ExtractedPage],
        strategy: ChunkingStrategy,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[ChunkItem]:
        """Dispatch to the selected chunking strategy."""
        if strategy == ChunkingStrategy.PARAGRAPH_SEMANTIC:
            return cls.chunk_paragraph_semantic(pages, target_size=chunk_size, overlap_size=chunk_overlap)
        else:
            return cls.chunk_recursive(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


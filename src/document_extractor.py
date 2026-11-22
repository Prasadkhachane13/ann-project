"""
Document Extractor Module

This module handles text extraction from various document formats:
- TXT files
- PDF files (using PyMuPDF)
- DOCX files (using python-docx)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from typing import Tuple, Optional
import io


class DocumentExtractor:
    """Extract text from various document formats."""

    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    @classmethod
    def is_supported(cls, filename: str) -> bool:
        """Check if file extension is supported."""
        return Path(filename).suffix.lower() in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def get_extension(cls, filename: str) -> str:
        """Get file extension in lowercase."""
        return Path(filename).suffix.lower()

    @classmethod
    def validate_file(cls, file_bytes: bytes, filename: str) -> Tuple[bool, str]:
        """
        Validate uploaded file.

        Returns:
            Tuple of (is_valid, error_message).
        """
        # Check file size
        if len(file_bytes) > cls.MAX_FILE_SIZE:
            return False, f"File size exceeds maximum allowed size of {cls.MAX_FILE_SIZE / (1024*1024):.0f} MB"

        # Check extension
        if not cls.is_supported(filename):
            return False, f"Unsupported file format. Supported formats: {', '.join(cls.SUPPORTED_EXTENSIONS)}"

        # Check if file is empty
        if len(file_bytes) == 0:
            return False, "File is empty"

        return True, ""

    @classmethod
    def extract_text(cls, file_bytes: bytes, filename: str) -> Tuple[str, Optional[str]]:
        """
        Extract text from file based on extension.

        Returns:
            Tuple of (extracted_text, error_message).
            If successful, error_message is None.
        """
        # Validate first
        is_valid, error = cls.validate_file(file_bytes, filename)
        if not is_valid:
            return "", error

        ext = cls.get_extension(filename)

        try:
            if ext == '.txt':
                return cls._extract_txt(file_bytes), None
            elif ext == '.pdf':
                return cls._extract_pdf(file_bytes)
            elif ext == '.docx':
                return cls._extract_docx(file_bytes)
            else:
                return "", f"Unsupported file format: {ext}"
        except Exception as e:
            return "", f"Error extracting text: {str(e)}"

    @classmethod
    def _extract_txt(cls, file_bytes: bytes) -> str:
        """Extract text from TXT file."""
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        # Fallback: replace undecodable characters
        return file_bytes.decode('utf-8', errors='replace')

    @classmethod
    def _extract_pdf(cls, file_bytes: bytes) -> Tuple[str, Optional[str]]:
        """Extract text from PDF file using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return "", "PyMuPDF not installed. Install with: pip install PyMuPDF"

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)

            doc.close()

            full_text = "\n\n".join(text_parts)

            if not full_text.strip():
                return "", "PDF contains no extractable text (may be scanned/image-based)"

            return full_text, None

        except Exception as e:
            return "", f"PDF extraction error: {str(e)}"

    @classmethod
    def _extract_docx(cls, file_bytes: bytes) -> Tuple[str, Optional[str]]:
        """Extract text from DOCX file using python-docx."""
        try:
            from docx import Document
        except ImportError:
            return "", "python-docx not installed. Install with: pip install python-docx"

        try:
            doc = Document(io.BytesIO(file_bytes))
            text_parts = []

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)

            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_parts.append(" | ".join(row_text))

            full_text = "\n\n".join(text_parts)

            if not full_text.strip():
                return "", "DOCX file contains no text content"

            return full_text, None

        except Exception as e:
            return "", f"DOCX extraction error: {str(e)}"

    @classmethod
    def extract_from_multiple(cls, files: list) -> list:
        """
        Extract text from multiple files.

        Args:
            files: List of tuples (file_bytes, filename).

        Returns:
            List of dicts with keys: filename, text, error, success.
        """
        results = []
        for file_bytes, filename in files:
            text, error = cls.extract_text(file_bytes, filename)
            results.append({
                'filename': filename,
                'text': text,
                'error': error,
                'success': error is None
            })
        return results


def main():
    """Test document extractor."""
    print("Testing Document Extractor...")

    # Test TXT
    txt_content = b"This is a test text file.\nIt has multiple lines.\nAnd some content."
    text, error = DocumentExtractor.extract_text(txt_content, "test.txt")
    print(f"TXT extraction: {'Success' if not error else 'Failed'} - {error or text[:50]}")

    # Test unsupported format
    text, error = DocumentExtractor.extract_text(b"test", "test.xyz")
    print(f"Unsupported format: {'Success' if not error else 'Failed'} - {error}")


if __name__ == "__main__":
    main()
"""PDF text extraction with OCR fallback."""

import os
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


def extract_text_from_pdf(pdf_path: str | Path) -> dict:
    """Extract text from a PDF file.

    Uses PyMuPDF for direct text extraction first. Falls back to
    Tesseract OCR for pages with little or no extractable text.

    Returns a dict with metadata and per-page text.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    pages = []
    ocr_used = False

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if len(text) < 50:
            ocr_text = _ocr_page(pdf_path, page_num)
            if ocr_text and len(ocr_text) > len(text):
                text = ocr_text
                ocr_used = True

        pages.append({
            "page_number": page_num + 1,
            "text": text,
        })

    doc.close()

    return {
        "filename": pdf_path.name,
        "filepath": str(pdf_path.resolve()),
        "num_pages": len(pages),
        "ocr_used": ocr_used,
        "pages": pages,
    }


def _ocr_page(pdf_path: Path, page_num: int) -> str:
    """Run Tesseract OCR on a single PDF page."""
    try:
        images = convert_from_path(
            str(pdf_path),
            first_page=page_num + 1,
            last_page=page_num + 1,
            dpi=300,
        )
        if images:
            return pytesseract.image_to_string(images[0]).strip()
    except Exception:
        pass
    return ""


def scan_pdf_folder(folder_path: str | Path) -> list[Path]:
    """Return all PDF files in a directory (non-recursive)."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(folder.glob("*.pdf"))

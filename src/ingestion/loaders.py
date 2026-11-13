"""
Multi-format document loading pipeline.

Supports:
- PDF (text + OCR fallback)
- DOCX
- TXT
- Markdown
- HTML

Includes metadata extraction.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import fitz
import pytesseract
from langchain_core.documents import Document
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ==================================================
# Windows Tesseract Path
# ==================================================
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


class DocumentLoader:
    """
    Unified document loader supporting
    multiple document formats.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".html",
        ".htm",
    }

    # ==================================================
    # Load Single File
    # ==================================================
    def load_file(
        self,
        file_path: str | Path,
    ) -> list[Document]:
        """
        Load a single file and return
        LangChain documents with metadata.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        ext = path.suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}"
            )

        logger.info(
            "loading_document",
            file=str(path),
            type=ext,
        )

        loader_map = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".txt": self._load_text,
            ".md": self._load_markdown,
            ".html": self._load_html,
            ".htm": self._load_html,
        }

        docs = loader_map[ext](path)

        # Add metadata
        for doc in docs:
            doc.metadata.update(
                {
                    "source": str(path),
                    "file_name": path.name,
                    "file_type": ext,
                    "file_size_bytes": path.stat().st_size,
                    "ingested_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "content_hash": hashlib.sha256(
                        doc.page_content.encode(
                            "utf-8"
                        )
                    ).hexdigest()[:16],
                }
            )

        logger.info(
            "document_loaded",
            file=str(path),
            chunks=len(docs),
        )

        return docs

    # ==================================================
    # Load Directory
    # ==================================================
    def load_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
        extensions: Optional[
            set[str]
        ] = None,
    ) -> list[Document]:
        """
        Load all supported files
        from a directory.
        """

        path = Path(dir_path)

        if not path.is_dir():
            raise NotADirectoryError(
                f"Not a directory: {path}"
            )

        allowed = (
            extensions
            or self.SUPPORTED_EXTENSIONS
        )

        pattern = (
            "**/*"
            if recursive
            else "*"
        )

        files = [
            f
            for f in path.glob(pattern)
            if (
                f.is_file()
                and f.suffix.lower()
                in allowed
            )
        ]

        logger.info(
            "loading_directory",
            dir=str(path),
            file_count=len(files),
        )

        all_docs = []

        for file in sorted(files):
            try:
                docs = self.load_file(file)
                all_docs.extend(docs)

            except Exception as e:
                logger.error(
                    "file_load_error",
                    file=str(file),
                    error=str(e),
                )

        return all_docs

    # ==================================================
    # PDF Loader
    # ==================================================
    def _load_pdf(
        self,
        path: Path,
    ) -> list[Document]:
        """
        Load PDF.

        First tries normal text extraction.
        Falls back to OCR if PDF has no text.
        """

        from langchain_community.document_loaders import (
            PyPDFLoader,
        )

        try:
            loader = PyPDFLoader(
                str(path)
            )

            docs = loader.load()

            combined_text = (
                " ".join(
                    doc.page_content
                    for doc in docs
                ).strip()
            )

            # If enough text found,
            # use standard parser
            if len(combined_text) > 100:

                logger.info(
                    "pdf_text_extraction_success",
                    file=str(path),
                )

                return docs

            logger.warning(
                "pdf_has_no_text_using_ocr",
                file=str(path),
            )

        except Exception as e:

            logger.warning(
                "pdf_loader_failed_using_ocr",
                file=str(path),
                error=str(e),
            )

        # OCR fallback
        return self._ocr_pdf(path)

    # ==================================================
    # OCR PDF
    # ==================================================
    def _ocr_pdf(
        self,
        path: Path,
    ) -> list[Document]:
        """
        OCR extraction for
        scanned/image PDFs.
        """

        logger.info(
            "ocr_pdf_started",
            file=str(path),
        )

        pdf = fitz.open(str(path))

        pages = []

        try:

            for page_num in range(
                len(pdf)
            ):

                page = pdf[
                    page_num
                ]

                pix = page.get_pixmap(
                    dpi=300
                )

                image = (
                    Image.frombytes(
                        "RGB",
                        [
                            pix.width,
                            pix.height,
                        ],
                        pix.samples,
                    )
                )

                text = (
                    pytesseract
                    .image_to_string(
                        image
                    )
                    .strip()
                )

                if text:

                    pages.append(
                        Document(
                            page_content=text,
                            metadata={
                                "page":
                                    page_num
                                    + 1,
                                "ocr":
                                    True,
                            },
                        )
                    )

            logger.info(
                "ocr_pdf_complete",
                file=str(path),
                pages_extracted=len(
                    pages
                ),
            )

            return pages

        finally:
            # Prevent temp-file lock
            pdf.close()

    # ==================================================
    # DOCX
    # ==================================================
    def _load_docx(
        self,
        path: Path,
    ) -> list[Document]:

        from langchain_community.document_loaders import (
            Docx2txtLoader,
        )

        loader = Docx2txtLoader(
            str(path)
        )

        return loader.load()

    # ==================================================
    # TXT
    # ==================================================
    def _load_text(
        self,
        path: Path,
    ) -> list[Document]:

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        return [
            Document(
                page_content=text
            )
        ]

    # ==================================================
    # Markdown
    # ==================================================
    def _load_markdown(
        self,
        path: Path,
    ) -> list[Document]:

        from langchain_community.document_loaders import (
            UnstructuredMarkdownLoader,
        )

        loader = (
            UnstructuredMarkdownLoader(
                str(path)
            )
        )

        return loader.load()

    # ==================================================
    # HTML
    # ==================================================
    def _load_html(
        self,
        path: Path,
    ) -> list[Document]:

        from langchain_community.document_loaders import (
            BSHTMLLoader,
        )

        loader = BSHTMLLoader(
            str(path)
        )

        return loader.load()
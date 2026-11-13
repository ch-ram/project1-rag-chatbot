"""
Document chunking strategies.
Optimized for:
- OCR PDFs
- Research papers
- RAG retrieval quality
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import re

from langchain_core.documents import (
    Document,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from src.utils.config import (
    get_settings,
)
from src.utils.logging import (
    get_logger,
)

logger = get_logger(__name__)


class ChunkingStrategy(
    str,
    Enum,
):
    RECURSIVE = (
        "recursive"
    )

    SEMANTIC = (
        "semantic"
    )

    SENTENCE = (
        "sentence"
    )


# ==================================================
# Base Chunker
# ==================================================
class BaseChunker(
    ABC
):
    @abstractmethod
    def split(
        self,
        documents:
        list[Document],
    ) -> list[
        Document
    ]:
        ...


# ==================================================
# Recursive Chunker
# ==================================================
class RecursiveChunker(
    BaseChunker
):
    """
    Best general-purpose chunker.
    Optimized for OCR docs.
    """

    def __init__(
        self,
        chunk_size:
        int = 700,
        chunk_overlap:
        int = 150,
    ):

        self.splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=
                    chunk_size,

                chunk_overlap=
                    chunk_overlap,

                length_function=
                    len,

                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    "? ",
                    "! ",
                    "; ",
                    ", ",
                    " ",
                    "",
                ],

                keep_separator=
                    True,
            )
        )

    def _clean_text(
        self,
        text: str,
    ) -> str:
        """
        Cleanup OCR noise.
        """

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    def split(
        self,
        documents:
        list[Document],
    ) -> list[
        Document
    ]:

        cleaned_docs = []

        for doc in documents:

            cleaned_text = (
                self
                ._clean_text(
                    doc.page_content
                )
            )

            cleaned_docs.append(
                Document(
                    page_content=
                        cleaned_text,

                    metadata=
                        doc.metadata,
                )
            )

        chunks = (
            self.splitter
            .split_documents(
                cleaned_docs
            )
        )

        for i, chunk in enumerate(
            chunks
        ):

            chunk.metadata[
                "chunk_index"
            ] = i

            chunk.metadata[
                "chunk_size"
            ] = len(
                chunk.page_content
            )

            chunk.metadata[
                "chunking"
            ] = (
                "recursive"
            )

        logger.info(
            "documents_chunked",
            total_chunks=
                len(chunks),
        )

        return chunks


# ==================================================
# Semantic Chunker
# ==================================================
class SemanticChunker(
    BaseChunker
):

    def __init__(
        self,
        chunk_size:
        int = 700,
        chunk_overlap:
        int = 150,
    ):

        self.fallback = (
            RecursiveChunker(
                chunk_size,
                chunk_overlap,
            )
        )

    def split(
        self,
        documents:
        list[Document],
    ) -> list[
        Document
    ]:

        try:

            from langchain_experimental.text_splitter import (
                SemanticChunker
                as LC_Semantic
            )

            from langchain_openai import (
                OpenAIEmbeddings
            )

            settings = (
                get_settings()
            )

            embeddings = (
                OpenAIEmbeddings(
                    model=
                        settings
                        .openai_embedding_model,

                    dimensions=
                        settings
                        .openai_embedding_dimensions,
                )
            )

            splitter = (
                LC_Semantic(
                    embeddings=
                        embeddings,

                    breakpoint_threshold_type=
                        "percentile",

                    breakpoint_threshold_amount=
                        85,
                )
            )

            chunks = (
                splitter
                .split_documents(
                    documents
                )
            )

            for i, chunk in enumerate(
                chunks
            ):

                chunk.metadata[
                    "chunk_index"
                ] = i

                chunk.metadata[
                    "chunk_size"
                ] = len(
                    chunk.page_content
                )

                chunk.metadata[
                    "chunking"
                ] = (
                    "semantic"
                )

            return chunks

        except ImportError:

            logger.warning(
                "semantic_chunker_fallback"
            )

            return (
                self
                .fallback
                .split(
                    documents
                )
            )


# ==================================================
# Sentence Chunker
# ==================================================
class SentenceChunker(
    BaseChunker
):

    def __init__(
        self,
        chunk_overlap:
        int = 50,

        tokens_per_chunk:
        int = 180,
    ):

        self.splitter = (
            SentenceTransformersTokenTextSplitter(
                chunk_overlap=
                    chunk_overlap,

                tokens_per_chunk=
                    tokens_per_chunk,
            )
        )

    def split(
        self,
        documents:
        list[Document],
    ) -> list[
        Document
    ]:

        chunks = (
            self.splitter
            .split_documents(
                documents
            )
        )

        for i, chunk in enumerate(
            chunks
        ):

            chunk.metadata[
                "chunk_index"
            ] = i

            chunk.metadata[
                "chunk_size"
            ] = len(
                chunk.page_content
            )

            chunk.metadata[
                "chunking"
            ] = (
                "sentence"
            )

        return chunks


# ==================================================
# Factory
# ==================================================
def get_chunker(
    strategy:
    str | None = None,
) -> BaseChunker:

    settings = (
        get_settings()
    )

    strategy = (
        strategy
        or settings
        .chunking_strategy
    )

    chunker_map = {
        ChunkingStrategy
        .RECURSIVE:
            lambda:
            RecursiveChunker(),

        ChunkingStrategy
        .SEMANTIC:
            lambda:
            SemanticChunker(),

        ChunkingStrategy
        .SENTENCE:
            lambda:
            SentenceChunker(),
    }

    factory = (
        chunker_map.get(
            ChunkingStrategy(
                strategy
            )
        )
    )

    if not factory:

        raise ValueError(
            f"Unknown chunking strategy: {strategy}"
        )

    logger.info(
        "chunker_initialized",
        strategy=
            strategy,
    )

    return factory()
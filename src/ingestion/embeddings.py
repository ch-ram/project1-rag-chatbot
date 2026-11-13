"""
Embedding generation and FAISS index management.
Production OpenAI embedding pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingManager:
    """
    Production embedding manager.

    Features:
    - OpenAI embeddings
    - FAISS indexing
    - Persistence
    - Batch embedding
    - Incremental document uploads
    """

    def __init__(self):
        settings = get_settings()

        self.index_path = Path(
            settings.faiss_index_path
        )

        self.dimension = (
            settings.openai_embedding_dimensions
        )

        self.embeddings = (
            OpenAIEmbeddings(
                model=(
                    settings
                    .openai_embedding_model
                ),
                dimensions=(
                    settings
                    .openai_embedding_dimensions
                ),
            )
        )

        self.index: Optional[
            faiss.Index
        ] = None

        self.documents: list[
            Document
        ] = []

        logger.info(
            "embedding_provider_initialized",
            provider="openai",
            model=(
                settings
                .openai_embedding_model
            ),
            dimension=self.dimension,
        )

    # ==================================================
    # Create Index
    # ==================================================
    def create_index(
        self,
        documents: list[Document],
    ) -> faiss.Index:
        """
        Create FAISS index
        from documents.
        """

        logger.info(
            "creating_embeddings",
            total=len(documents),
        )

        texts = [
            doc.page_content
            for doc in documents
        ]

        vectors = (
            self._batch_embed(
                texts
            )
        )

        vectors_np = np.array(
            vectors,
            dtype=np.float32,
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(
            vectors_np
        )

        vector_count = len(
            vectors_np
        )

        # Small corpus
        if vector_count < 1000:
            self.index = (
                faiss.IndexFlatIP(
                    self.dimension
                )
            )

        # Larger corpus
        else:
            settings = (
                get_settings()
            )

            nlist = min(
                settings.faiss_nlist,
                max(
                    1,
                    vector_count // 10,
                ),
            )

            quantizer = (
                faiss.IndexFlatIP(
                    self.dimension
                )
            )

            self.index = (
                faiss.IndexIVFFlat(
                    quantizer,
                    self.dimension,
                    nlist,
                    faiss.METRIC_INNER_PRODUCT,
                )
            )

            self.index.train(
                vectors_np
            )

            self.index.nprobe = (
                settings
                .faiss_nprobe
            )

        self.index.add(
            vectors_np
        )

        self.documents = (
            documents
        )

        logger.info(
            "index_created",
            vectors=vector_count,
            dimension=self.dimension,
        )

        return self.index

    # ==================================================
    # Add Documents
    # ==================================================
    def add_documents(
        self,
        documents: list[Document],
    ) -> int:
        """
        Add documents to
        existing FAISS index.

        Used by upload API.
        """

        if not documents:
            logger.warning(
                "no_documents_to_add"
            )
            return 0

        logger.info(
            "adding_documents",
            count=len(documents),
        )

        texts = [
            doc.page_content
            for doc in documents
        ]

        vectors = (
            self._batch_embed(
                texts
            )
        )

        vectors_np = np.array(
            vectors,
            dtype=np.float32,
        )

        faiss.normalize_L2(
            vectors_np
        )

        # Create index if empty
        if self.index is None:

            logger.info(
                "creating_new_index"
            )

            self.index = (
                faiss.IndexFlatIP(
                    self.dimension
                )
            )

        # Add vectors
        self.index.add(
            vectors_np
        )

        # Save docs
        self.documents.extend(
            documents
        )

        logger.info(
            "documents_added",
            added_count=len(
                documents
            ),
            total_documents=len(
                self.documents
            ),
            total_vectors=(
                self.index.ntotal
            ),
        )

        return len(documents)

    # ==================================================
    # Search
    # ==================================================
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[
        tuple[Document, float]
    ]:
        """
        Semantic search.
        """

        if (
            self.index is None
            or self.index.ntotal == 0
        ):
            logger.warning(
                "empty_index_search"
            )
            return []

        query_embedding = (
            self.embeddings
            .embed_query(query)
        )

        query_vector = (
            np.array(
                [query_embedding],
                dtype=np.float32,
            )
        )

        faiss.normalize_L2(
            query_vector
        )

        scores, indices = (
            self.index.search(
                query_vector,
                min(
                    top_k,
                    self.index.ntotal,
                ),
            )
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0],
        ):
            if idx < 0:
                continue

            results.append(
                (
                    self.documents[idx],
                    float(score),
                )
            )

        logger.info(
            "search_complete",
            query=query[:50],
            results=len(results),
        )

        return results

    # ==================================================
    # Save Index
    # ==================================================
    def save(self):
        """
        Persist FAISS index.
        """

        if self.index is None:
            raise ValueError(
                "No index to save."
            )

        self.index_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(
                self.index_path
                / "index.faiss"
            ),
        )

        docs_data = [
            {
                "content":
                    doc.page_content,
                "metadata":
                    doc.metadata,
            }
            for doc
            in self.documents
        ]

        with open(
            self.index_path
            / "documents.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                docs_data,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        logger.info(
            "index_saved",
            path=str(
                self.index_path
            ),
        )

    # ==================================================
    # Load Index
    # ==================================================
    def load(self) -> bool:
        """
        Load persisted index.
        """

        index_file = (
            self.index_path
            / "index.faiss"
        )

        docs_file = (
            self.index_path
            / "documents.json"
        )

        if (
            not index_file.exists()
            or not docs_file.exists()
        ):
            logger.warning(
                "no_saved_index",
                path=str(
                    self.index_path
                ),
            )
            return False

        self.index = (
            faiss.read_index(
                str(index_file)
            )
        )

        with open(
            docs_file,
            "r",
            encoding="utf-8",
        ) as f:
            docs_data = json.load(f)

        self.documents = [
            Document(
                page_content=d[
                    "content"
                ],
                metadata=d[
                    "metadata"
                ],
            )
            for d
            in docs_data
        ]

        logger.info(
            "index_loaded",
            vectors=(
                self.index.ntotal
            ),
            documents=len(
                self.documents
            ),
        )

        return True

    # ==================================================
    # Batch Embeddings
    # ==================================================
    def _batch_embed(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[
        list[float]
    ]:
        """
        Batch embedding generation.
        """

        all_embeddings = []

        for i in range(
            0,
            len(texts),
            batch_size,
        ):
            batch = texts[
                i: i + batch_size
            ]

            embeddings = (
                self.embeddings
                .embed_documents(
                    batch
                )
            )

            all_embeddings.extend(
                embeddings
            )

            logger.debug(
                "embedding_batch",
                progress=(
                    f"{min(i + batch_size, len(texts))}"
                    f"/{len(texts)}"
                ),
            )

        return all_embeddings
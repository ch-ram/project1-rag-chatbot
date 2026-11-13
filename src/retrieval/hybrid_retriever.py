"""
Improved hybrid retrieval for OCR-heavy RAG.

Dense (FAISS) +
Sparse (BM25) +
FlashRank reranking
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from langchain_core.documents import Document

from src.ingestion.embeddings import EmbeddingManager
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RetrievalStrategy(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


# ==================================================
# BM25
# ==================================================
class BM25Retriever:

    def __init__(
        self,
        documents: list[Document],
    ):
        from rank_bm25 import BM25Okapi

        self.documents = documents

        tokenized = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(tokenized)

        logger.info(
            "bm25_initialized",
            doc_count=len(documents),
        )

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple[Document, float]]:

        query_tokens = (
            query.lower().split()
        )

        scores = (
            self.bm25
            .get_scores(
                query_tokens
            )
        )

        top_indices = (
            np.argsort(scores)[::-1][:top_k]
        )

        results = []

        for idx in top_indices:

            score = float(
                scores[idx]
            )

            if score > 0:
                results.append(
                    (
                        self.documents[idx],
                        score,
                    )
                )

        return results


# ==================================================
# Reranker
# ==================================================
class CrossEncoderReranker:

    def __init__(
        self,
        model_name:
        str = (
            "ms-marco-MiniLM-L-12-v2"
        ),
    ):

        try:
            from flashrank import (
                Ranker
            )

            self.ranker = (
                Ranker(
                    model_name=
                        model_name
                )
            )

            self.enabled = True

            logger.info(
                "reranker_initialized",
                backend=
                    "flashrank",
                model=
                    model_name,
            )

        except ImportError:

            self.enabled = False

            logger.warning(
                "reranker_disabled"
            )

    def rerank(
        self,
        query: str,
        documents:
        list[
            tuple[
                Document,
                float
            ]
        ],
        top_k: int = 10,
    ):

        if (
            not self.enabled
            or not documents
        ):
            return documents[:top_k]

        from flashrank import (
            RerankRequest
        )

        passages = []

        for i, (
            doc,
            score,
        ) in enumerate(
            documents
        ):
            passages.append(
                {
                    "id": i,
                    "text":
                        doc.page_content,
                    "meta":
                        doc.metadata,
                }
            )

        request = (
            RerankRequest(
                query=query,
                passages=passages,
            )
        )

        results = (
            self.ranker
            .rerank(
                request
            )
        )

        reranked = []

        for result in (
            results[:top_k]
        ):

            idx = (
                result["id"]
            )

            reranked.append(
                (
                    documents[idx][0],
                    float(
                        result[
                            "score"
                        ]
                    ),
                )
            )

        return reranked


# ==================================================
# Hybrid Retriever
# ==================================================
class HybridRetriever:

    def __init__(
        self,
        embedding_manager:
        EmbeddingManager,
        strategy:
        Optional[str] = None,
    ):

        settings = (
            get_settings()
        )

        self.embedding_manager = (
            embedding_manager
        )

        self.strategy = (
            RetrievalStrategy(
                strategy
                or settings
                .retrieval_strategy
            )
        )

        # Better defaults for OCR
        self.bm25_weight = 0.7
        self.dense_weight = 0.3

        self.top_k = (
            settings
            .retrieval_top_k
        )

        self.reranker = (
            CrossEncoderReranker()
        )

        self.bm25 = None

        if (
            embedding_manager
            .documents
        ):
            self.bm25 = (
                BM25Retriever(
                    embedding_manager
                    .documents
                )
            )

    # ==================================================
    # Retrieve
    # ==================================================
    def retrieve(
        self,
        query: str,
        top_k:
        Optional[int] = None,
        strategy:
        Optional[str] = None,
    ):

        k = (
            top_k
            or self.top_k
        )

        strat = (
            RetrievalStrategy(
                strategy
            )
            if strategy
            else self.strategy
        )

        logger.info(
            "retrieving",
            query=query,
            strategy=strat,
            top_k=k,
        )

        if strat == (
            RetrievalStrategy
            .DENSE
        ):
            results = (
                self
                ._dense_search(
                    query,
                    k * 4,
                )
            )

        elif strat == (
            RetrievalStrategy
            .SPARSE
        ):
            results = (
                self
                ._sparse_search(
                    query,
                    k * 4,
                )
            )

        else:
            results = (
                self
                ._hybrid_search(
                    query,
                    k * 4,
                )
            )

        # Better reranking
        results = (
            self.reranker
            .rerank(
                query,
                results,
                top_k=k,
            )
        )

        logger.info(
            "retrieval_complete",
            results=
                len(results),
        )

        return results

    # ==================================================
    # Dense
    # ==================================================
    def _dense_search(
        self,
        query: str,
        top_k: int,
    ):

        return (
            self
            .embedding_manager
            .search(
                query,
                top_k,
            )
        )

    # ==================================================
    # Sparse
    # ==================================================
    def _sparse_search(
        self,
        query: str,
        top_k: int,
    ):

        if (
            self.bm25
            is None
        ):
            return (
                self
                ._dense_search(
                    query,
                    top_k,
                )
            )

        return (
            self.bm25
            .search(
                query,
                top_k,
            )
        )

    # ==================================================
    # Hybrid
    # ==================================================
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
    ):

        dense_results = (
            self
            ._dense_search(
                query,
                top_k,
            )
        )

        sparse_results = (
            self
            ._sparse_search(
                query,
                top_k,
            )
        )

        return (
            self
            ._weighted_fusion(
                dense_results,
                sparse_results,
                top_k=top_k,
            )
        )

    # ==================================================
    # Weighted Fusion
    # ==================================================
    def _weighted_fusion(
        self,
        dense_results,
        sparse_results,
        top_k=20,
    ):

        doc_scores = {}
        doc_map = {}

        for rank, (
            doc,
            score,
        ) in enumerate(
            dense_results
        ):

            doc_id = (
                doc.metadata.get(
                    "content_hash"
                )
            )

            weighted = (
                self
                .dense_weight
                * (
                    score
                    /
                    (rank + 1)
                )
            )

            doc_scores[
                doc_id
            ] = (
                doc_scores.get(
                    doc_id,
                    0,
                )
                + weighted
            )

            doc_map[
                doc_id
            ] = doc

        for rank, (
            doc,
            score,
        ) in enumerate(
            sparse_results
        ):

            doc_id = (
                doc.metadata.get(
                    "content_hash"
                )
            )

            weighted = (
                self
                .bm25_weight
                * (
                    score
                    /
                    (rank + 1)
                )
            )

            doc_scores[
                doc_id
            ] = (
                doc_scores.get(
                    doc_id,
                    0,
                )
                + weighted
            )

            doc_map[
                doc_id
            ] = doc

        ranked = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            (
                doc_map[doc_id],
                score,
            )
            for doc_id, score
            in ranked[:top_k]
        ]
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder


class DocumentReranker:
    """Rerank retrieved documents using a cross-encoder model."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.batch_size = batch_size or int(os.getenv("RERANKER_BATCH_SIZE", "16"))
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-score retrieved documents by relevance to the query.
        Returns documents sorted by rerank score (descending).
        """
        if not retrieved_docs:
            return []

        valid_docs: List[Dict[str, Any]] = []
        pairs: List[List[str]] = []

        for doc in retrieved_docs:
            content = (doc.get("content") or "").strip()
            if not content:
                continue
            valid_docs.append(doc)
            pairs.append([query, content])

        if not valid_docs:
            return []

        scores = self.model.predict(pairs, batch_size=self.batch_size)

        reranked_docs: List[Dict[str, Any]] = []
        for doc, score in zip(valid_docs, scores):
            updated_doc = dict(doc)
            updated_doc["retrieval_score"] = doc.get("score")
            updated_doc["score"] = float(score)
            reranked_docs.append(updated_doc)

        reranked_docs.sort(key=lambda d: d["score"], reverse=True)

        for new_rank, doc in enumerate(reranked_docs, start=1):
            doc["rank"] = new_rank

        if top_k is not None:
            return reranked_docs[:top_k]

        return reranked_docs
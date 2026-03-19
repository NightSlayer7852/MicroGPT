from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    SparseVector,
    PointStruct,
)
import uuid
from fastembed import SparseTextEmbedding
from qdrant_client.models import (
    Prefetch,
    FusionQuery,
    Fusion,
)
from typing import List, Dict, Any

from embedding import EmbeddingManager
from vector_store import VectorStore

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager, sparse_model):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.sparse_model = sparse_model

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:

        print(f"Retrieving documents for query: {query}")

        dense_vector = self.embedding_manager.generate_embeddings([query])[0]
        sparse_vector = list(self.sparse_model.embed([query]))[0]

        prefetch = [
            Prefetch(
                query=dense_vector.tolist(),
                using="dense",
                limit=20,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vector.indices.tolist(),
                    values=sparse_vector.values.tolist(),
                ),
                using="sparse",
                limit=20,
            ),
        ]

        results = self.vector_store.client.query_points(
            collection_name=self.vector_store.collection_name,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        retrieved_docs = []

        for rank, result in enumerate(results.points, start=1):
            payload = result.payload or {}

            retrieved_docs.append({
                "rank": rank,
                "score": result.score,
                "content": payload.get("content"),
                "chapter": payload.get("chapter"),
                "page": payload.get("page"),
            })

        return retrieved_docs

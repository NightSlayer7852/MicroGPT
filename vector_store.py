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
class VectorStore:
    def __init__(self, collection_name: str, url=None, api_key=None, vector_size=384):
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)
        self.vector_size = vector_size
        self._initialize_store()

    def _initialize_store(self):
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )
            print("Hybrid vector store initialized successfully")
        else:
            print("Collection already exists")

    def add_documents(
        self,
        documents: List[Any],
        dense_embeddings: np.ndarray,
        sparse_vectors: List[Any],
        batch_size: int = 200
    ):

        if not (len(documents) == len(dense_embeddings) == len(sparse_vectors)):
            raise ValueError("Documents, dense embeddings, and sparse vectors must match in length")

        total = len(documents)
        print(f"Adding {total} documents in batches of {batch_size}")

        batch_points = []

        for i, (doc, dense, sparse) in enumerate(
            zip(documents, dense_embeddings, sparse_vectors),
            start=1
        ):

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense.tolist(),
                    "sparse": SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist()
                        )
                },
                payload={
                    "content": doc["content"],
                    "chapter": doc["chapter"],
                    "page": doc["page"],
                    "content_length": len(doc["content"]),
                },
            )

            batch_points.append(point)

            if len(batch_points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points,
                )
                print(f"Uploaded {i}/{total}")
                batch_points = []

        if batch_points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch_points,
            )
            print(f"Uploaded {total}/{total}")

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding import EmbeddingManager
from vector_store import VectorStore
from retreiver import RAGRetriever
from rag import rag, llm
from reranker import DocumentReranker
from graphretriever import GraphRetriever
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MicroGPT RAG API")

# Add CORS middleware to allow the frontend to interact with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "chapterwiseReferenceManual")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_data")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "20"))
ENABLE_RERANKING = _as_bool(os.getenv("ENABLE_RERANKING"), default=True)
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K") or os.getenv("RERANKING_TOP_K") or str(RAG_TOP_K))

if not QDRANT_URL:
    raise ValueError("Missing required environment variable: QDRANT_URL")

if not QDRANT_API_KEY:
    raise ValueError("Missing required environment variable: QDRANT_API_KEY")

embedding_manager = EmbeddingManager()
vector_store = VectorStore(
    collection_name=QDRANT_COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    local_fallback_path=QDRANT_LOCAL_PATH,
)
rag_retriever = RAGRetriever(
    vector_store=vector_store,
    embedding_manager=embedding_manager,
)

try:
    graph_retriever = GraphRetriever()
    print("Graph retriever initialized successfully.")
except Exception as e:
    print(f"Failed to initialize GraphRetriever: {e}")
    graph_retriever = None

reranker = DocumentReranker() if ENABLE_RERANKING else None


class QueryRequest(BaseModel):
    query: str

class Source(BaseModel):
    chapter: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float

@app.post("/query", response_model=QueryResponse)
def query_model(request: QueryRequest):
    try:
        response = rag(
            request.query,
            rag_retriever,
            llm,
            top_k=RAG_TOP_K,
            return_context=False,
            reranker=reranker,
            rerank_top_k=RERANK_TOP_K,
            graph_retriever=graph_retriever,
        )
        return QueryResponse(
            answer=response["answer"],
            sources=response["sources"],
            confidence=response["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "MicroGPT API is running. Use /api/query to interact with the model."}

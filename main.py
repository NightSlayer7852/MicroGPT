from embedding import EmbeddingManager
from vector_store import VectorStore
from retreiver import RAGRetriever
from rag import rag, llm
from reranker import DocumentReranker
import os
from dotenv import load_dotenv

from fastembed import SparseTextEmbedding

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "chapterwiseReferenceManual")
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm25")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "20"))
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", str(RAG_TOP_K)))

if not QDRANT_URL:
    raise ValueError("Missing required environment variable: QDRANT_URL")

if not QDRANT_API_KEY:
    raise ValueError("Missing required environment variable: QDRANT_API_KEY")

embedding_manager = EmbeddingManager()
vector_store = VectorStore(
    collection_name=QDRANT_COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


rag_retriever = RAGRetriever(
    vector_store=vector_store,
    embedding_manager=embedding_manager,
    sparse_model=sparse_model
)

reranker = DocumentReranker() if ENABLE_RERANKING else None

#clear the console

os.system('cls' if os.name == 'nt' else 'clear')
while True:
    query = input("Enter your question: ")
    response = rag(
        query,
        rag_retriever,
        llm,
        top_k=RAG_TOP_K,
        return_context=True,
        reranker=reranker,
        rerank_top_k=RERANK_TOP_K,
    )
    print(response["answer"])
    print("\nSources:")
    for source in response["sources"]:
        print(f"Chapter: {source.get('chapter')}, Page: {source.get('page')}, Score: {source.get('score')}")
    print(f"\nConfidence Score: {response['confidence']:.4f}")
    print()
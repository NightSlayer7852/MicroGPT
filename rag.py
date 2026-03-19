import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing required environment variable: GROQ_API_KEY")

GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "2"))

from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=GROQ_TEMPERATURE,
    max_tokens=GROQ_MAX_TOKENS,
    timeout=None,
    max_retries=GROQ_MAX_RETRIES,
)

def rag(query, retriever, llm, top_k=10, return_context=False):

    results = retriever.retrieve(query, top_k=top_k)

    if not results:
        return {
            "answer": "No relevant context found.",
            "sources": [],
            "confidence": 0.0
        }

    context = "\n\n".join([doc["content"] for doc in results])

    sources = [
        {
            "chapter": doc.get("chapter"),
            "page": doc.get("page"),
            "score": doc.get("score"),
        }
        for doc in results
    ]

    confidence = max([doc["score"] for doc in results])

    prompt = f"""
You are a technical documentation assistant.

You MUST answer ONLY using the provided context.
Do NOT use outside knowledge.
Question: {query}
Context:
{context}
Sources:
{sources}

STRICT:
Citations should be accurate
FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

Answer:
<clear, structured answer>

Citations:
- Page <number>: <short supporting quote>

Confidence:
<High / Medium / Low>
"""

    response = llm.invoke(prompt)

    output = {
        "answer": response.content,
        "sources": sources,
        "confidence": confidence,
    }

    if return_context:
        output["context"] = context

    return output

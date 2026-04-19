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

def expand_query_with_graph(query, graph_retriever):
    related_entities = graph_retriever.get_related_entities(query)

    if not related_entities:
        return query

    expanded_query = query + " " + " ".join(related_entities)
    return expanded_query

def rag(
    query,
    retriever,
    llm,
    top_k=10,
    return_context=False,
    reranker=None,
    rerank_top_k=None,
    graph_retriever=None,
):

    # 🔹 1. Base retrieval (hybrid)
    base_results = retriever.retrieve(query, top_k=top_k)

    # 🔹 2. Graph-based expansion
    graph_results = []
    if graph_retriever is not None:
        expanded_query = query

        try:
            related_entities = graph_retriever.get_related_entities(query)
            if related_entities:
                expanded_query = query + " " + " ".join(related_entities)

                graph_results = retriever.retrieve(
                    expanded_query,
                    top_k=max(3, top_k // 2)
                )

                # 🔹 Optional: reduce weight of graph results
                for doc in graph_results:
                    doc["score"] *= 0.8

        except Exception as e:
            print(f"[Graph Retrieval Error]: {e}")

    # 🔹 3. Merge results
    all_results = base_results + graph_results

    # 🔹 4. Deduplicate (VERY IMPORTANT)
    seen = set()
    unique_results = []
    for doc in all_results:
        key = (doc["content"], doc.get("page"))
        if key not in seen:
            seen.add(key)
            unique_results.append(doc)

    # 🔹 5. Rerank (optional)
    if reranker is not None and unique_results:
        unique_results = reranker.rerank(
            query,
            unique_results,
            top_k=rerank_top_k or top_k
        )

    # 🔹 6. Handle empty
    if not unique_results:
        return {
            "answer": "No relevant context found.",
            "sources": [],
            "confidence": 0.0
        }

    # 🔹 7. Build context
    context = "\n\n".join([doc["content"] for doc in unique_results])

    # 🔹 8. Sources
    sources = [
        {
            "chapter": doc.get("chapter"),
            "page": doc.get("page"),
            "score": doc.get("score"),
        }
        for doc in unique_results
    ]

    confidence = max([doc["score"] for doc in unique_results])

    # 🔹 9. Prompt (UNCHANGED)
    prompt = f"""
You are a technical documentation assistant.

STRICT RULES:
- Use ONLY the provided context.
- Do NOT assume missing values.
- If something is not explicitly stated, say "Not specified in context".

When answering:
- Identify ALL relevant components involved in the query.
- Provide a complete, structured explanation covering those components.
- Do NOT skip necessary steps if they are mentioned in context.

If the question involves configuration, provide step-by-step answer.

Context:
{context}

Question:
{query}

FORMAT:

Answer:
<structured answer>

Citations:
- Page <number>: "<exact sentence>"

Confidence:
<High / Medium / Low>

Follow-up Question:
<generate 2 or 3 highly relevant follow-up question based on the topic and context provided>
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
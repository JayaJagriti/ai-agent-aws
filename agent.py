from typing import TypedDict
from langgraph.graph import StateGraph, END
from groq import Groq
import os
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# ---------------- ENV ----------------
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- MODELS ----------------
MODEL_CANDIDATES = [
    "llama-3.1-8b-instant",   # primary (fast + stable)
    "qwen/qwen3-32b"         # fallback (better reasoning)
]

# ---------------- STATE ----------------
class AgentState(TypedDict):
    query: str
    retriever: any
    history: list
    result: str


# ---------------- LLM CALL (SAFE) ----------------
def call_llm(messages):
    for model in MODEL_CANDIDATES:
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3   # lower hallucination
            )
            return res.choices[0].message.content
        except Exception:
            continue

    return "⚠️ LLM unavailable. Please try again."


# ---------------- WEB SEARCH ----------------
def web_search(query):
    try:
        with DDGS(timeout=5) as ddgs:
            results = list(ddgs.text(query, max_results=2))

        if results:
            return results[0]["body"]
    except:
        return "Web search failed."

    return "No relevant results found."


# ---------------- MAIN RAG NODE ----------------
def rag_node(state: AgentState):
    query = state["query"]
    retriever = state.get("retriever")

    # 👉 STEP 1: Try RAG
    if retriever:
        docs = retriever.invoke(query)

        if docs:
            context = "\n".join([d.page_content[:300] for d in docs])

            # 👉 ALWAYS try answering from context first
            response = call_llm([
                {
                    "role": "system",
                    "content": "Answer naturally using the provided context if relevant."
                },
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Question:
{query}

Instructions:
- If context contains answer → answer clearly
- If partially relevant → try your best using it
- If completely unrelated → say NOT_FOUND
"""
                }
            ])

            # 👉 ONLY fallback if clearly useless
            if "NOT_FOUND" not in response and len(response.strip()) > 20:
                return {"result": response}

    # 👉 STEP 2: fallback (web + LLM)
    web_result = web_search(query)

    final_response = call_llm([
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": f"""
Web Info:
{web_result}

Question:
{query}
"""
        }
    ])

    return {"result": final_response}
   


# ---------------- GRAPH ----------------
def create_agent():
    graph = StateGraph(AgentState)
    graph.add_node("rag", rag_node)
    graph.set_entry_point("rag")
    graph.add_edge("rag", END)
    return graph.compile()


# ---------------- RUN ----------------
def run_agent(query, retriever=None, history=None):
    agent = create_agent()

    result = agent.invoke({
        "query": query,
        "retriever": retriever,
        "history": history or []
    })

    return result["result"]
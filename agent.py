from typing import TypedDict
from langgraph.graph import StateGraph, END
from groq import Groq
import os
from duckduckgo_search import DDGS

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- STATE ----------------
class AgentState(TypedDict):
    query: str
    retriever: list   # 👈 now list of retrievers
    history: list
    result: str


# ---------------- SMALL TALK ----------------
def is_small_talk(query: str):
    query = query.lower()
    small_talk_keywords = [
        "hi", "hello", "hey", "how are you",
        "what's up", "good morning", "good evening",
        "who are you", "what can you do", "tell me a joke"
    ]
    return any(k in query for k in small_talk_keywords)


# ---------------- WEB SEARCH ----------------
def web_search(query):
    try:
        with DDGS(timeout=5) as ddgs:
            return list(ddgs.text(query, max_results=2))
    except:
        return []


# ---------------- RAG NODE ----------------
def rag_node(state: AgentState):
    query = state["query"]
    retrievers = state.get("retriever", [])

    # ---------------- SMALL TALK ----------------
    if is_small_talk(query):
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"""
You are a friendly AI assistant.

Respond casually, naturally, and briefly.
Keep it human-like and engaging.

User: {query}
"""
            }]
        )
        return {"result": res.choices[0].message.content}

    # ---------------- BOOST QUERY ----------------
    boosted_query = query + " company organization CEO COO role name PixelNerve team meeting"

    # ---------------- RETRIEVE FROM MULTIPLE DBS ----------------
    docs = []

    if retrievers:
        # ✅ base first (important knowledge)
        docs.extend(retrievers[0].invoke(boosted_query))

        # ✅ user pdf next (optional)
        if len(retrievers) > 1:
            docs.extend(retrievers[1].invoke(boosted_query))

    # ---------------- RAG ANSWER ----------------
    if docs:
        context = "\n".join([d.page_content for d in docs])

        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"""
You MUST answer ONLY from the given context.

STRICT RULES:
- If answer is in context → answer clearly
- If NOT → reply EXACTLY: NOT_FOUND
- DO NOT guess
- DO NOT use outside knowledge

Context:
{context}

Question:
{query}
"""
            }]
        )

        answer = res.choices[0].message.content.strip()

        if "NOT_FOUND" not in answer and len(answer) > 10:
            return {"result": answer}

    # ---------------- WEB FALLBACK ----------------
    web_result = web_search(query)

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"""
Answer using this web result:

{web_result}

Question:
{query}
"""
        }]
    )

    return {"result": res.choices[0].message.content}


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
        "retriever": retriever or [],
        "history": history or []
    })

    return result["result"]
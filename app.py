import os
import streamlit as st
import time

from agent import run_agent
from rag import load_base_knowledge, add_user_pdf, get_retriever
from memory import save_message, load_history, clear_history

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="AI AGENT", layout="wide")

# ---------------- STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- DB INIT (SAFE + LIGHT) ----------------
if "db_loaded" not in st.session_state:
    try:
        if os.path.exists("data/Knowledge.pdf"):
            load_base_knowledge("data/Knowledge.pdf")
            st.session_state.retriever = get_retriever()
        else:
            st.session_state.retriever = None
    except Exception as e:
        st.session_state.retriever = None
        st.error(f"DB load failed: {e}")

    st.session_state.db_loaded = True

user_id = "default_user"

# ---------------- UI ----------------
st.title("🌙 AI Assistant ✨")

# Sidebar
with st.sidebar:
    st.markdown("### 💖 Session")

    uploaded = st.file_uploader("📂 Upload PDF", type="pdf")

    if uploaded:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded.read())

        add_user_pdf("temp.pdf")
        st.session_state.retriever = get_retriever()
        st.success("PDF added!")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🧠 Clear Memory"):
        clear_history(user_id)
        st.success("Memory cleared!")

# ---------------- CHAT DISPLAY ----------------
for msg in st.session_state.messages:
    role = "🧚‍♀️" if msg["role"] == "user" else "🔮"
    st.markdown(f"**{role} {msg['content']}**")

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message(user_id, "user", user_input)

    history = load_history(user_id, limit=5)
    retriever = st.session_state.retriever

    with st.spinner("Thinking..."):
        result = run_agent(user_input, retriever, history)

    st.session_state.messages.append({"role": "assistant", "content": result})
    save_message(user_id, "assistant", result)

    st.rerun()
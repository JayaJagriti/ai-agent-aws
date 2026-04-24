from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ✅ lightweight embeddings (NO TORCH)
from langchain_community.embeddings import FakeEmbeddings

db = None

# ---------------- LOAD BASE PDF ----------------
def load_base_knowledge(pdf_path):
    global db

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    embeddings = FakeEmbeddings(size=384)

    db = FAISS.from_documents(docs, embeddings)
    return db


# ---------------- ADD USER PDF ----------------
def add_user_pdf(pdf_path):
    global db

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    docs = splitter.split_documents(documents)

    embeddings = FakeEmbeddings(size=384)

    if db is None:
        db = FAISS.from_documents(docs, embeddings)
    else:
        db.add_documents(docs)

    return db


# ---------------- GET RETRIEVER ----------------
def get_retriever():
    global db
    return db.as_retriever(search_kwargs={"k": 5}) if db else None
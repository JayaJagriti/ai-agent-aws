from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

base_db = None
user_db = None


def create_db(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.split_documents(documents)

    embeddings = FakeEmbeddings(size=384)

    return FAISS.from_documents(docs, embeddings)


# ---------------- BASE KNOWLEDGE ----------------
def load_base_knowledge(pdf_path):
    global base_db
    base_db = create_db(pdf_path)
    return base_db


# ---------------- USER PDF ----------------
def add_user_pdf(pdf_path):
    global user_db
    user_db = create_db(pdf_path)
    return user_db


# ---------------- GET RETRIEVER ----------------
def get_retriever():
    global base_db, user_db

    retrievers = []

    if base_db:
        retrievers.append(base_db.as_retriever(search_kwargs={"k": 3}))

    if user_db:
        retrievers.append(user_db.as_retriever(search_kwargs={"k": 3}))

    return retrievers
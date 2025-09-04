import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.makedirs("data/faiss_index", exist_ok=True)


FAISS_INDEX_PATH = "data/faiss_index"

def create_faiss_index(text):
    """Split text, embed chunks, and store FAISS index."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_texts(chunks, embedding_model)
    vector_db.save_local(FAISS_INDEX_PATH)
    return len(chunks)

def load_faiss_index():
    index_path = "data/faiss_index"
    if not os.path.exists(os.path.join(index_path, "index.pkl")):
        return None, None

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    return vector_db, embedding_model

def get_similar_chunks(query, vector_db, embedding_model, top_k=5):
    """Retrieve top-k most relevant chunks using vector similarity."""
    query_embedding = embedding_model.embed_query(query)
    similar_docs = vector_db.similarity_search_by_vector(query_embedding, k=top_k)
    return [doc.page_content for doc in similar_docs]

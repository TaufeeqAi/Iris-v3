from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize embeddings

embed = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_folder="./sentence_transformers_cache",
    model_kwargs={"device": "cpu"}
)


# Load your .txt files
docs = []
for fn in os.listdir("docs"):
    if fn.endswith(".txt"):
        full = open(os.path.join("docs", fn), encoding="utf-8").read()
        docs.append(Document(page_content=full, metadata={"source": fn}))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
documents = splitter.split_documents(docs)

# Build embeddings and vector store in one go:
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embed,
    persist_directory="./chroma"
)

print(f"✅ Initialized and added docs: {vectordb._collection.count()}")

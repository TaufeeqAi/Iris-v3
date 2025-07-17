# scripts/load_initial_rag_data.py

import os
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --------- Logging Setup ---------
logger = logging.getLogger(__name__)
# Basic logging setup if common.utils.setup_logging is not available
try:
    from common.utils import setup_logging
    setup_logging(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.warning("Could not import common.utils.setup_logging. Using default logging.")

# Load environment variables (for GROQ_API_KEY if needed, though not directly in this script)
load_dotenv()

# --- Configuration for HuggingFace Embeddings and ChromaDB Persistence ---
PERSIST_DIRECTORY = "./chroma" # This must match the mountPath in the K8s Job and RAG Deployment
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_CACHE_DIR = os.path.join(PERSIST_DIRECTORY, "sentence_transformers_cache")

# Directory where your source documents are located within the Docker image
# We'll copy a 'data' folder into the image that contains your docs.
DOCUMENTS_DIR = "./data"

def load_documents(doc_dir: str):
    """Loads documents from a directory, handling different file types."""
    documents = []
    for root, _, files in os.walk(doc_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".txt"):
                logger.info(f"Loading text file: {file_path}")
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(".pdf"):
                logger.info(f"Loading PDF file: {file_path}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            # Add more loaders for other file types as needed (e.g., CSVLoader, JSONLoader)
            else:
                logger.warning(f"Skipping unsupported file type: {file_path}")
    return documents

async def main():
    logger.info("Starting RAG data loading process...")

    # Ensure the persistent directories exist (they will be created on the PVC)
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)

    # Initialize HuggingFace Embeddings
    # Model will be downloaded to EMBEDDINGS_CACHE_DIR if not present
    logger.info(f"Initializing embeddings model '{EMBEDDINGS_MODEL_NAME}' (downloading if not cached)...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL_NAME,
        cache_folder=EMBEDDINGS_CACHE_DIR,
        model_kwargs={'device': 'cpu'}
    )
    logger.info("Embeddings model initialized.")

    # Initialize ChromaDB
    logger.info(f"Initializing ChromaDB at: {PERSIST_DIRECTORY}")
    vectordb = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings_model
    )
    logger.info(f"ChromaDB current document count: {vectordb._collection.count()}")

    # Load documents from the 'data' directory
    logger.info(f"Loading documents from: {DOCUMENTS_DIR}")
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        logger.warning("No documents found to load. Ensure 'data/' directory contains documents.")
        return

    logger.info(f"Loaded {len(documents)} raw documents.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunked_documents)} chunks.")

    # Add chunks to ChromaDB
    logger.info("Adding document chunks to ChromaDB...")
    # Add documents to the collection. Chroma will handle embedding them.
    vectordb.add_documents(chunked_documents)
    logger.info(f"Successfully added {len(chunked_documents)} chunks to ChromaDB.")
    logger.info(f"ChromaDB total document count after loading: {vectordb._collection.count()}")

    # Important: Persist the changes if ChromaDB is set to persist
    # vectordb.persist() # ChromaDB's add_documents usually handles persistence if configured

    logger.info("RAG data loading process completed.")

if __name__ == "__main__":
    # Use asyncio.run for async main function
    import asyncio
    asyncio.run(main())

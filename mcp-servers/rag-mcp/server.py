from fastapi import FastAPI
from fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import logging
import json

logger = logging.getLogger(__name__)
from common.utils import setup_logging
setup_logging(__name__)

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(redirect_slashes=False)
mcp = FastMCP(name="rag")

# --- Configuration for Groq LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set for RAG MCP")

# Initialize Groq LLM
llm_model = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192",
    temperature=0.0
)

# --- Configuration for HuggingFace Embeddings and ChromaDB Persistence ---

# Base directory for ChromaDB persistence.
PERSIST_DIRECTORY = "./chroma" 
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Define the cache directory for HuggingFace embeddings *within* the persistent directory
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_CACHE_DIR = os.path.join(PERSIST_DIRECTORY, "sentence_transformers_cache")

# Ensure the embeddings cache directory also exists.
os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)

# Initialize HuggingFace Embeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDINGS_MODEL_NAME,
    cache_folder=EMBEDDINGS_CACHE_DIR,
    model_kwargs={'device': 'cpu'}
)

# Initialize global variables
qa_chain = None
vectordb = None

try:
    vectordb = Chroma(
        persist_directory=PERSIST_DIRECTORY, # ChromaDB data will also persist here
        embedding_function=embeddings_model
    )
    if vectordb._collection.count() == 0:
        print("ChromaDB is empty. Please run a separate script to load your documents.")

    retriever = vectordb.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use 4-5 sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create the document chain
    question_answer_chain = create_stuff_documents_chain(llm_model, prompt)

    # Create the retrieval chain
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)
    print(f"RAG MCP: ChromaDB initialized successfully with {vectordb._collection.count()} documents using {EMBEDDINGS_MODEL_NAME} embeddings.")

except Exception as e:
    print(f"RAG MCP Error initializing ChromaDB or LangChain: {e}")

@mcp.tool()
async def query_docs(query: str) -> dict:
    """
    Queries a specialized, custom document knowledge base to find specific answers or detailed information.
    Use this tool for questions that require deep retrieval from a curated set of documents,
    such as internal company policies, specific research papers, or detailed product specifications.
    It is designed for factual question-answering based on the provided context.

    :param query: The natural language question or query to search within the documents (e.g., "What is our company's vacation policy?", "Explain the principles of quantum entanglement as per the provided texts").
    :returns: A dictionary containing the generated answer and relevant source document snippets.
    """
    try:
        if qa_chain is None:
            return {"answer": "RAG system not initialized. Check server logs for errors.", "source_documents": []}

        result = await qa_chain.ainvoke({"input": query})

        source_docs = []
        if "context" in result and result["context"]:
            for doc in result["context"]:
                source_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })

        return {"answer": result.get("answer", "No answer found."), "source_documents": source_docs}
    except Exception as e:
        print(f"RAG MCP Error querying documents for '{query}': {e}")
        return {"answer": f"Error querying documents: {e}", "source_documents": []}


# --- FastMCP to FastAPI Integration ---
http_mcp = mcp.http_app(transport="streamable-http")
app = FastAPI(lifespan=http_mcp.router.lifespan_context)
app.mount("/", http_mcp)
logger.info("RAG MCP server initialized and tools registered.")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG MCP FastAPI server!"}

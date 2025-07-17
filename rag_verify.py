from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embed = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_folder="./sentence_transformers_cache",
    model_kwargs={"device": "cpu"}
)

vectordb = Chroma(
    persist_directory="./chroma",
    embedding_function=embed
)
print("Collection count:", vectordb._collection.count())

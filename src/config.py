

import os

# identify the absolute path of the local-rag-api project root

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# root-level data paths

DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# model configurations

EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_MODEL = "llama3.2"  # defaults ro 3b model on Ollama

# dynamic model choices mapping

SUPPORTED_MODELS = {
    "Llama 3.2 (3B) - Standard RAG": "llama3.2",
    "Qwen 2.5 (3B) - Math & Code": "qwen2.5:3b"
}

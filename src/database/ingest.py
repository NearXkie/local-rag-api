import sys
import os

# Automatically add the project root directory (local-rag-api) to Python's search path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Import central settings from config
from src.config import DOCS_DIR, CHROMADB_DIR, EMBEDDING_MODEL

def run_ingestion():
    print("🚀 Starting the Local RAG Document Ingestion Pipeline...")

    # Look for PDF files in /docs
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in '{DOCS_DIR}'!")
        return

    print(f"📚 Found {len(pdf_files)} PDF(s) to process.")
    
    all_documents = []
    for pdf_path in pdf_files:
        print(f"📄 Loading: {os.path.basename(pdf_path)}...")
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_documents.extend(docs)
            print(f"✅ Loaded {len(docs)} pages.")
        except Exception as e:
            print(f"❌ Error loading {os.path.basename(pdf_path)}: {e}")
            return

    # Text Splitting
    print("✂️ Splitting documents into semantic chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(all_documents)
    total_chunks = len(chunks)
    print(f"✅ Created {total_chunks} distinct chunks.")

    # Convert text to embeddings using Nomic
    print(f"🧠 Connecting to Ollama Embedding Model: {EMBEDDING_MODEL}...")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return

    # Persist the vector database locally in safe, low-RAM batches of 100
    print(f"💾 Vectorizing and saving database to '{CHROMADB_DIR}' in safe batches...")
    try:
        # Initialize ChromaDB client empty
        vector_store = Chroma(
            persist_directory=CHROMADB_DIR,
            embedding_function=embeddings
        )
        
        # Batch execution loop
        batch_size = 100
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            current_batch_num = (i // batch_size) + 1
            total_batches = ((total_chunks - 1) // batch_size) + 1
            
            print(f"⏳ Processing batch {current_batch_num}/{total_batches} (chunks {i} to {min(i + batch_size, total_chunks)})...")
            vector_store.add_documents(documents=batch_chunks)
            
        print("🎉 Successfully built and saved your local vector database!")
    except Exception as e:
        print(f"❌ Error persisting to ChromaDB: {e}")

if __name__ == "__main__":
    run_ingestion()

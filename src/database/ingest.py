

import os
import glob
from langchain_community.document_loaders import PyPDFLoader # pyright: ignore[reportMissingImports]
from langchain_text_splitters import RecursiveCharacterTextSplitter # pyright: ignore[reportMissingImports]
from langchain_ollama import OllamaEmbeddings # pyright: ignore[reportMissingImports]
from langchain_community.vectorstores import Chroma # pyright: ignore[reportMissingImports]

# Import central settings from config
from src.config import DOCS_DIR, CHROMA_DIR, EMBEDDING_MODEL

def get_indexed_files(vector_store) -> set:
    """Helper function to extract unique PDF filenames already recorded in ChromaDB metadata."""
    try:
        # Retrieve all items from the collection
        collection_data = vector_store.get()
        indexed_filenames = set()
        
        if collection_data and "metadatas" in collection_data:
            for metadata in collection_data["metadatas"]:
                if metadata and "source" in metadata:
                    # Extracts 'the_way_of_kings.pdf' from the absolute path
                    indexed_filenames.add(os.path.basename(metadata["source"]))
        return indexed_filenames
    except Exception as e:
        print(f"⚠️ Could not check indexed documents in vector database: {e}")
        return set()

def run_ingestion(force_rebuild=False):
    print("🔍 Scanning local document folder and database status...")

    # 1. Find all PDFs in the docs/ directory
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    if not pdf_files:
        print(f"⚠️ No PDFs found in '{DOCS_DIR}'. Database is idle.")
        return

    # 2. Check connection to Ollama Embeddings
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("💡 Ensure that the Ollama app is running on your system!")
        return

    # 3. Determine which files are new
    db_exists = os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0
    vector_store = None
    files_to_ingest = []

    if db_exists and not force_rebuild:
        # Load the existing database
        vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        indexed_files = get_indexed_files(vector_store)
        
        # Compare physical files against what is in ChromaDB
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            if filename not in indexed_files:
                files_to_ingest.append(pdf_path)
                print(f"🆕 Found new document: '{filename}'")
            else:
                print(f"✅ Already indexed: '{filename}'")
    else:
        # Database doesn't exist or we forced a clean install
        files_to_ingest = pdf_files
        print("🆕 Database not found. Building completely from scratch...")

    # 4. If everything is indexed, exit early!
    if not files_to_ingest:
        print("✨ Database is 100% up-to-date. No new PDFs to ingest!")
        return

    # 5. Load and parse only the new documents
    all_documents = []
    for pdf_path in files_to_ingest:
        print(f"📄 Loading: {os.path.basename(pdf_path)}...")
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_documents.extend(docs)
            print(f"✅ Loaded {len(docs)} pages.")
        except Exception as e:
            print(f"❌ Error loading {os.path.basename(pdf_path)}: {e}")
            return

    # 6. Split text into semantic chunks
    print("✂️ Splitting documents into semantic chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(all_documents)
    total_chunks = len(chunks)
    print(f"✅ Created {total_chunks} distinct chunks.")

    # 7. Add document chunks to Chroma in safe, CPU-friendly batches of 100
    print(f"💾 Saving new chunks to database at '{CHROMA_DIR}'...")
    try:
        if vector_store is None:
            vector_store = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )
        
        batch_size = 100
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            current_batch_num = (i // batch_size) + 1
            total_batches = ((total_chunks - 1) // batch_size) + 1
            
            print(f"⏳ Processing batch {current_batch_num}/{total_batches} (chunks {i} to {min(i + batch_size, total_chunks)})...")
            vector_store.add_documents(documents=batch_chunks)
            
        print("🎉 Successful database update complete!")
    except Exception as e:
        print(f"❌ Error persisting new documents to ChromaDB: {e}")

if __name__ == "__main__":
    run_ingestion()
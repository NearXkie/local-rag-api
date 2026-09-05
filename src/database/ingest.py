import os
import glob
from langchain_community.document_loaders import PyPDFLoader # pyright: ignore[reportMissingImports]
from langchain_text_splitters import RecursiveCharacterTextSplitter # pyright: ignore[reportMissingImports]
from langchain_ollama import OllamaEmbeddings # pyright: ignore[reportMissingImports]
from langchain_community.vectorstores import Chroma # pyright: ignore[reportMissingImports]

# Import central settings from config
from src.config import DOCS_DIR, CHROMA_DIR, EMBEDDING_MODEL

def get_db_data(vector_store) -> tuple[set, list[str], list[dict]]:
    """Helper to extract indexed filenames, document IDs, and metadatas from ChromaDB."""
    try:
        collection_data = vector_store.get()
        indexed_filenames = set()
        ids = collection_data.get("ids", [])
        metadatas = collection_data.get("metadatas", [])
        
        if metadatas:
            for metadata in metadatas:
                if metadata and "source" in metadata:
                    indexed_filenames.add(os.path.basename(metadata["source"]))
        return indexed_filenames, ids, metadatas
    except Exception as e:
        print(f"⚠️ Could not read active vector database: {e}")
        return set(), [], []

def run_ingestion(force_rebuild=False):
    print("🔍 Scanning local document folder and database status...")

    # 1. Locate current physical PDFs in /docs
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    current_filenames = {os.path.basename(path) for path in pdf_files}

    # 2. Check connection to Ollama Embeddings
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("💡 Ensure that the Ollama app is running on your system!")
        return

    # 3. Determine database status and handle synchronization
    db_exists = os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0
    vector_store = None
    files_to_ingest = []

    if db_exists and not force_rebuild:
        # Load active database
        vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        indexed_files, db_ids, db_metadatas = get_db_data(vector_store)
        
        # --- SYNCHRONIZATION: DELETE REMOVED PDFS ---
        ids_to_purge = []
        purged_files = set()
        
        for doc_id, meta in zip(db_ids, db_metadatas):
            if meta and "source" in meta:
                source_file = os.path.basename(meta["source"])
                if source_file not in current_filenames:
                    ids_to_purge.append(doc_id)
                    purged_files.add(source_file)
                    
        if ids_to_purge:
            print(f"🗑️ Detected deleted PDFs: {list(purged_files)}")
            print(f"🧹 Surgically purging {len(ids_to_purge)} orphaned chunks from ChromaDB...")
            try:
                vector_store.delete(ids=ids_to_purge)
                print("✅ Orphaned database entries successfully deleted!")
                # Re-fetch remaining index status
                indexed_files, _, _ = get_db_data(vector_store)
            except Exception as e:
                print(f"❌ Failed to delete chunks from ChromaDB: {e}")
        # ---------------------------------------------

        # Compare remaining indexed items to see if there is any new work to do
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            if filename not in indexed_files:
                files_to_ingest.append(pdf_path)
                print(f"🆕 Found new document to index: '{filename}'")
            else:
                print(f"✅ Already indexed: '{filename}'")
    else:
        # DB does not exist, everything in /docs needs indexing
        files_to_ingest = pdf_files
        print("🆕 Database not found. Building completely from scratch...")

    # 4. If there is nothing to ingest and we cleaned up, we can exit early!
    if not files_to_ingest:
        print("✨ Database is 100% synchronized with your '/docs' folder!")
        return

    # 5. Parse and slice the new books
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

    print("✂️ Splitting documents into semantic chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(all_documents)
    total_chunks = len(chunks)
    print(f"✅ Created {total_chunks} distinct chunks.")

    # 6. Append new chunks to ChromaDB in safe batches of 100
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

import os
from fastapi import FastAPI, HTTPException # pyright: ignore[reportMissingImports]
from pydantic import BaseModel # pyright: ignore[reportMissingImports]
from langchain_ollama import OllamaEmbeddings, ChatOllama # pyright: ignore[reportMissingImports]
from langchain_community.vectorstores import Chroma # pyright: ignore[reportMissingImports]

# Import central settings from config
from src.config import CHROMA_DIR, EMBEDDING_MODEL, DEFAULT_MODEL, SUPPORTED_MODELS

# Import the ingestion function we just optimized
from src.database.ingest import run_ingestion

# Initialize FastAPI app
app = FastAPI(
    title="Local RAG API", 
    description="A lightweight, offline document assistant running completely locally."
)

# Initialize global connections
embeddings = None
vector_store = None

@app.on_event("startup")
def startup_event():
    global embeddings, vector_store
    print("⚙️ Initiating startup pipeline...")
    try:
        # Run smart ingestion automatically on boot
        run_ingestion()
        
        print("🔌 Connecting to your local ChromaDB vector store...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # Load the database connection directly
        vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        print("✅ Vector store connection established successfully!")
        
    except Exception as e:
        print(f"❌ Error during server startup: {e}")


def correct_query_typos(question: str, model_name: str) -> str:
    """Uses a fast local LLM pass to correct spelling mistakes and preserve fantasy names."""
    try:
        # We use temperature 0.0 to keep the correction strictly deterministic and fast
        llm = ChatOllama(model=model_name, temperature=0.0)
        
        correction_prompt = (
            "You are an offline search query optimizer. Your sole task is to correct any spelling mistakes or typos "
            "in the input search query. "
            "Pay special attention to character names, locations, and terminology.\n\n"
            "Rules:\n"
            "1. If there are no spelling mistakes, return the input query exactly as-is.\n"
            "2. Do NOT answer the question. Do NOT add explanations or pleasantries.\n"
            "3. Return ONLY the cleaned search query.\n\n"
            f"Input Query: {question}"
        )
        
        response = llm.invoke(correction_prompt)
        cleaned_query = response.content.strip()
        
        # Strips occasional wrapping quotation marks if the LLM adds them
        if cleaned_query.startswith('"') and cleaned_query.endswith('"'):
            cleaned_query = cleaned_query[1:-1]
            
        return cleaned_query
    except Exception as e:
        print(f"⚠️ Query auto-correction bypassed: {e}")
        return question


# Request & Response structures
class QueryRequest(BaseModel):
    question: str
    model_name: str = DEFAULT_MODEL
    k: int = 5  

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if not vector_store:
        raise HTTPException(status_code=500, detail="Database is not connected.")
    
    allowed_models = list(SUPPORTED_MODELS.values())
    if request.model_name not in allowed_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{request.model_name}' is not supported. Use one of: {allowed_models}"
        )
    
    try:
        # 1. Silently correct typos in the user's question before search
        print(f"🧹 Checking query for typos: '{request.question}'...")
        cleaned_question = correct_query_typos(request.question, request.model_name)
        
        if cleaned_question != request.question:
            print(f"✨ Auto-corrected query to: '{cleaned_question}'")
        else:
            print("✅ No typos detected.")

        # 2. Retrieve the top 'k' chunks using the CLEANED question
        print(f"🔍 Searching database for top {request.k} chunks: '{cleaned_question}'...")
        relevant_docs = vector_store.similarity_search(cleaned_question, k=request.k)
        
        if not relevant_docs:
            return QueryResponse(
                answer="I couldn't find any relevant sections in the uploaded documents.",
                sources=[]
            )
        
        # 3. Extract context and metadata
        context_blocks = []
        sources_metadata = []
        for doc in relevant_docs:
            context_blocks.append(doc.page_content)
            sources_metadata.append({
                "file": os.path.basename(doc.metadata.get("source", "Unknown")),
                "page": doc.metadata.get("page", 0) + 1
            })
            
        context = "\n\n---\n\n".join(context_blocks)
        
        # 4. Assemble system prompt
        system_prompt = (
            "You are an expert, precise offline assistant. Your task is to answer the user's question relying ONLY on the provided documents.\n\n"
    "CRITICAL RULES:\n"
    "1. You must answer using ONLY the information present in the context below. Do NOT use prior knowledge.\n"
    "2. You must answer naturally, not like a robot who is trying to prove its intelligence.\n"
    "3. If the provided context does not contain the answer, you must say exactly: 'I cannot answer this based on the provided context.' and stop.\n"
    "4. Every factual claim in your response must end with an inline citation, formatted as [Source: filename, Page: X].\n"
    "5. A claim with no supporting context must not appear in the answer.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {request.question}\n"
    "Answer (with citations):"
        )
        
       # 5. Generate answer
        print(f"🧠 Generating response using local model '{request.model_name}'...")
        llm = ChatOllama(model=request.model_name, temperature=0.2)
        response = llm.invoke(system_prompt)
        
        return QueryResponse(
            answer=response.content,
            sources=sources_metadata
        )
        
    except Exception as e:
        print(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_connected": vector_store is not None,
        "embedding_model": EMBEDDING_MODEL
    }
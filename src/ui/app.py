

import streamlit as st # pyright: ignore[reportMissingImports]
import requests

# Clean import from config (made possible by your PYTHONPATH setup!)
from src.config import SUPPORTED_MODELS

# FastAPI Endpoint Config
API_URL = "http://127.0.0.1:8000"

# Configure the web browser page
st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Local RAG Assistant")
st.caption("Running 100% offline, private, and locally on your machine.")

# Sidebar Configuration Layout
st.sidebar.header("Configuration")

# 1. Model Switching Selector
model_display_name = st.sidebar.selectbox(
    "Select AI Brain:",
    options=list(SUPPORTED_MODELS.keys())
)
selected_model_id = SUPPORTED_MODELS[model_display_name]

# 2. Live Connection status check with the FastAPI Backend
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=2)
    if health_resp.status_code == 200 and health_resp.json().get("status") == "healthy":
        st.sidebar.success("🟢 Backend Connected")
    else:
        st.sidebar.error("🔴 Backend Running but Unhealthy")
except requests.exceptions.ConnectionError:
    st.sidebar.error("🔴 RAG Backend is Offline. Start FastAPI!")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "### About This App\n"
    "This system performs **Semantic Vector Search** on your uploaded PDFs "
    "using **ChromaDB** and uses local LLMs via **Ollama** to synthesize "
    "strict context-based responses."
)

# Initialize Session State Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Conversational History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 Citations & Sources"):
                for src in message["sources"]:
                    st.write(f"📄 **{src['file']}** (Page {src['page']})")

# Handle Incoming User Prompts
if prompt := st.chat_input("Ask a question about your documents..."):
    # Render user query
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call FastAPI server to fetch the answer
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Retrieving document context and generating answer..."):
            try:
                payload = {
                    "question": prompt,
                    "model_name": selected_model_id
                }
                response = requests.post(f"{API_URL}/query", json=payload, timeout=120)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    # Update message in the UI
                    response_placeholder.markdown(answer)
                    
                    # Show expandable citation list if sources were retrieved
                    if sources:
                        # Ensure we deduplicate sources for cleaner rendering
                        unique_sources = {f"{s['file']}_p{s['page']}": s for s in sources}.values()
                        with st.expander("🔍 Citations & Sources"):
                            for src in unique_sources:
                                st.write(f"📄 **{src['file']}** (Page {src['page']})")
                    
                    # Save to memory
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": list(unique_sources)
                    })
                else:
                    error_detail = response.json().get("detail", "Internal server error.")
                    response_placeholder.error(f"Error from API Backend: {error_detail}")
            except Exception as e:
                response_placeholder.error(f"Failed to reach FastAPI backend. Is your server running? \n\n*Details: {e}*")
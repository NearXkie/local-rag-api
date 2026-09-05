

import streamlit as st # pyright: ignore[reportMissingImports]
import requests

# Import settings from config
from src.config import SUPPORTED_MODELS

API_URL = "http://127.0.0.1:8000"

# Configure the page layout
st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="📚",
    layout="centered"
)

# --- CUSTOM CSS: Professional Chat App Bubbles ---
# This styles the chat interface: User bubble on the right (Blue), AI on the left (Gray/Dark)
st.markdown("""
    <style>
    /* 1. Push user messages to the right side of the container */
    div[data-testid="stChatMessage"]:has(div[data-testid="chat-avatar-user"]) {
        flex-direction: row-reverse !important;
        text-align: right !important;
    }
    
    /* 2. Style the User Message Bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="chat-avatar-user"]) div[data-testid="stChatMessageContent"] {
        background-color: #007aff !important; /* Modern iOS/MacOS Blue */
        color: white !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 12px 16px !important;
        display: inline-block !important;
        text-align: left !important; /* Keep internal text aligned left for readability */
        max-width: 80% !important;
        box-shadow: 0px 1px 2px rgba(0,0,0,0.1) !important;
    }
    
    /* 3. Style the Assistant Message Bubble (Light Mode) */
    div[data-testid="stChatMessage"]:has(div[data-testid="chat-avatar-assistant"]) div[data-testid="stChatMessageContent"] {
        background-color: #f0f2f6 !important;
        color: #1f2937 !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 12px 16px !important;
        max-width: 80% !important;
    }
    
    /* 4. Style the Assistant Message Bubble (Dark Mode override) */
    @media (prefers-color-scheme: dark) {
        div[data-testid="stChatMessage"]:has(div[data-testid="chat-avatar-assistant"]) div[data-testid="stChatMessageContent"] {
            background-color: #262730 !important;
            color: #f5f5f5 !important;
            box-shadow: 0px 1px 2px rgba(0,0,0,0.2) !important;
        }
    }
    
    /* Hide the default avatar icons to keep the bubble design ultra-clean */
    div[data-testid="chat-avatar-user"], div[data-testid="chat-avatar-assistant"] {
        display: none !important;
    }
    
    /* Adjust spacing between bubbles */
    div[data-testid="stChatMessage"] {
        padding: 0.5rem 0rem !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- HEADER ---
st.title("📚 Local RAG Assistant")

# --- SMART INLINE CONTROLLER (Right above the chat window) ---
# This replaces the sidebar with a compact configuration panel
with st.container(border=True):
    cols = st.columns([3, 3, 2])
    
    with cols[0]:
        model_display_name = st.selectbox(
            "AI Brain:",
            options=list(SUPPORTED_MODELS.keys()),
            help="Select which local LLM handles your queries."
        )
        selected_model_id = SUPPORTED_MODELS[model_display_name]
        
    with cols[1]:
        k_value = st.slider(
            "Context Passages (k):",
            min_value=1,
            max_value=100,
            value=20,
            step=2,
            help="Higher values give the AI more context but slow down CPU generation."
        )
        
    with cols[2]:
        st.write("Connection:")
        # Check API status
        try:
            health_resp = requests.get(f"{API_URL}/health", timeout=1.5)
            if health_resp.status_code == 200 and health_resp.json().get("status") == "healthy":
                st.success("🟢 Online")
            else:
                st.warning("🟡 Unhealthy")
        except requests.exceptions.ConnectionError:
            st.error("🔴 Offline")

st.write("---")

# --- CHAT WINDOW SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render Conversational History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 Citations & Sources"):
                for src in message["sources"]:
                    st.write(f"📄 **{src['file']}** (Page {src['page']})")

# --- USER INPUT & AI GENERATION ---
if prompt := st.chat_input("Ask a question about your documents..."):
    # Render user query in real-time
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call FastAPI backend
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Analyzing document context..."):
            try:
                payload = {
                    "question": prompt,
                    "model_name": selected_model_id,
                    "k": k_value
                }
                response = requests.post(f"{API_URL}/query", json=payload, timeout=120)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    response_placeholder.markdown(answer)
                    
                    # If sources exist, deduplicate and display
                    if sources:
                        unique_sources = {f"{s['file']}_p{s['page']}": s for s in sources}.values()
                        with st.expander("🔍 Citations & Sources"):
                            for src in unique_sources:
                                st.write(f"📄 **{src['file']}** (Page {src['page']})")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": list(unique_sources) if sources else []
                    })
                else:
                    error_detail = response.json().get("detail", "Internal server error.")
                    response_placeholder.error(f"Error: {error_detail}")
            except Exception as e:
              response_placeholder.error(f"Failed to connect to the backend. Is your server running? \n\n*Details: {e}*")
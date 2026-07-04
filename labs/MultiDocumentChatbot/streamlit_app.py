import streamlit as st
import time
import shutil
import os
import uuid
import traceback

# ⚠️ MUST be the first Streamlit command
st.set_page_config(
    page_title="Multi-Document AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.settings import ChatSettings
from app.chains import build_chatbot
from app.file_manager import save_uploaded_files
from app.ingestion import ingest_documents
from app.retriever import get_retriever
from app.vectordb import load_vector_db

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "settings" not in st.session_state:
    st.session_state.settings = ChatSettings(
        model="mistral",
        temperature=0.0,
        top_k=4,
        search_type="similarity",
        max_tokens=2048
    )

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploads")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# Cache the chatbot loading
@st.cache_resource
def load_chatbot(_settings):
    """Load and cache the chatbot."""
    return build_chatbot(settings=_settings)

# Check if knowledge base exists before loading chatbot
kb_exists = os.path.exists(CHROMA_DB_PATH) and st.session_state.doc_count > 0
if kb_exists:
    chatbot = load_chatbot(st.session_state.settings)
else:
    chatbot = None

# ========== UI LAYOUT ==========

st.title("📚 Multi-Document AI Assistant")
st.markdown(
    "Upload multiple documents and chat with them using LangChain + Ollama + ChromaDB."
)

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.markdown("---")
    
    # Project Info with better styling
    st.subheader("🔧 Tech Stack")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("Mistral")
    with col2:
        st.success("ChromaDB")
    with col3:
        st.success("nomic-embed")

    st.markdown("---")

    st.sidebar.subheader("⚙ Chat Settings")

    model = st.sidebar.selectbox(
        "LLM Model",
        [
            "mistral",
            "llama3",
            "gemma3"
        ]
    )

    temperature = st.sidebar.slider(
        "Temperature",
        0.0,
        1.0,
        0.0,
        0.1
    )
    
    top_k = st.sidebar.slider(
        "Retriever Top-K",
        1,
        10,
        4
    )

    search_type = st.sidebar.selectbox(
        "Retriever",
        [
            "similarity",
            "mmr"
        ]
    )

    max_tokens = st.sidebar.slider(
        "Max Tokens",
        256,
        4096,
        2048,
        128
    )

    if st.sidebar.button("🔄 Apply Settings"):
        st.session_state.settings = ChatSettings(
            model=model,
            temperature=temperature,
            top_k=top_k,
            search_type=search_type,
            max_tokens=max_tokens
        )
        st.cache_resource.clear()
        st.rerun()
    
    # File Upload
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, TXT, CSV)",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
    )

    # Show upload count
    if uploaded_files:
        st.info(f"📎 {len(uploaded_files)} file(s) selected")

    # Process Button
    process_button = st.button(
        "📄 Process Documents",
        use_container_width=True,
        type="primary",
    )

    # Handle document processing
    if process_button:
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one file.")
        else:
            try:
                # Progress bar for processing steps
                progress = st.progress(0, text="Starting...")
                
                progress.progress(20, text="📁 Saving files...")
                saved, skipped = save_uploaded_files(uploaded_files)
                time.sleep(0.3)
                if skipped:
                    st.warning(
                        f"Skipped {len(skipped)} duplicate file(s)."
                    )
                
                progress.progress(50, text="📄 Loading documents...")
                time.sleep(0.3)
                
                progress.progress(70, text="✂️ Splitting text...")
                time.sleep(0.3)
                
                progress.progress(85, text="🧬 Generating embeddings...")
                
                # Actual ingestion
                doc_count, chunk_count = ingest_documents(UPLOAD_FOLDER)
                
                progress.progress(100, text="✅ Complete!")
                
                # Store in session state
                st.session_state.doc_count = doc_count
                st.session_state.chunk_count = chunk_count
                
                # Clear cache to reload chatbot
                st.cache_resource.clear()
                
                st.success(f"✅ {doc_count} document(s) processed ({chunk_count} chunks)")
                st.balloons()
                
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

    # Display Knowledge Base Statistics
    if st.session_state.doc_count > 0:
        st.markdown("---")
        st.subheader("📊 Knowledge Base Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", st.session_state.doc_count)
        with col2:
            st.metric("Chunks", st.session_state.chunk_count)

    st.markdown("---")
    
    # Chat Controls
    st.subheader("💬 Chat Controls")
    clear_chat = st.button(
        "🧹 Clear Chat",
        use_container_width=True,
    )

    reset_db = st.button(
        "🗑️ Reset Knowledge Base",
        use_container_width=True,
        type="secondary",
    )

    # Handle clear chat
    if clear_chat:
        st.session_state.messages = []
        st.rerun()

    # Handle reset database
    if reset_db:
        with st.spinner("🗑️ Deleting knowledge base..."):
            # Delete ChromaDB
            if os.path.exists(CHROMA_DB_PATH):
                shutil.rmtree(CHROMA_DB_PATH)
            
            # Delete uploaded files
            if os.path.exists(UPLOAD_FOLDER):
                shutil.rmtree(UPLOAD_FOLDER)
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Reset session state
            st.session_state.doc_count = 0
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            
            # Clear cache
            st.cache_resource.clear()
            
            st.success("✅ Knowledge base and uploaded files deleted successfully!")
            time.sleep(1)
            st.rerun()

# ========== MAIN CHAT AREA ==========
st.markdown("---")
st.subheader("💬 Chat")

# Empty state - show welcome message
if not st.session_state.messages:
    st.info(
        """
        👋 **Welcome to Multi-Document AI Assistant!**
        
        1. 📤 **Upload documents** using the sidebar
        2. 🔄 Click **Process Documents** to index them
        3. 💬 **Start asking questions** about your documents
        
        ---
        **Tip:** You can upload PDFs, text files, and CSV files. The AI will answer based only on the content of your documents.
        """
    )

# Display chat messages with avatars
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Check if knowledge base exists before allowing chat
if not kb_exists:
    st.warning("⚠️ Please upload and process documents first to start chatting.")
    st.stop()

# ========== CHAT INPUT ==========
question = st.chat_input("Ask a question about your documents...")

if question:
    # Prevent empty questions
    if not question.strip():
        st.warning("⚠️ Please enter a valid question.")
    else:
        # Add user message to chat
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get retrieved documents for sources
                    try:
                        vector_db = load_vector_db()
                        retriever = get_retriever(vector_db)
                        docs = retriever.invoke(question)
                    except Exception as e:
                        st.warning(f"Could not retrieve sources: {str(e)}")
                        docs = []
                    
                    # Stream response
                    placeholder = st.empty()
                    full_response = ""

                    for chunk in chatbot.stream(
                        {"question": question},
                        config={
                            "configurable": {
                                "session_id": st.session_state.session_id
                            }
                        },
                    ):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")

                    placeholder.markdown(full_response)
                    
                    # Display sources if available
                    if docs:
                        with st.expander("📚 Sources", expanded=False):
                            for i, doc in enumerate(docs, 1):
                                source = doc.metadata.get("source", "Unknown")
                                page = doc.metadata.get("page", "N/A")
                                content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                
                                st.markdown(f"**📄 {source}**")
                                st.caption(f"Page: {page}")
                                st.markdown(f"> {content}")
                                if i < len(docs):
                                    st.divider()
                    
                    # Store assistant response in session state
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
                    
                    # Store error message
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"🔒 Session: {st.session_state.session_id[:8]}...")
with col2:
    st.caption(f"📊 Docs: {st.session_state.doc_count} | Chunks: {st.session_state.chunk_count}")
with col3:
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")
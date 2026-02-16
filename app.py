import streamlit as st
import chromadb
from dotenv import load_dotenv

load_dotenv()  # .env locally
# On Streamlit Cloud, set GROQ_API_KEY in app secrets (injected as env var)

from langchain_groq import ChatGroq

# Import the functions from your rag_logic.py file
from rag_logic import initialize_embeddings, answer_research_question

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Python Expert Assistant",
    page_icon="🐍",
    layout="wide"
)

st.title("🐍 Python Expert Assistant")
st.markdown("Ask any question about Python, and get answers sourced from official documentation.")

# --- LOAD RESOURCES (CACHED) ---
# Load DB + LLM at startup (fast). Load embedding model only on first query to avoid
# PyTorch/torch.classes noise and slow startup.
@st.cache_resource
def load_db_and_llm():
    db_client = chromadb.PersistentClient(path="./research_db")
    db_collection = db_client.get_or_create_collection(
        name="python_publications",
        metadata={"hnsw:space": "cosine"}
    )
    llm = ChatGroq(model="llama-3.1-8b-instant")
    return db_collection, llm


@st.cache_resource
def get_embeddings():
    """Loaded on first query to avoid torch/sentence-transformers at startup."""
    return initialize_embeddings("sentence-transformers/all-MiniLM-L6-v2")


collection, llm = load_db_and_llm()

# --- CHATBOT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("What is the key difference between a list and a tuple?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                embeddings = get_embeddings()  # Loads on first use (may show torch msg in console once)
                answer, sources = answer_research_question(prompt, collection, embeddings, llm)
                answer_text = answer if isinstance(answer, str) else str(getattr(answer, "content", answer))
                st.markdown(answer_text)

                # Display sources in an expander
                with st.expander("View Sources"):
                    if not sources:
                        st.caption("No sources retrieved. Run `python rag_logic.py` to build the database.")
                    for i, source in enumerate(sources):
                        sim = source.get("similarity", 0)
                        content = source.get("content", "")
                        st.info(f"Source {i+1} (Similarity: {sim:.4f})")
                        st.text(content)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
            except Exception as e:
                st.error("Something went wrong")
                st.exception(e)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
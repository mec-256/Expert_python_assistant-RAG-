import streamlit as st
from main import ( # Assuming your original file is main.py
    initialize_embeddings, 
    answer_research_question
)
import chromadb

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Python Expert Assistant",
    page_icon="🐍",
    layout="wide"
)

st.title("🐍 Python Expert Assistant")
st.markdown("Ask any question about Python, and get answers sourced from official documentation.")

# --- LOAD RESOURCES (CACHED) ---
# Use caching to avoid reloading models and data on every interaction
@st.cache_resource
def load_resources():
    print("Loading resources...")
    embeddings = initialize_embeddings("sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./research_db")
    collection = client.get_or_create_collection(
        name="python_publications",
        metadata={"hnsw:space": "cosine"}
    )
    return embeddings, collection

embeddings, collection = load_resources()

# --- CHATBOT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is the key difference between a list and a tuple?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # You'll need to re-initialize the LLM here or pass it in
            from langchain_groq import ChatGroq
            llm = ChatGroq(model="mixtral-8x7b-32768") 

            answer, sources = answer_research_question(prompt, collection, embeddings, llm)

            st.markdown(answer)

            with st.expander("View Sources"):
                for source in sources:
                    st.info(f"Similarity: {source['similarity']:.4f}\n\nContent: {source['content'][:250]}...")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
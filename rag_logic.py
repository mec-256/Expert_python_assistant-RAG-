#importing all the necessary libraries
import os
import chromadb
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from typing import List, Any


#giving values to all the necessary variables
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "./research_db"
COLLECTION_NAME = "python_publications"
LLM_MODEL_NAME = "llama-3.1-8b-instant" 
PDF_INPUT_PATH = "fluent_python.pdf"
TXT_OUTPUT_PATH = "fluent_python.txt"

# --- HELPER FUNCTIONS ---
#converts the fluent python txtbook in pdf format to text format    
def convert_pdf_to_text(pdf_path: str, txt_path: str):
    """Converts a PDF file to a plain text file."""
    try:
        doc = fitz.open(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            for page in doc:
                f.write(page.get_text())
        print(f"Successfully converted PDF to {txt_path}")
    except Exception as e:
        print(f"Error converting PDF. Check if '{pdf_path}' exists: {e}")

#load all documentation from offical python websote using a web scraper from langchain
def load_all_documents() -> List[Any]:
    """Loads Python documentation from the web and local file."""
    docs_urls = [
        "https://docs.python.org/3/tutorial/index.html",
        "https://docs.python.org/3/library/index.html",
        "https://docs.python.org/3/howto/index.html",
        "https://docs.python.org/3/faq/index.html",
    ]
    print("Loading web documentation...")
    loader = WebBaseLoader(docs_urls)
    docs = loader.load()

    if os.path.exists(TXT_OUTPUT_PATH):
        print(f"Loading local file: {TXT_OUTPUT_PATH}")
        text_loader = TextLoader(TXT_OUTPUT_PATH)
        docs.extend(text_loader.load())

    return docs
#chunks the text documents into smaller parts ot make it easier for the model to process
#experimented with diff chunk sizes and overlaps to get the best results
def chunk_documents(documents: List[Any]) -> List[Any]:
    """Splits a list of LangChain Documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total documents chunked: {len(chunks)}")
    return chunks

#initializes the embedding model from 
#embedding is nothing but converting this text into vectors of numbers so that the model can understand it
def initialize_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Initializes the HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(model_name=model_name)

#searches the research database for relevant chunks based on the user query
def search_research_db(query: str, collection: chromadb.Collection, embeddings: HuggingFaceEmbeddings, top_k: int = 5) -> List[dict]:
    """Retrieves relevant chunks from the database for a given query."""
    query_vector = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "distances"]
    )

    docs = results.get("documents") or [[]]
    dists = results.get("distances") or [[]]
    doc_list = docs[0] if docs else []
    dist_list = dists[0] if dists else []

    relevant_chunks = []
    for i, doc in enumerate(doc_list):
        if doc is None:
            continue
        dist = dist_list[i] if i < len(dist_list) else 0
        similarity = 1 - dist if isinstance(dist, (int, float)) else 0
        relevant_chunks.append({
            "content": doc if isinstance(doc, str) else str(doc),
            "title": "Python Expert Documentation",
            "similarity": similarity
        })
    return relevant_chunks

#generates the final answer using the retrieved chunks and the llm
def answer_research_question(query: str, collection: chromadb.Collection, embeddings: HuggingFaceEmbeddings, llm: ChatGroq) -> (str, List[dict]):
    """Generates a final answer using RAG."""
    relevant_chunks = search_research_db(query, collection, embeddings, top_k=5)

    context = "\n\n---\n\n".join([chunk["content"] for chunk in relevant_chunks])

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a Python Expert assistant. Answer the user's question 
        using ONLY the provided Python documentation and context. 
        If the context does not contain the answer, state that you cannot find the answer.
        
        RESEARCH CONTEXT: {context}
        RESEARCHER'S QUESTION: {question}
        
        EXPERT ANSWER:"""
    )

    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content, relevant_chunks


def build_research_db():
    """Load documents, chunk them, embed, and populate the ChromaDB. Run this once before using the app."""
    print("Building research database...")
    # Optional: convert PDF to text if the file exists
    if os.path.exists(PDF_INPUT_PATH):
        convert_pdf_to_text(PDF_INPUT_PATH, TXT_OUTPUT_PATH)
    else:
        print(f"PDF not found at {PDF_INPUT_PATH}, skipping. Using web docs only.")

    documents = load_all_documents()
    chunks = chunk_documents(documents)
    embeddings_model = initialize_embeddings(EMBEDDING_MODEL_NAME)

    # Embed all chunks (in batches to avoid memory issues)
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content for c in batch]
        all_embeddings.extend(embeddings_model.embed_documents(texts))

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Add to ChromaDB with ids, documents, and our embeddings
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents_list = [c.page_content for c in chunks]
    collection.add(ids=ids, documents=documents_list, embeddings=all_embeddings)
    print(f"Added {len(chunks)} chunks to '{COLLECTION_NAME}' at {DB_PATH}")


if __name__ == "__main__":
    build_research_db()

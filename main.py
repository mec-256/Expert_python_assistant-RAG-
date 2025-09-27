#importing all the necessary libraries
import os
import chromadb
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from typing import List, Any


#giving values to all the necessary variables
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "./research_db"
COLLECTION_NAME = "python_publications"
LLM_MODEL_NAME = "llama-3.1-8b-instant" 
PDF_INPUT_PATH = "fluent_python.pdf"
TXT_OUTPUT_PATH = "fluent_python.txt"

# --- HELPER FUNCTIONS ---
#converts the fluent python txtbook in pdf format to taxt format    
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

    relevant_chunks = []
    for i, doc in enumerate(results["documents"][0]):
        similarity = 1 - results["distances"][0][i]
        relevant_chunks.append({
            "content": doc,
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


#main function
def main():
    print("--- Starting Python Expert Assistant Setup ---")

    # 1. API Key Check
    if not os.environ.get("GROQ_API_KEY"):
        print("[ERROR] GROQ_API_KEY environment variable not set. Please set it before running this script.")
        return

    # 2. Initialize Components
    embeddings = initialize_embeddings(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # This correctly avoids the embedding function error with newer ChromaDB versions
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    try:
        llm = ChatGroq(model=LLM_MODEL_NAME)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize LLM: {e}")
        return

    # 3. INGESTION (Runs only if the database is empty)
    if collection.count() == 0:
        print("Database is empty. Starting data ingestion...")

        # A. Convert PDF (Uncomment the line below if you have 'fluent_python.pdf')
        # convert_pdf_to_text(PDF_INPUT_PATH, TXT_OUTPUT_PATH)

        # B. Load documents
        all_documents = load_all_documents()

        # C. Chunk the loaded documents
        all_chunks = chunk_documents(all_documents)

        # D. Store in ChromaDB
        chunk_contents = [doc.page_content for doc in all_chunks]
        print(f"Generating embeddings for {len(chunk_contents)} chunks...")
        chunk_embeddings = embeddings.embed_documents(chunk_contents)
        ids = [f"doc_{i}" for i in range(len(all_chunks))]

        print("Adding chunks to the database...")
        collection.add(
            documents=chunk_contents,
            embeddings=chunk_embeddings,
            ids=ids
        )
        print(f"Ingestion complete. Total chunks stored: {collection.count()}")

    else:
        print(f"Database found with {collection.count()} chunks. Skipping ingestion.")

    # 4. INTERACTION (Test Query)
    print("\n--- RAG System Ready ---")
    while True:
        query = input("\nAsk your Python question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        print(f"\nThinking...")
        answer, sources = answer_research_question(
            query, collection, embeddings, llm
        )

        print("\n--- AI EXPERT ANSWER ---")
        print(answer)
        print("\n--- Sources ---")
        for source in sources:
            print(f"- {source['title']} (Similarity: {source['similarity']:.4f})")

if __name__ == "__main__":
    main()
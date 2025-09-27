#  Python Expert RAG System with Streamlit UI

This project is an interactive AI assistant that answers questions about the Python programming language. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers based on official Python documentation, all through a clean and user-friendly web interface built with Streamlit.

---

###  Live Demo



****

---

## Core Features

-   **Interactive Chat Interface**: A simple and intuitive web UI built with Streamlit allows for easy interaction.
-   **Accurate & Sourced Answers**: The RAG system grounds its responses in a knowledge base created from official Python documentation, ensuring high accuracy.
-   **High-Speed Performance**: Leverages the Groq API for near-instantaneous LLM inference, providing a smooth user experience.
-   **Efficient Local Search**: Uses ChromaDB for fast, local vector similarity search to retrieve relevant context for every query.
-   **Automated Data Ingestion**: On the first run, the system automatically downloads the necessary documentation and builds the vector database.

---

##  Tech Stack: 

-   **Language**: Python
-   **Web Framework**: Streamlit
-   **LLM Orchestration**: LangChain
-   **LLM Provider**: Groq (Llama 3.1)
-   **Embeddings**: Sentence-Transformers
-   **Vector Database**: ChromaDB
-   **Document Loading**: PyMuPDF

---

## Getting Started 

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

-   Python 3.9 or higher
-   A free Groq API Key, which you can get from the [Groq Console](https://console.groq.com/keys).

###  Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Groq API key:**
    Create a new file named `.env` in the root of your project folder and add your API key to it:
    ```
    GROQ_API_KEY="YOUR_API_KEY_HERE"
    ```
    The application uses the `python-dotenv` library to load this key automatically.

###  Running the Application

1.  **Launch the Streamlit app from your terminal:**
    ```bash
    streamlit run app.py
    ```

2.  Your web browser will automatically open to the application's local address (usually `http://localhost:8501`).

3.  **Note on First Run**: The very first time you launch the app, it will take a few minutes to download the Python documentation and build the ChromaDB vector database. This is a one-time setup process. Subsequent launches will be much faster.

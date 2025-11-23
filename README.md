That's a great addition\! Clear instructions for setting up the Python environment are crucial for any project's `README.md`.

Here is the updated `README.md` content, with the new **"Setup Python Environment"** section added under **"Prerequisites"**.

# ☁️ LangGraph Weather + PDF RAG Demo

This project demonstrates a multi-functional AI agent built using **LangGraph** (for state management and routing) and **LangChain**. The agent can handle two distinct types of queries:

1.  **Weather Queries:** Uses the **OpenWeatherMap API** to get current weather data and summarizes it using an LLM.
2.  **RAG Queries:** Uses **Retrieval-Augmented Generation (RAG)** over a user-uploaded PDF, indexing chunks into a **Qdrant Vector Store** to answer document-specific questions.

The agent's core is a **router node** that directs the user's query to the appropriate tool (Weather API or RAG Retriever).

## ✨ Features

  * **Intelligent Routing:** Dynamically routes queries (e.g., "What is the temperature in Paris?" vs. "What is attention?") to the correct workflow.
  * **Vector Store Integration:** Uses **Qdrant** for vector indexing and retrieval, powered by **Sentence Transformers** embeddings.
  * **LLM Acceleration:** Leverages **Groq's** high-speed inference for fast responses.
  * **Interactive UI:** Hosted via **Streamlit** for easy PDF upload, indexing, and querying.
  * **Observability:** Integrated with **LangSmith** for full tracing of the graph's execution, including routing decisions and retrieval steps.

-----

## ⚙️ Setup and Installation

### Prerequisites

1.  **Python:** Python 3.10+
2.  **Qdrant:** A running instance of Qdrant (e.g., via Docker):
    ```bash
    docker pull qdrant/qdrant
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
    ```

### 1\. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2\. Setup Python Environment 🐍

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment named 'venv'
python3 -m venv .venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows (Command Prompt):
# venv\Scripts\activate.bat
# On Windows (PowerShell):
# venv\Scripts\Activate.ps1
```
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip3 install -r requirements.txt
```


### 3\. Install Dependencies

Install all required Python packages (assuming they are listed in `requirements.txt`):

```bash
pip install -r requirements.txt
```

### 4\. Configure Environment Variables

Create a file named `.env` in the root directory of your project and populate it with your API keys.

```dotenv
# .env file content

# Groq API Key (for LLM inference)
GROQ_API_KEY="your_groq_api_key"

# OpenWeatherMap API Key (for weather queries)
OPENWEATHER_API_KEY="your_openweathermap_api_key"

# LangSmith Tracing (for observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_PROJECT="LangGraph-Weather-RAG-Demo"
```

-----

## 🚀 Usage

### 1\. Start the Streamlit Application

Ensure your virtual environment is active and Qdrant is running, then start the application:

```bash
streamlit run streamlit_app.py
```

### 2\. Run Weather Queries (Default Mode)

When the app loads, you can immediately ask a weather question:

  * **Query:** `What is the current temperature in Berlin?`
  * **Result:** The graph routes to the `weather` node, calls the API, and summarizes the result.

### 3\. Run RAG Queries (After Indexing)

To enable RAG, follow these steps in the sidebar:

1.  **Upload PDF:** Use the **"Upload a PDF for RAG (optional)"** button to select your document.
2.  **Index PDF:** Click the **"Index PDF and enable RAG"** button. This process chunks the document, creates embeddings, and upserts them into the local Qdrant instance.
3.  **Ask RAG Query:** Ask a question specifically about the PDF content.
      * **Query (Example):** `What does the document say about the attention mechanism?`
      * **Result:** The graph routes to the `rag` node, retrieves relevant chunks from Qdrant, and uses the LLM to synthesize the answer based **only** on the retrieved context.

-----

## 📐 Architecture Overview

The system is built around a LangGraph `StateGraph` which manages the `AgentState`.

| Component | File | Role |
| :--- | :--- | :--- |
| **State Management** | `src/graph.py` | Defines `AgentState` (`query`, `route`, `answer`, etc.). |
| **Router** | `src/graph.py` (`router_node`) | Decides the workflow path based on query keywords (`weather`, `temperature`) and RAG enablement. |
| **RAG Node** | `src/graph.py` (`rag_node`) | Runs the retrieval pipeline: **Query $\rightarrow$ Embed $\rightarrow$ Qdrant Search $\rightarrow$ LLM Synthesis.** |
| **Weather Node** | `src/graph.py` (`weather_node`) | Calls the external `WeatherClient` and uses the LLM to format the response. |
| **Vector Store** | `rag.py` | Wrapper for **Qdrant** client, handling collection creation and vector upserting. |
| **Embedding** | `rag.py` (`HFEmbeddings`) | Uses `sentence-transformers/all-MiniLM-L6-v2` to convert text to vectors. |
| **UI** | `streamlit_app.py` | Manages file upload, state, indexing trigger, and displays results/debug info. |

-----

Would you like to refine any other sections of the `README.md` or perhaps focus on deployment instructions next?
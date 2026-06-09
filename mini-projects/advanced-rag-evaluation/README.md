# Advanced RAG Debugging & Evaluation

This project demonstrates how to systematically evaluate and debug a Retrieval-Augmented Generation (RAG) pipeline using **Ragas** (RAG Assessment) framework and LangChain.

Instead of guessing if a RAG pipeline is "good", this setup calculates specific performance metrics:
- **Faithfulness**: Is the answer derived *only* from the context?
- **Answer Relevancy**: Does the answer actually address the question?
- **Context Precision & Recall**: Did the retriever fetch the right documents?

## 📁 Files
- `rag_pipeline.py`: A basic RAG setup using LangChain, ChromaDB, and OpenAI.
- `evaluate_rag.py`: Script that generates a test dataset, evaluates it using Ragas metrics, and outputs a detailed CSV report.

## 🚀 Setup & Execution

1. **Install Dependencies**:
   ```bash
   pip install langchain langchain-openai chromadb ragas datasets pandas python-dotenv
   ```

2. **Environment Variables**:
   Create a `.env` file in this directory and add your API key:
   ```env
   OPENAI_API_KEY="your_api_key_here"
   ```

3. **Run Evaluation**:
   ```bash
   python evaluate_rag.py
   ```
   *This will run the test questions, evaluate the responses against the ground truth, and generate a `rag_evaluation_results.csv`.*

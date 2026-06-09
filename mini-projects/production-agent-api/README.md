# Production Agent API

This project demonstrates **End-to-End Production Deployment Enhancements** by taking a simple LangChain AI agent and deploying it using **FastAPI** and **Docker**.

Instead of running agents strictly via CLI scripts, production environments require robust APIs that can handle concurrency, validation, and containerization.

## 📁 Files
- `app.py`: The FastAPI application defining the REST endpoints, Request/Response validation, and agent execution.
- `requirements.txt`: Python dependencies needed for the application.
- `Dockerfile`: Configuration to containerize the FastAPI app into a reproducible environment.

## 🚀 Setup & Execution

### Option 1: Running Locally (Without Docker)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your API key:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
3. Run the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```
4. Open the interactive API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 2: Running with Docker
1. Build the Docker image:
   ```bash
   docker build -t production-agent-api .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 -e OPENAI_API_KEY="your_api_key_here" production-agent-api
   ```
3. Test the API:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/query" -H "Content-Type: application/json" -d '{"query": "What is machine learning?", "temperature": 0.5}'
   ```

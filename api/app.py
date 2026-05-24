from fastapi import FastAPI

app = FastAPI(title="TDP AI & ML API", description="Beginner friendly API for ML and GenAI models.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the TDP AI API! Check /docs for Swagger UI."}

@app.get("/ml/predict")
def ml_predict():
    return {"model": "Linear Regression", "prediction": 42.5}

@app.get("/genai/ask")
def genai_ask(query: str):
    return {"query": query, "response": "This is a mock response from the RAG pipeline."}

# Run using: uvicorn app:app --reload

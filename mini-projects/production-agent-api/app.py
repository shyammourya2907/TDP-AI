from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(
    title="Production Agent API",
    description="An end-to-end production deployment example of a simple AI Agent",
    version="1.0.0"
)

# API Models
class QueryRequest(BaseModel):
    query: str
    temperature: float = 0.7

class QueryResponse(BaseModel):
    answer: str

# Initialize generic components globally for reuse
llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

def get_agent_chain(temp: float):
    llm = ChatOpenAI(model=llm_model, temperature=temp)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert deployed in a production environment. Keep your answers concise and accurate."),
        ("user", "{input}")
    ])
    return prompt | llm | StrOutputParser()

@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Endpoint to process a user query using the LLM agent.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        chain = get_agent_chain(request.temperature)
        response = chain.invoke({"input": request.query})
        return QueryResponse(answer=response)
    except Exception as e:
        # In a real app, you would log this to a monitoring system
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker/Kubernetes."""
    return {"status": "healthy"}

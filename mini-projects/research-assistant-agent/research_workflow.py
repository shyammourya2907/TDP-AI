"""
Topic: Day 5 - LangGraph Project + Multi-Agent Workflow
Practical: Research Assistant Agent (Input -> Classifier -> Search -> Summary)
"""
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class ResearchState(TypedDict):
    question: str
    topic: str
    search_data: str
    summary: str

def question_classifier(state: ResearchState):
    print("[Classifier] Analyzing question...")
    return {"topic": "AI Trends"}

def search_agent(state: ResearchState):
    print(f"[Search] Googling topic: {state['topic']}...")
    return {"search_data": "AI is growing fast in 2024. Agentic workflows are popular."}

def summary_agent(state: ResearchState):
    print("[Summary] Synthesizing final answer...")
    return {"summary": f"Summary of {state['topic']}: {state['search_data']}"}

builder = StateGraph(ResearchState)
builder.add_node("classifier", question_classifier)
builder.add_node("search", search_agent)
builder.add_node("summary", summary_agent)

builder.add_edge(START, "classifier")
builder.add_edge("classifier", "search")
builder.add_edge("search", "summary")
builder.add_edge("summary", END)

graph = builder.compile()

print("--- Expected Outputs ---")
final_state = graph.invoke({
    "question": "What are the latest AI trends?", 
    "topic": "", 
    "search_data": "", 
    "summary": ""
})
print("Final Output:\n", final_state["summary"])

"""
Topic: LangGraph Tool Agent (Multi-Agent System)
Beginner Explanation: An agent that can decide whether to answer directly or use a "Tool" (like a calculator or search engine) based on the user's prompt.
"""
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

class AgentState(TypedDict):
    input: str
    tool_used: bool
    final_answer: str

def llm_node(state: AgentState):
    # Mock LLM Logic
    if "calculate" in state["input"].lower():
        return {"tool_used": True}
    return {"final_answer": "I can answer this directly.", "tool_used": False}

def tool_node(state: AgentState):
    return {"final_answer": "Calculated via Tool: 42"}

def router(state: AgentState):
    if state["tool_used"]:
        return "tool_node"
    return "end"

builder = StateGraph(AgentState)
builder.add_node("llm", llm_node)
builder.add_node("tool", tool_node)

builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", router, {"tool_node": "tool", "end": END})
builder.add_edge("tool", END)

graph = builder.compile()

print("--- Expected Outputs ---")
res1 = graph.invoke({"input": "What is AI?", "tool_used": False, "final_answer": ""})
print("Result 1:", res1["final_answer"])

res2 = graph.invoke({"input": "Calculate 2+2", "tool_used": False, "final_answer": ""})
print("Result 2:", res2["final_answer"])

# Interview Question:
# Q: What is a Conditional Edge in LangGraph?
# A: A conditional edge uses a router function to dynamically decide the next node to execute based on the current state (e.g., routing to a tool if the LLM requests it).

"""
Topic: Day 3 - Multi-AI Agent Systems (Planner + Tool Agent)
Practical: Planner agent (decides next steps) -> Tool agent (executes tasks)
"""
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class MultiAgentState(TypedDict):
    task: str
    plan: list
    results: list

def planner_agent(state: MultiAgentState):
    print("[Planner Agent] Breaking down task...")
    plan = ["Step 1: Search", "Step 2: Calculate"]
    return {"plan": plan, "results": []}

def tool_agent(state: MultiAgentState):
    print(f"[Tool Agent] Executing plan: {state['plan']}")
    results = ["Search result found", "Calculation done: 42"]
    return {"results": results}

builder = StateGraph(MultiAgentState)
builder.add_node("planner", planner_agent)
builder.add_node("executor", tool_agent)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", END)

graph = builder.compile()

print("--- Expected Outputs ---")
res = graph.invoke({"task": "Find Apple stock and multiply by 10", "plan": [], "results": []})
print("Final Results:", res["results"])

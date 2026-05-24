"""
Topic: LangGraph Basics (Nodes & Edges)
Beginner Explanation: LangGraph is like a flowchart for AI.
- Nodes: Do the work (e.g., an LLM thinking).
- Edges: Connect nodes to pass the "State" (memory) around.
"""
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1. Define the State (Memory)
class State(TypedDict):
    message: str

# 2. Define Nodes (Functions)
def node_a(state: State):
    print("Node A: Adding Hello")
    return {"message": state["message"] + " Hello"}

def node_b(state: State):
    print("Node B: Adding World")
    return {"message": state["message"] + " World!"}

# 3. Build the Graph
builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)

# 4. Connect with Edges
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

# 5. Compile and Run
graph = builder.compile()
result = graph.invoke({"message": ""})

print("--- Expected Outputs ---")
print("Final State:", result["message"])

# Interview Question:
# Q: What is the 'State' in LangGraph?
# A: The State is a dictionary or object that holds the shared data (like conversation history) passed between all nodes during the graph's execution.

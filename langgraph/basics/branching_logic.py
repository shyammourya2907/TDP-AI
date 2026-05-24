"""
Topic: Day 2 - State Management & Branching Logic (3 Agents)
Practical: Input Agent -> Decision Agent -> Action Agent with branching based on input type.
"""
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    user_input: str
    intent: str
    action_taken: str

# 1. Input Agent
def input_agent(state: State):
    print(f"[Input Agent] Received: {state['user_input']}")
    return state

# 2. Decision Agent (LLM Router Mock)
def decision_agent(state: State):
    intent = "math" if "calculate" in state["user_input"].lower() else "chat"
    print(f"[Decision Agent] Decided intent is: {intent}")
    return {"intent": intent}

# 3a. Action Agent (Math)
def math_action_agent(state: State):
    print("[Action Agent - Math] Calculating...")
    return {"action_taken": "Calculated Math Equation"}

# 3b. Action Agent (Chat)
def chat_action_agent(state: State):
    print("[Action Agent - Chat] Generating conversation...")
    return {"action_taken": "Responded to Chat"}

# Branching Logic
def route_action(state: State):
    if state["intent"] == "math":
        return "math_action"
    return "chat_action"

builder = StateGraph(State)
builder.add_node("input", input_agent)
builder.add_node("decision", decision_agent)
builder.add_node("math_action", math_action_agent)
builder.add_node("chat_action", chat_action_agent)

builder.add_edge(START, "input")
builder.add_edge("input", "decision")
builder.add_conditional_edges("decision", route_action, {"math_action": "math_action", "chat_action": "chat_action"})
builder.add_edge("math_action", END)
builder.add_edge("chat_action", END)

graph = builder.compile()

print("--- Expected Outputs ---")
graph.invoke({"user_input": "calculate 5 * 10", "intent": "", "action_taken": ""})
print("-" * 20)
graph.invoke({"user_input": "Hello, how are you?", "intent": "", "action_taken": ""})

"""
Topic: Day 2 - LangChain Agents & Tools
Practical: Implementing LangChain agents and integrating tools.
"""
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import OpenAI
import os

# Mock Tool Functions
def search_wikipedia(query):
    return "LangChain is a framework for developing applications powered by language models."

def calculate(equation):
    return "42"

# 1. Define Tools
tools = [
    Tool(
        name="Wikipedia Search",
        func=search_wikipedia,
        description="Useful for when you need to answer questions about current events or facts."
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="Useful for when you need to answer questions about math."
    )
]

print("--- Expected Output ---")
print("Agent initialized with tools:", [t.name for t in tools])
print("Agent decides to use Calculator for: 'What is 20 + 22?' -> Output:", calculate("20+22"))

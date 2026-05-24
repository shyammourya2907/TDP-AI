"""
Topic: Day 3 - LangChain Memory & Retrievers
Practical: Implementing memory for context management.
"""
from langchain.memory import ConversationBufferMemory

# 1. Initialize Memory
memory = ConversationBufferMemory()

# 2. Simulate Chat
memory.save_context({"input": "Hi, my name is Shyam"}, {"output": "Hello Shyam!"})
memory.save_context({"input": "What is GenAI?"}, {"output": "It creates new content."})

print("--- Expected Output ---")
print("Memory Buffer History:")
print(memory.load_memory_variables({}))

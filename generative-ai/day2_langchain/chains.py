"""
Topic: Day 2 - LangChain Chains
Practical: Building and using chains for NLP tasks.
"""
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# 1. Prompt Template
prompt = PromptTemplate.from_template("Translate the following English text to French: {text}")

# 2. Mock LLM execution (Simulating chain invoke)
def mock_chain_invoke(text):
    if "hello" in text.lower():
        return "Bonjour"
    return "Je ne sais pas"

print("--- Expected Output ---")
print("Input: Hello")
print("Output:", mock_chain_invoke("Hello"))

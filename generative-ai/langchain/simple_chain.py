"""
Topic: LangChain Simple Chain
Beginner Explanation: LangChain allows us to connect a Prompt to an LLM, and then to an Output Parser. This "Chain" handles the flow of data automatically.
"""
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. Define Prompt
prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")

# 2. Define LLM (using a mock object for beginner running without API key)
class MockLLM:
    def invoke(self, prompt_text):
        return f"Eco{prompt_text.split()[-1].title()} Co."

# 3. Form Chain
# In real code: chain = prompt | ChatOpenAI() | StrOutputParser()
print("--- Expected Outputs ---")
print("Mock Output:", MockLLM().invoke("eco-friendly water bottles"))

# Interview Question:
# Q: What is the benefit of LangChain over calling the OpenAI API directly?
# A: LangChain provides abstractions for chaining, memory, tools, and retrievers, making it much easier to build complex, stateful applications.

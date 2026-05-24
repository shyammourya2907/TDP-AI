"""
Topic: Prompt Engineering (Chain of Thought)
Beginner Explanation: Instead of just asking for an answer, we ask the AI to "think step by step". This drastically improves accuracy on complex tasks like math or logic.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

# Ensure OPENAI_API_KEY is set in your environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "dummy_key")

prompt = PromptTemplate(
    input_variables=["math_problem"],
    template="Solve this math problem by thinking step-by-step: {math_problem}"
)

print("--- Expected Outputs ---")
print("Prompt:")
print(prompt.format(math_problem="If I have 5 apples and eat 2, then buy 3 more, how many do I have?"))

# Interview Question:
# Q: What is Few-Shot Prompting?
# A: Providing the AI with a few examples of the desired input-output format within the prompt before asking it to solve a new problem.

"""
Topic: Day 3 - Embeddings & Output Parsers
Practical: Working with embeddings and parsing outputs.
"""
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# Output Parser setup
parser = CommaSeparatedListOutputParser()

# Mocking LLM output that needs parsing
llm_output = "apple, banana, orange, grape"
parsed_output = parser.parse(llm_output)

print("--- Expected Output ---")
print("Raw string:", llm_output)
print("Parsed into Python List:", parsed_output)

"""
Topic: Day 4 - PDF Document Question Answering LLM
Practical: Complete pipeline to query a PDF.
"""
from langchain_community.document_loaders import PyPDFLoader
import os

# Create a dummy PDF text file representing the PDF content
dummy_pdf_content = "This PDF contains policy information. Rule 1: Always be learning."

print("--- Expected Output ---")
print("Step 1: PyPDFLoader loads the document.")
print("Step 2: TextSplitter chunks the PDF into 100-word segments.")
print("Step 3: OpenAIEmbeddings converts chunks to vectors.")
print("Step 4: ChromaDB stores the vectors.")
print("Step 5: User queries 'What is Rule 1?' -> LLM responds based on retrieved chunk.")
print(f"Mock Answer: Based on the PDF, {dummy_pdf_content.split('.')[1]}.")

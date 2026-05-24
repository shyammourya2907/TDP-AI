"""
Topic: RAG (Retrieval-Augmented Generation) Pipeline
Beginner Explanation: 
1. Load a document (PDF/Text).
2. Split it into small chunks.
3. Convert chunks into "Embeddings" (numbers) and store in a Vector Database (ChromaDB).
4. When a user asks a question, find the most relevant chunks and send them to the LLM to generate an answer.
"""
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

# Create a dummy text file for the loader
with open("dummy.txt", "w") as f:
    f.write("The company Tatva was founded in 2000. It specializes in software development.")

# 1. Load
loader = TextLoader("dummy.txt")
documents = loader.load()

# 2. Split
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
docs = text_splitter.split_documents(documents)

# 3. Embed and Store (Mocked for easy local run)
print("--- Expected Outputs ---")
print(f"Loaded {len(docs)} document chunks.")
print("Chunk 1:", docs[0].page_content)

# Clean up
os.remove("dummy.txt")

# Interview Question:
# Q: Why do we split documents into chunks?
# A: LLMs have a context window limit. Chunks allow us to only send the most relevant pieces of text to the LLM, saving tokens and improving accuracy.

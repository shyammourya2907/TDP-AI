# Day 3: Debugging a RAG Pipeline

When a Retrieval-Augmented Generation (RAG) pipeline fails, it's usually due to one of three things:

## 1. Bad Chunking
- **Issue:** The LLM gets cut-off sentences or lacks context.
- **Fix:** Adjust the `chunk_size` and `chunk_overlap`. Ensure chunks are large enough to contain full thoughts but small enough to fit in the context window.

## 2. Wrong Embeddings
- **Issue:** Searching for "car" returns documents about "apples".
- **Fix:** Ensure you are using a high-quality embedding model (like OpenAI's `text-embedding-ada-002` or HuggingFace's `all-MiniLM-L6-v2`).

## 3. No Retrieval Results
- **Issue:** The Vector DB returns zero documents for a query.
- **Fix:** Lower the similarity threshold metric (e.g., Cosine Similarity). Turn on LangChain debug mode (`langchain.debug = True`) to see the exact query being sent to the DB.

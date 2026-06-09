import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Sample Documents
documents = [
    "The TDP program focuses on training developers in AI and Machine Learning.",
    "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
    "RAG stands for Retrieval-Augmented Generation, which improves LLM accuracy by fetching context.",
    "PEFT allows fine-tuning of large language models with minimal compute resources."
]

def build_rag_pipeline():
    # 1. Create Embeddings & VectorStore
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 2. Setup LLM & Prompt
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    Answer:"""
    prompt = PromptTemplate.from_template(template)

    # 3. Create Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

if __name__ == "__main__":
    chain, _ = build_rag_pipeline()
    print("Testing RAG Pipeline...")
    response = chain.invoke("What does RAG stand for?")
    print(f"Response: {response}")

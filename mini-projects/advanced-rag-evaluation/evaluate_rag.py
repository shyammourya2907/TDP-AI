import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from rag_pipeline import build_rag_pipeline
from dotenv import load_dotenv

load_dotenv()

def generate_evaluation_dataset():
    chain, retriever = build_rag_pipeline()
    
    # Ground truth data for evaluation
    questions = [
        "What does RAG stand for?",
        "What is the focus of the TDP program?",
        "How does PEFT help?"
    ]
    ground_truths = [
        ["RAG stands for Retrieval-Augmented Generation."],
        ["The TDP program focuses on training developers in AI and Machine Learning."],
        ["PEFT allows fine-tuning large models with minimal compute resources."]
    ]
    
    answers = []
    contexts = []
    
    print("Generating responses for evaluation dataset...")
    for q in questions:
        # Get contexts
        docs = retriever.invoke(q)
        contexts.append([doc.page_content for doc in docs])
        
        # Get answer
        ans = chain.invoke(q)
        answers.append(ans)
        
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    return Dataset.from_dict(data)

def main():
    print("Preparing Dataset...")
    dataset = generate_evaluation_dataset()
    
    print("\nRunning Ragas Evaluation...")
    # Evaluate using RAGAS metrics
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    )
    
    print("\n--- Evaluation Results ---")
    df = result.to_pandas()
    print(df[['question', 'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']])
    
    df.to_csv("rag_evaluation_results.csv", index=False)
    print("\nDetailed results saved to 'rag_evaluation_results.csv'")

if __name__ == "__main__":
    main()

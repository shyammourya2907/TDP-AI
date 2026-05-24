# Day 5: Model Fine-Tuning

## What is Fine-Tuning?
Fine-tuning takes a pre-trained model (like BERT or GPT-3) that already understands general language, and trains it further on a specific, smaller dataset to make it an expert in a specific domain (e.g., Legal documents).

## Fine-Tuning vs RAG
- **RAG:** Gives the model a "textbook" to read before answering. Cheaper and updates instantly.
- **Fine-Tuning:** Changes the model's internal "brain". Better for teaching a model a new *format* or *style* of responding rather than specific facts.

## Evaluation Metrics
1. **Accuracy:** Percentage of correct predictions (best for classification).
2. **F1-Score:** Balance between Precision and Recall. Good for imbalanced datasets.
3. **BLEU Score:** Measures how similar the generated text is to a human reference text (used heavily in translation).

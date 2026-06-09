# Advanced LLM Fine-Tuning using PEFT & LoRA

This project demonstrates how to fine-tune a Large Language Model (LLM) on a custom dataset using **Parameter-Efficient Fine-Tuning (PEFT)** and **Low-Rank Adaptation (LoRA)**. 

Instead of training billions of parameters (which requires massive GPU clusters), LoRA freezes the base model weights and only trains a tiny set of adapter weights.

## 📁 Files
- `dataset.jsonl`: Sample custom dataset for instruction tuning.
- `fine_tune.py`: Script to load the base model, apply LoRA, and train using `SFTTrainer`.
- `inference.py`: Script to load the base model + LoRA adapter to generate text.

## 🚀 Setup & Execution

1. **Install Dependencies** (It's recommended to do this in an environment with GPU support like Google Colab if you don't have a local GPU):
   ```bash
   pip install transformers peft datasets trl bitsandbytes accelerate
   ```

2. **Run Fine-Tuning**:
   ```bash
   python fine_tune.py
   ```
   *This will create a `results-lora` folder containing the adapter weights.*

3. **Run Inference**:
   ```bash
   python inference.py
   ```
   *This loads the adapter and base model to generate a response.*

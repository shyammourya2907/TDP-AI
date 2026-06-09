import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Configuration
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # A small, lightweight model for testing
OUTPUT_DIR = "./results-lora"
DATASET_PATH = "dataset.jsonl"

def main():
    print(f"Loading Base Model: {MODEL_ID}")
    
    # 2. Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Load model with 4-bit quantization for memory efficiency (QoRA/LoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False
    
    # 4. Prepare for PEFT/LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # 5. Load Dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # Format dataset prompt
    def format_prompt(example):
        instruction = example['instruction']
        output = example['output']
        return f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{instruction}</s>\n<|assistant|>\n{output}</s>"
    
    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=10,
        logging_steps=1,
        learning_rate=2e-4,
        fp16=False,
        bf16=False, # Set to True if your GPU supports bfloat16
        max_grad_norm=0.3,
        max_steps=20, # Small steps for demonstration
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant"
    )
    
    # 7. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=512,
        dataset_text_field="text", # SFTTrainer requires a specific formatting, we handle it below
        formatting_func=format_prompt,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    # 8. Start Fine-Tuning
    print("Starting fine-tuning...")
    trainer.train()
    
    # 9. Save the Adapter Model
    print(f"Saving adapter to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()

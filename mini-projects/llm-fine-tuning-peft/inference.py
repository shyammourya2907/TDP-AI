import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Configuration
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./results-lora"

def main():
    print(f"Loading Base Model: {BASE_MODEL_ID}")
    
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load PEFT model (Adapter)
    print(f"Loading Adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # Generate text
    instruction = "What is machine learning?"
    prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{instruction}</s>\n<|assistant|>\n"
    
    print("\n--- Generating Response ---")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

if __name__ == "__main__":
    main()

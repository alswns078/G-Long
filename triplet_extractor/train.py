import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import os
import glob
import xml.etree.ElementTree as ET

# ==========================================
# 1. Data Loading and Preprocessing Functions
# ==========================================
def parse_convai_data(file_path):
    """
    Converts ConvAI format to Chat Format (messages) for LLM training.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    formatted_data = []
    
    # Define system prompt
    system_prompt = (
        "You are an expert in Knowledge Graph Construction. "
        "Extract (head, relation, tail) triplets from the text and output them in JSON format."
    )
    
    for entry in raw_data:
        t_data = entry.get('triplets', {})
        if not t_data: continue
        
        t_data=t_data[0]

        tokens = t_data['tokens']
        head_idxs = t_data['head']
        tail_idxs = t_data['tail']
        label = t_data['label']

        # 1. Restore sentence
        sentence = " ".join(tokens).replace(" ,", ",").replace(" .", ".").replace(" !", "!")

        # 2. Restore Entity
        head_entity = " ".join([tokens[i] for i in head_idxs])
        tail_entity = " ".join([tokens[i] for i in tail_idxs])

        # 3. Generate target JSON
        target_json = [{
            "head": head_entity,
            "relation": label,
            "tail": tail_entity
        }]

        # 4. [Key Change] Save as Messages structure
        # Qwen and the latest TRL automatically recognize this structure and apply the Chat Template.
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract triplets:\n{sentence}"},
            {"role": "assistant", "content": json.dumps(target_json, ensure_ascii=False)}
        ]

        # Dictionary key must be 'messages'
        formatted_data.append({"messages": conversation})
    
    return formatted_data


# ==========================================
# Main Training Configuration
# ==========================================
if __name__ == "__main__":
    # Path configuration
    #DATA_PATH = "YOUR/UPLOADED/FILE/PATH.json" # Path to uploaded file
    DATA_PATH = "YOUR/DATASET/PATH" # Path to uploaded file
    MODEL_ID = "Qwen/Qwen3-8B"
    OUTPUT_DIR = f"./{MODEL_ID.split('/')[-1]}-{DATA_PATH.split('/')[-1]}-lora-1triples"
    

    # 1. Prepare data
    # Uncomment below to load if the actual file exists
    data = parse_convai_data(DATA_PATH)
    
    dataset = Dataset.from_list(data)
    print(f"Dataset Size: {len(dataset)}")

    # 2. Load Model & Tokenizer (QLoRA: 4bit Quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Set Qwen padding

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        r=32,                        # ConvAI is 16, dialogRE is 8, webnlg is 32?
        lora_alpha=32,               # Scaling factor
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Target all linear layers
    )

    # 4. Trainer Configuration
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=16,     # ConvAI is 64, dialogRE is 16
        gradient_accumulation_steps=16,      # ConvAI is 4, dialogRE is 16
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        report_to="none", # Change to "wandb" if using wandb
        completion_only_loss=True,
    )

    # Configure to calculate Loss only for the response (Assistant) part
    response_template = "<|im_start|>assistant\n" 

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        #formatting_func=formatting_prompts_func, # Use formatting function defined above
        #data_collator=collator,
        args=args,
        #max_length=512, # Adjust according to data length
    )

    # 5. Start Training
    print("Starting Training...")
    trainer.train()

    # 6. Save
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
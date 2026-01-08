import json
import re
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

# ==========================================
# 1. Text parsing
# ==========================================

def aggressive_clean(text_value):
    words = text_value.split()
    if not words: return ""
    
    # 1. Remove consecutive duplicates by iterating through the list
    deduped = [words[i] for i in range(len(words)) if i == 0 or words[i] != words[i-1]]
    
    # 2. (Optional) Remove the last word if it is too short (1~2 characters) or a stopword
    if len(deduped) > 1 and len(deduped[-1]) < 3:
        deduped.pop()
        
    return " ".join(deduped)

def parse_llm_output(output_text):
    try:
        # 1. Remove Markdown and miscellaneous text
        clean_text = re.sub(r'```json\s*', '', output_text)
        clean_text = re.sub(r'```', '', clean_text).strip()
        clean_text = re.sub(r'<think>', '', clean_text).strip()
        clean_text = re.sub(r'</think>', '', clean_text).strip()
        clean_text = re.sub(r'json\n', '', clean_text).strip()
        clean_text = re.sub(r'JSON\n', '', clean_text).strip()
        
        # Attempt to extract parts enclosed in []
        match = re.search(r'(\[.*\])', clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1)
        
        clean_text = clean_text.strip("\"")
        
        # Attempt to correct JSON syntax errors
        if not clean_text.endswith(']'):
            clean_text = clean_text.rstrip('}')
            clean_text = aggressive_clean(clean_text)

        if not clean_text.endswith('}]'):
            if clean_text.endswith("\\\""):
                clean_text += ('\\\"}]')
            clean_text += ('}]')
        
        if clean_text.endswith('}}]'):
            clean_text = clean_text.replace('}}]','}]')
        
        if not clean_text:
            return [] # Return empty list if value is empty
            
        # 2. JSON Parsing
        return json.loads(clean_text)
    
    except json.JSONDecodeError:
        # Return original text (or empty list) after logging if parsing fails
        # print(f"JSON Parsing Error: {output_text[:100]}...") 
        return [output_text]

def create_prompt(sentence, tokenizer, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract triplets:\n{sentence}"}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False # Check support depending on model (e.g., Qwen)
    )
    # Force add <|im_start|>assistant: (if necessary)
    return prompt + "\n<|im_start|>assistant:"

# ==========================================
# 2. Inference
# ==========================================

def run_inference(input_file, output_file, model_path, lora_path=None, batch_size=500):
    
    # Define System Prompt
    SYSTEM_PROMPT = (
        "You are an expert in Knowledge Graph Construction. "
        "Read the text and extract (subject, relation, object) triplets in JSON format."
        "Output strictly in JSON format as a list of dictionaries."
    )

    # 1. Load Data and Preprocessing (Flattening)
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f) # List[dict] format
        except:
            f.seek(0)
            data = [json.loads(line) for line in f] # JSONL format

    flat_texts = []
    map_info = [] 
    fields_to_process = ['persona1', 'persona2', 'dialogue']
    
    print("Flattening data structures...")
    for idx, entry in enumerate(data):
        for field in fields_to_process:
            if field in entry:
                lines = entry[field]
                if isinstance(lines, str):
                    lines = [lines]
                    
                for sent_idx, sentence in enumerate(lines):
                    flat_texts.append(sentence)
                    map_info.append((idx, field, sent_idx))

    total_len = len(flat_texts)
    print(f"Total sentences to process: {total_len}")

    # 2. Initialize structure for result saving
    processed_data = [entry.copy() for entry in data]
    for idx in range(len(data)):
        for field in fields_to_process:
            if field in entry:
                original_len = len(entry[field]) if isinstance(entry[field], list) else 1
                processed_data[idx][f"{field}_triplets"] = [None] * original_len

    # 3. Load Tokenizer
    print(f"Loading Tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 4. Generate Prompts
    print("Generating prompts...")
    prompts = [create_prompt(text, tokenizer, SYSTEM_PROMPT) for text in flat_texts]

    # 5. Initialize vLLM model
    print(f"Initializing vLLM model: {model_path}")
    if lora_path:
        print(f"LoRA Adapter Enabled: {lora_path}")
    
    llm = LLM(
        model=model_path,
        enable_lora=(lora_path is not None), # True if LoRA path exists
        max_lora_rank=32 if lora_path else None,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        max_model_len=4096
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["<|im_end|>"]
    )

    # Prepare LoRA Request object (only in LoRA mode)
    lora_request = LoRARequest("custom_adapter", 1, lora_path) if lora_path else None

    # 6. Execute Batch Inference
    print("Starting vLLM Inference...")
    
    for i in tqdm(range(0, total_len, batch_size), desc="Batch Processing"):
        batch_prompts = prompts[i : i + batch_size]
        
        # Pass lora_request when calling generate (General inference if None)
        batch_outputs = llm.generate(
            batch_prompts, 
            sampling_params, 
            lora_request=lora_request,
            use_tqdm=False
        )
        
        # Result Mapping
        for j, output in enumerate(batch_outputs):
            global_idx = i + j
            data_idx, field, sent_idx = map_info[global_idx]
            original_text = flat_texts[global_idx]
            
            generated_text = output.outputs[0].text.strip()
            triplets = parse_llm_output(generated_text)
            
            result_item = {
                "sentence": original_text,
                "triplets": triplets,
                # "raw_output": generated_text 
            }
            
            target_list = processed_data[data_idx].get(f"{field}_triplets")
            if target_list is not None and sent_idx < len(target_list):
                target_list[sent_idx] = result_item

        if (i // batch_size) % 5 == 0:
             with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=4, ensure_ascii=False)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    
    # 1. Common Configuration
    BASE_MODEL = "Qwen/Qwen3-8B" 
    INPUT_FILE = "YOUR/INFERENCE/DATAFILE/PATH.json"
    
    DATASET = "YOUR/TRAIN/DATAFILE/PATH.json"
    LORA_PATH = "YOUR/LORA/ADAPTER/FILEPATH"  # None to use untrained model
    OUTPUT_FILE = "YOUR/OUTPUT/FILEPATH.json"


    # 3. Execution
    run_inference(
        input_file=INPUT_FILE, 
        output_file=OUTPUT_FILE, 
        model_path=BASE_MODEL, 
        lora_path=LORA_PATH,
        batch_size=1000
    )
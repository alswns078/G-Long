import json
import torch
import os
import re
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.nn.utils.rnn import pad_sequence

# tqdm configuration
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, **kwargs): return iterator

# ==========================================
# 1. Robust Parser
# ==========================================
def fix_and_parse_json(dirty_data):
    if isinstance(dirty_data, list):
        if not dirty_data: return []
        if isinstance(dirty_data[0], dict): return dirty_data
        
    if isinstance(dirty_data, list):
        text_candidates = []
        for item in dirty_data:
            if isinstance(item, str): text_candidates.append(item)
            elif isinstance(item, dict): text_candidates.append(json.dumps(item))
            elif isinstance(item, list): text_candidates.append(str(item))
        text = " ".join(text_candidates)
    else:
        text = str(dirty_data)
        
    clean_text = text.replace('\n', '').replace('\r', '')
    matches = re.findall(r'\{[^{}]+\}', clean_text)
    
    recovered_items = []
    for match in matches:
        try:
            obj = json.loads(match)
            recovered_items.append(obj)
        except json.JSONDecodeError:
            try:
                cleaned_match = match.replace('\\"', '"')
                obj = json.loads(cleaned_match)
                recovered_items.append(obj)
            except:
                continue
    return recovered_items

def check_triplet_same(triple1, triple2):
    required = ["head", "relation", "tail"]
    if not all(k in triple1 for k in required) or not all(k in triple2 for k in required):
        return False
    return (triple1["head"] == triple2['head'] and 
            triple1["relation"] == triple2["relation"] and 
            triple1['tail'] == triple2['tail'])

# ==========================================
# 2. Load and Group
# ==========================================
def load_and_group_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Loading raw data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]

    grouped_data = defaultdict(list)
    
    for entry in data:
        d_id = entry.get('dialogue_id')
        if d_id is not None:
            grouped_data[d_id].append(entry)
            
    for d_id in grouped_data:
        grouped_data[d_id].sort(key=lambda x: x.get('session_id', 0))

    return grouped_data

# ==========================================
# 3. Importance Scorer
# ==========================================
class TripletSummarizationScorer:
    def __init__(self, model_name="chanifrusydi/t5-dialogue-summarization", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model: {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()

    def get_batch_triplet_scores(self, batch_triplets_list):
        if not batch_triplets_list:
            return []

        batch_input_ids = []
        batch_ranges = []
        
        for triplets in batch_triplets_list:
            current_ids = []
            current_ranges = []
            idx_cursor = 0
            
            for t_text in triplets:
                ids = self.tokenizer.encode(t_text + ". ", add_special_tokens=False)
                if not ids:
                    current_ranges.append(None)
                    continue
                
                current_ids.extend(ids)
                start = idx_cursor
                end = idx_cursor + len(ids)
                current_ranges.append((start, end))
                idx_cursor = end
            
            if not current_ids:
                current_ids = [self.tokenizer.pad_token_id]
            
            if len(current_ids) > 1024:
                current_ids = current_ids[:1024]
            
            batch_input_ids.append(torch.tensor(current_ids, dtype=torch.long))
            batch_ranges.append(current_ranges)

        input_tensor = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        attention_mask = (input_tensor != self.tokenizer.pad_token_id).long().to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_length=150,
                min_length=10, 
                num_beams=4,
                early_stopping=True
            )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                decoder_input_ids=summary_ids,
                output_attentions=True
            )

        cross_attn = outputs.cross_attentions[-1] 
        attn_avg = cross_attn.mean(dim=1)
        
        final_batch_scores = []
        for i in range(len(batch_triplets_list)):
            token_scores = attn_avg[i].sum(dim=0).cpu().float().numpy()
            session_scores = []
            ranges = batch_ranges[i]
            
            for r in ranges:
                if r is None:
                    session_scores.append(0.0)
                    continue
                
                start, end = r
                if start >= len(token_scores):
                    session_scores.append(0.0)
                    continue
                
                real_end = min(end, len(token_scores))
                score = np.sum(token_scores[start:real_end])
                session_scores.append(float(score))
                
            final_batch_scores.append(session_scores)

        return final_batch_scores

# ==========================================
# 4. Main Execution Function
# ==========================================
def main():
    input_file = "YOUR/INPUT/FILE/PATH.json" 
    output_file = "YOUR/OUTPUT/FILE/PATH.json"
    BATCH_SIZE = 8
    
    try:
        grouped_data = load_and_group_data(input_file)
    except Exception as e:
        print(f"Error: {e}")
        return

    scorer = TripletSummarizationScorer()
    final_output_list = []
    
    print(f"Total Dialogue Groups: {len(grouped_data)}")
    
    for d_id, entries in tqdm(grouped_data.items(), desc="Scoring MSC"):
        
        # --- [Step 1] Prepare Data per Session ---
        session_triplets_batch_input = []
        
        # Save structure per session: [ {parsed_triplets, item_index}, ... ]
        # Pre-parse and store for reuse later
        session_parsed_data = [] 
        
        # Collect all raw scores for Global Max calculation
        all_raw_scores_flat = []

        for entry in entries:
            triplet_strs_session = []
            
            # List to temporarily store parsed info for this session
            current_session_items = []
            
            raw_dt_list = entry.get('dialogue_triplets', [])
            
            for item in raw_dt_list:
                if not isinstance(item, dict): 
                    # Handle empty items
                    current_session_items.append({'text': '', 'triplets': [], 'count': 0})
                    continue
                
                # MSC typically has 'sentence' or 'text', and 'triplets'
                text_content = item.get('sentence', item.get('text', ''))
                
                raw_val = item.get('triplets', item.get('triplet', []))
                parsed = fix_and_parse_json(raw_val)
                
                unique = []
                for t in parsed:
                    if not isinstance(t, dict): continue
                    if not all(k in t for k in ["head", "relation", "tail"]): continue
                    unique.append(t)
                
                # Generate text for batch input
                for t_obj in unique:
                    t_str = f"{t_obj.get('head','')} {t_obj.get('relation','')} {t_obj.get('tail','')}"
                    triplet_strs_session.append(t_str)
                
                # Save parsing results
                current_session_items.append({
                    'text': text_content,
                    'triplets': unique,
                    'count': len(unique)
                })
            
            session_triplets_batch_input.append(triplet_strs_session)
            session_parsed_data.append(current_session_items)

        # --- [Step 2] Execute Batch Model ---
        batch_results_per_session = [] 
        
        for i in range(0, len(session_triplets_batch_input), BATCH_SIZE):
            batch = session_triplets_batch_input[i : i + BATCH_SIZE]
            scores_list = scorer.get_batch_triplet_scores(batch)
            batch_results_per_session.extend(scores_list)
            
            for s_scores in scores_list:
                all_raw_scores_flat.extend(s_scores)

        # --- [Step 3] Global Normalization and Structure Transformation ---
        global_max = 0.0
        if all_raw_scores_flat:
            global_max = max(all_raw_scores_flat)
        
        # Iterate through sessions again to assemble results
        for entry_idx, entry in enumerate(entries):
            session_flat_scores = batch_results_per_session[entry_idx]
            session_items = session_parsed_data[entry_idx]
            
            new_dt_list = []
            score_cursor = 0
            
            for item_info in session_items:
                count = item_info['count']
                valid_triplets = item_info['triplets']
                text_content = item_info['text']
                
                utterance_score = 0.0
                
                if count > 0:
                    # Retrieve scores for triplets belonging to this sentence
                    my_scores = session_flat_scores[score_cursor : score_cursor + count]
                    score_cursor += count
                    
                    # [Key] Use Max Score as the representative score
                    raw_max = max(my_scores)
                    
                    if global_max > 1e-9:
                        utterance_score = round((raw_max / global_max) * 10, 2)
                
                # [Final Structure Generation]
                new_obj = {
                    "text": text_content,
                    "triple": valid_triplets, # 'triplets' -> 'triple' (maintain list format)
                    "importance_score": utterance_score
                }
                new_dt_list.append(new_obj)
            
            # Update results
            entry['dialogue_triplets'] = new_dt_list
            final_output_list.append(entry)

    # 4. Save
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output_list, f, indent=4, ensure_ascii=False)
        
    print("Done!")

if __name__ == "__main__":
    main()
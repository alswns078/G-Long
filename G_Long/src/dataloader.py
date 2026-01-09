import os 
import time
import json
import numpy as np
from nltk.util import ngrams
from nlgeval import calc_nlg_metrics

from src.client import OpenAIClient
from src.memory import GraphEnhancedMemory
from src.generator import Generator

def convert_seconds_to_full_time(seconds):
    units = [("years", 31536000), ("months", 2592000), ("days", 86400), ("hours", 3600), ("minutes", 60)]
    parts = []
    for name, count in units:
        value, seconds = divmod(seconds, count)
        if value: parts.append(f"{value} {name}")
    return " ".join(parts)

class MSC():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.detailed_generation_logs = []
        
        # Initialize Unified OpenAI Client
        self.client = OpenAIClient(args.model, logger, args)
        
        self.usr_name = args.usr_name
        self.agent_name = args.agent_name

        # Load MSC Dataset
        data_path = os.path.join(args.data_path, args.data_name)
        with open(data_path, 'r') as f:
            self.dataset = json.load(f)
            if args.test_num > 0:
                self.dataset = self.dataset[:args.test_num]
            self.logger.info(f"Loaded {len(self.dataset)} samples from {data_path}.")
        
        self.relevance_memory_number = args.relevance_memory_number
        self.context_memory_number = args.context_memory_number

    def memory_bank_init(self, sample_id, args):
        """ Initialize the first session and memory graph """
        memory_bank = GraphEnhancedMemory(self.client, sample_id=sample_id, logger=self.logger, args=args, memory_cache=args.memory_cache)
        response_generator = Generator(self.client, logger=self.logger, args=args)
        
        # Process Session 0 (Context Initialization)
        for idx, dial in enumerate(self.dataset[sample_id][0]['dialog']):
            current_time = time.time() + memory_bank.current_time_pass
            # Update Short-term Context
            response_data = {"idx": len(memory_bank.short_term_memory), "time": current_time, "dialog": f"SPEAKER_2: {dial['SPEAKER_2']}"}
            memory_bank.short_term_memory.append(response_data)
            
        # Pre-load sLM triplets for Session 0
        if args.triplet_mode == 'slm':
            triplets = memory_bank._extract_triplets_from_slm(sample_id, 0)
            if triplets:
                session_data = self.dataset[sample_id][0]['dialog']
                context_str = "\n".join([f"{t['SPEAKER_1']}\n{t['SPEAKER_2']}" for t in session_data])
                memory_bank.store_triplets(triplets, time.time(), context=context_str)
                self.logger.info(f"Initialized sLM triplets for Sample {sample_id} Session 0")
    
        return memory_bank, response_generator

    def compute_scores(self, response, reference):
        try:
            metrics_dict = calc_nlg_metrics([response], [reference], "response")
            bleu_score = np.array([metrics_dict['Bleu_1'], metrics_dict['Bleu_2'], metrics_dict['Bleu_3'], metrics_dict['Bleu_4']])
            rl_score = np.array([metrics_dict['ROUGE_L']])
        except:
            bleu_score = np.zeros(4)
            rl_score = np.zeros(1)
            
        dist_score = np.array([self.calculate_dist_n(response, i) for i in [1, 2, 3]])
        return np.concatenate((bleu_score, rl_score, dist_score))

    def calculate_dist_n(self, text, n):
        words = text.split()
        n_grams = list(ngrams(words, n))
        unique_n_grams = len(set(n_grams))
        return unique_n_grams / len(n_grams) if n_grams else 0

    def interative_eval(self, sample_id, session_num, memory_bank, response_generator, args):
        sum_score = np.zeros(8) 
        dialogs = self.dataset[sample_id][session_num]['dialog']
        total_turns = len(dialogs)

        for idx, dial in enumerate(dialogs):
            self.logger.info(f"    >> [S{session_num}] Turn {idx+1}/{total_turns} Processing...")
            
            current_time = time.time() + memory_bank.current_time_pass
            
            # 1. Retrieve
            context_memories = memory_bank.context_retrieve(dial['SPEAKER_1'], n_results=self.context_memory_number, current_time=current_time)
            related_memories, _ = memory_bank.relevance_retrieve(dial['SPEAKER_1'], n_results=self.relevance_memory_number, current_time=current_time)

            # 2. Format Memory String
            summarized_memories = []
            if not related_memories:
                summarized_memories = ["No relevant Memories."]
            else:
                for i, mem in enumerate(related_memories):
                    past_time = convert_seconds_to_full_time(current_time - mem['time'])
                    memory_entry = f"[Rank {i+1}] (Active {past_time} ago) {mem['summary']}"
                    
                    # Add Multi-hop Context (if any)
                    hopped_list = mem.get('hopped_triplets')
                    if hopped_list:
                         extras = [f"{h[0]['head']} {h[0]['relation']} {h[0]['tail']}" for h in hopped_list if isinstance(h, list) and len(h)>=1]
                         if extras:
                             memory_entry += " (Associative Context: " + ", ".join(extras) + ")"
                    summarized_memories.append(memory_entry)

            merged_memory = "\n".join(summarized_memories)
            context_str = "\n".join([f"[TURN {m['idx']}] {m['dialog']}" for m in context_memories])

            # 3. Generate
            response, final_prompt = response_generator.response_build(dial['SPEAKER_1'], context_str, merged_memory)
            
            if response.startswith("SPEAKER_2:"):
                response = response.replace("SPEAKER_2:", "", 1).strip()
            
            # 4. Log
            self.detailed_generation_logs.append({
                "sample_id": sample_id, "session_num": session_num,
                "inquiry": dial['SPEAKER_1'], "response": response, "reference": dial['SPEAKER_2'],
                "full_prompt": final_prompt
            })

            # 5. Score
            score = self.compute_scores(response, dial['SPEAKER_2'])
            sum_score += score
            
            # Update Short-term Context
            memory_bank.short_term_memory.append({"idx": len(memory_bank.short_term_memory), "time": current_time, "dialog": f"SPEAKER_2: {dial['SPEAKER_2']}"})

        return sum_score / (idx + 1), memory_bank, response_generator

    def evaluation(self):
        all_samples_score = []
        total_samples = len(self.dataset)
        
        
        for idx, sample in enumerate(self.dataset):
            self.logger.info(f"========== Progress: [{idx+1}/{total_samples}] Processing Sample ID: {idx} ==========")
            
            
            all_sessions_score = []
            
            # Init Session 0
            memory_bank, response_generator = self.memory_bank_init(idx, self.args)
            memory_bank.current_time_pass += self.dataset[idx][0]['time_pass']
            
            # Eval Next Sessions
            for session_num in range(self.args.min_session, self.args.max_session):
                self.logger.info(f"  > Starting Session {session_num}...")
                
                score, memory_bank, response_generator = self.interative_eval(idx, session_num, memory_bank, response_generator, self.args)
                
                # Store Triplets (Simulating post-session processing)
                if self.args.triplet_mode == 'slm':
                    triplets = memory_bank._extract_triplets_from_slm(idx, session_num)
                    if triplets:
                        session_data = self.dataset[idx][session_num]['dialog']
                        context_str = "\n".join([f"{t['SPEAKER_1']}\n{t['SPEAKER_2']}" for t in session_data])
                        memory_bank.store_triplets(triplets, time.time(), context=context_str)
                        
                        self.logger.info(f"  > [End S{session_num}] Stored {len(triplets)} triplets.")
                
                memory_bank.current_time_pass += self.dataset[idx][session_num]['time_pass']
                all_sessions_score.append(score)
            
            avg_score = sum(all_sessions_score) / len(all_sessions_score)
            
            score_log = (
                f"B-1: {avg_score[0]:.4f}\nB-2: {avg_score[1]:.4f}\nB-3: {avg_score[2]:.4f}\nB-4: {avg_score[3]:.4f}\n"
                f"R-L: {avg_score[4]:.4f}"
            )
            
            self.logger.info("-" * 80)
            self.logger.info(f"✅ Sample {idx} Finished.")
            self.logger.info(f"📊 Avg Score: \n{score_log}")
            self.logger.info("-" * 80)
            
            all_samples_score.append(all_sessions_score)
            
            if (idx + 1) % self.args.log_step == 0:
                self.logger.info(f"Saving intermediate logs...")
                with open(self.detailed_log_file, "w", encoding='utf-8') as f:
                    
                    json.dump(self.detailed_generation_logs, f, indent=4, ensure_ascii=False)
            
            
            del memory_bank, response_generator
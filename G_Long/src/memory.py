import time
import math
import json
import chromadb
import networkx as nx
import numpy as np
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

class GraphEnhancedMemory():
    """
    Manages the structured memory graph and vector storage.
    Renamed from 'EventMemory' to reflect the G-Long methodology.
    """
    def __init__(self, client, sample_id, logger, args, memory_cache=None):
        self.args = args
        self.logger = logger
        self.embedding_function = DefaultEmbeddingFunction()
        self.data_loader = ImageLoader()
        
        # Initialize ChromaDB (Vector Store)
        try:
            if memory_cache:
                self.dbclient = chromadb.PersistentClient(path=memory_cache)
            else:
                self.dbclient = chromadb.Client()
        except Exception as e:
            self.logger.error(f"ChromaDB init failed: {e}")

        self.LLMclient = client
        self.current_time_pass = 0
        self.short_term_memory = []

        # Collection for Node Retrieval (Anchoring)
        self.collection = self.dbclient.get_or_create_collection(
            name=f"collection_{sample_id}", 
            embedding_function=self.embedding_function, 
            data_loader=self.data_loader
        )

        # NetworkX MultiDiGraph for Relation Storage
        self.graph = nx.MultiDiGraph()
        
        # Load pre-extracted triplets (sLM)
        self.slm_triplets_data = {}
        if args.triplet_mode == 'slm':
            self._load_slm_data(args.slm_data_path)

    def _load_slm_data(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                for item in raw_data:
                    # Key mapping: (dialogue_id, session_id)
                    key = (item.get('dialogue_id'), item.get('session_id'))
                    triplets = []
                    if 'dialogue_triplets' in item:
                        for dt in item['dialogue_triplets']:
                            if 'triple' in dt and len(dt['triple']) > 0:
                                t_obj = dt['triple'][0]
                                triplets.append({
                                    'head': t_obj.get('head'),
                                    'relation': t_obj.get('relation'),
                                    'tail': t_obj.get('tail'),
                                    'raw_content': dt.get('text', ""),
                                    'importance_score': float(dt.get('importance_score')) / 10.0,
                                    'hopped_triplets': [] # For multi-hop expansion
                                })
                    self.slm_triplets_data[key] = triplets
            self.logger.info(f"Loaded sLM triplets for {len(self.slm_triplets_data)} sessions.")
        except Exception as e:
            self.logger.error(f"Failed to load sLM data: {e}")

    def _extract_triplets_from_slm(self, sample_id, session_num):
        key = (sample_id, session_num)
        return self.slm_triplets_data.get(key, [])

    def store_triplets(self, triplets, current_time, context):
        """ Store extracted triplets into Graph and Vector DB """
        for item in triplets:
            head = item.get('head')
            relation = item.get('relation')
            tail = item.get('tail')
            
            if not head or not tail: continue
            
            full_text = f"{head} {relation} {tail}"
            importance = item.get('importance_score')

            # 1. Add Anchor Nodes to ChromaDB
            for entity in [head, tail]:
                # Check existence to avoid duplication overhead
                existing = self.collection.get(ids=[entity])
                if not existing['ids']:
                    self.collection.add(ids=[entity], documents=[entity], metadatas=[{"type": "node"}])

            # 2. Add Edges to Knowledge Graph
            self.graph.add_edge(
                head, tail, 
                relation=relation,
                time=current_time,
                importance_score=importance,
                last_accessed_time=current_time,
                full_text=full_text,
                hopped_triplets=json.dumps(item.get('hopped_triplets', [])),
                raw_content=item.get('raw_content', "")
            )

    def context_retrieve(self, query, n_results=10, current_time=0):
        # Update short term memory for context tracking
        data = {"idx": len(self.short_term_memory), "time": current_time, "dialog": f"User: {query}"}
        self.short_term_memory.append(data)
        return self.short_term_memory 

    def relevance_retrieve(self, query, n_results=3, current_time=0):
        """ 
        G-Long Retrieval Mechanism:
        Vector Anchor Search -> Subgraph Expansion -> Semantic Filtering -> Re-ranking
        """
        
        # 1. Anchor Node Search (Vector)
        try:
            results = self.collection.query(query_texts=[query], n_results=5)
        except:
            return [], []

        candidate_triplets = []
        visited_edges = set()

        # 2. Subgraph Expansion (Retrieving associated edges)
        if results['ids']:
            anchor_nodes = results['ids'][0]
            for anchor in anchor_nodes:
                if not self.graph.has_node(anchor): continue
                
                # Get both outgoing and incoming edges
                edges = list(self.graph.out_edges(anchor, data=True)) + list(self.graph.in_edges(anchor, data=True))
                for u, v, data in edges:
                    edge_id = (u, v, data['relation'])
                    if edge_id in visited_edges: continue
                    
                    hopped_data = []
                    if 'hopped_triplets' in data:
                        try: hopped_data = json.loads(data['hopped_triplets'])
                        except: pass
                            
                    candidate_triplets.append({
                        'head': u, 'tail': v, 'relation': data['relation'],
                        'summary': data['full_text'],
                        'importance_score': data['importance_score'],
                        'last_accessed_time': data['last_accessed_time'],
                        'time': data['time'],
                        'hopped_triplets': hopped_data,
                    })
                    visited_edges.add(edge_id)

        if not candidate_triplets: return [], []

        # 3. Semantic Filtering
        try:
            query_emb = self.embedding_function([query])[0]
            cand_texts = [c['summary'] for c in candidate_triplets]
            cand_embs = self.embedding_function(cand_texts)
            
            for i, cand in enumerate(candidate_triplets):
                # Cosine Similarity
                score = np.dot(query_emb, cand_embs[i]) / (np.linalg.norm(query_emb) * np.linalg.norm(cand_embs[i]))
                cand['score_semantic'] = max(0.0, float(score))
        except:
            return [], []
            
        # Pre-filter Top-K based on semantic relevance
        candidate_triplets.sort(key=lambda x: x['score_semantic'], reverse=True)
        filtered_candidates = candidate_triplets[:n_results+2]

        # 4. Multi-factor Re-ranking (Semantic + Importance + Recency)
        W_SEM, W_IMP, W_REC = 0.5, 0.3, 0.2
        ranked_candidates = []
        
        for cand in filtered_candidates:
            time_gap = current_time - cand['last_accessed_time']
            s_rec = math.exp(-1E-7 * time_gap) # Recency decay
            overall = (W_SEM * cand['score_semantic']) + (W_IMP * cand['importance_score']) + (W_REC * s_rec)
            cand['score_overall'] = overall
            ranked_candidates.append(cand)
            
        ranked_candidates.sort(key=lambda x: x['score_overall'], reverse=True)
        final_results = ranked_candidates[:n_results]
        
        self.logger.info(f"[Step 3] Retrieved {len(final_results)} memories.")
        
        # Update 'Last Accessed Time' for retrieved edges
        for item in final_results:
            try:
                u, v, rel = item['head'], item['tail'], item['relation']
                # Naive update for matching edges
                if self.graph.has_edge(u, v):
                    for key, attr in self.graph[u][v].items():
                        if attr['relation'] == rel:
                            self.graph[u][v][key]['last_accessed_time'] = current_time
            except: pass

        return final_results, []
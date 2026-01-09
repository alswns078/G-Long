
# G-Long: Graph-Enhanced Long-Term Memory for Dialogue Systems

This is the official anonymous repository for the paper **"G-Long: Graph-Enhanced Long-Term Memory for Dialogue Systems"**.

## 🚀 Overview
G-Long introduces a structured graph memory framework to overcome the limitations of unstructured based approach in long-term conversations. This repository contains the implementation for the **MSC (Multi-Session Chat)** dataset experiments.
( * The code for **CC (Conversation Chronicles)** and **LME(LongMemEval)** is undergoing final refactoring and will be released in the next update.)

## 🛠️ Requirements
* python >= 3.8
* torch
* chromadb
* networkx
* openai
* nlgeval

Install dependencies via:
```bash
pip install -r requirements.txt
```
## 🧩 Triplet Extraction Pipeline (sLM)

Before running the main evaluation, we employ a fine-tuned sLM to extract and score knowledge triplets. The pipeline follows a sequential process: **Training $\rightarrow$ Inference $\rightarrow$ Scoring**.

**Pipeline Steps:**
1.  **Configure:** Set the model and save paths directly inside `triplet_extractor/train.py`.
2.  **Train:** Run the training script to generate the LoRA adapter (hyperparameters follow the paper's settings).
3.  **Inference:** Update the model and adapter paths in `triplet_extractor/inference.py` to extract triplets from the text into a JSON file.
4.  **Scoring:** Specify the input JSON path in `triplet_extractor/attention_scorer.py` and execute it to calculate/append attention scores for each triplet.

### Example Commands
> **Note:** Please ensure to modify the model paths and file paths hard-coded in each script before execution.

```bash
# 1. Train: Fine-tune the model to get the LoRA adapter
python triplet_extractor/train.py

# 2. Inference: Extract triplets using the trained adapter
python triplet_extractor/inference.py

# 3. Scoring: Calculate attention scores for the extracted triplets
python triplet_extractor/attention_scorer.py
```

## 🏃 How to Run

1. Place your OpenAI API Key in `run_msc.sh` or set it via environment variable.
2. Ensure the MSC dataset samples are in `data/MSC/`.
3. Run the evaluation script:

```bash
bash run_msc.sh

```

## 🙏 Acknowledgements

This project is built upon the foundational framework provided by **LD-Agent** (Li et al., 2025). We adapted their modular architecture to integrate our **Graph-Enhanced Memory** and **sLM-based Triplet Extraction** mechanisms. We strictly followed their prompt configurations to ensure a fair comparison. We thank the authors for their open-source contribution.




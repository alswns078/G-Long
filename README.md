네, 1저자님! 앞서 정리해드린 내용을 바탕으로, **바로 복사해서 사용하실 수 있는 `README.md` 마크다운 원본**입니다.

이 내용을 `README.md` 파일에 그대로 저장하시면 됩니다.

```markdown
# G-Long: Graph-Enhanced Long-Term Memory for Dialogue Systems

This is the official anonymous repository for the paper **"G-Long: Graph-Enhanced Long-Term Memory for Dialogue Systems"**.

## 🚀 Overview
G-Long introduces a structured graph memory framework to overcome the limitations of vector-based retrieval in long-term conversations. This repository contains the implementation for the **MSC (Multi-Session Chat)** dataset experiments.

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

## 🏃 How to Run

1. Place your OpenAI API Key in `run_msc.sh` or set it via environment variable.
2. Ensure the MSC dataset samples are in `data/MSC/`.
3. Run the evaluation script:

```bash
bash run_msc.sh

```

## 🙏 Acknowledgements

This project is built upon the foundational framework provided by **LD-Agent** (Li et al., 2025). We adapted their modular architecture to integrate our **Graph-Enhanced Memory** and **sLM-based Triplet Extraction** mechanisms. We strictly followed their prompt configurations to ensure a fair comparison. We thank the authors for their open-source contribution.

```

```

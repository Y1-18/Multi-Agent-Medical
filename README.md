# Multi-Agent Medical System

This repository presents a **Multi-Agent Medical Assistant** designed for research and educational purposes.  
The system integrates medical image analysis, retrieval-augmented generation (RAG), and real-time medical web search within a modular, agent-based architecture.

---

## Overview

The system routes user inputs (text or medical images) to specialized agents:

- Image Analysis Agent for medical image inference
- RAG Agent for evidence-based knowledge retrieval
- Web Search Agent for up-to-date medical research
- Guardrails Agent for safety, validation, and hallucination control

A central Decision Agent coordinates agent selection based on user intent and input modality.

---

## System Architecture


---

## Project Structure

```text
.
├── agents/
│   ├── image_analysis_agent/
│   ├── rag_agent/
│   ├── web_search_processor_agent/
│   ├── guardrails/
│   └── agent.py
├── data/
├── uploads/
├── app.py
├── main.py
├── rag_data.py
├── requirements.txt
└── Docker
```

---
## Medical Image Analysis

The system supports medical image inference tasks including:

- Skin lesion classification (Swin Transformer)

- Chest X-ray disease detection (Swin Transformer)

- Brain tumor classification from MRI (Vision Transformer)

-Models are implemented using Hugging Face Transformers and PyTorch.
---

---
## Retrieval-Augmented Generation (RAG)

- RAG is implemented using a Qdrant vector database and Azure OpenAI embeddings. Responses are generated only when sufficient, relevant medical context is retrieved, ensuring grounded and reliable outputs.
---

---
## Safety and Guardrails

Safety is enforced through multiple layers:

- Human-in-the-loop validation for critical or low-confidence outputs

- Confidence thresholds for document retrieval

- Source-grounded generation

-Refusal or fallback behavior when evidence is insufficient

The system is designed to minimize hallucinations and prevent unsafe medical conclusions
---
---
##Installation
```bash

git clone https://github.com/Y1-18/Multi-Agent-Medical.git
cd Multi-Agent-Medical
pip install -r requirements.txt
python main.py
python app.py
```
---

---
## Disclaimer

- This system is for research and educational purposes only.
---

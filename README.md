# Reducing Hallucinations in Arabic Legal LLMs Using RAG and Explainable AI

> **Graduation Project — Department of Computer Science and Artificial Intelligence, University of Jeddah**
>
> Alaa Almunyah · Aseel Almehmadi · Fatima Almaashi · Hadeel Alabdulhadi · Shaden Alamri
>
> Supervisor: Dr. Safa Alsafari

---

## Overview

This repository contains the full implementation of a two-phase research project that addresses hallucination in Arabic Legal Large Language Models (LLMs).

- **Phase I** *(prior work)*: Established a dedicated hallucination benchmark for Arabic legal QA — revealing a hallucination score of **0.810** and a pass rate of only **15.2%** among leading multilingual LLMs operating without retrieved legal context.
- **Phase II** *(this repo)*: Designs and evaluates a **RAG + XAI framework** that reduces hallucination and improves transparency in Arabic legal question answering.

### Key Results (Phase II)

| Metric | Phase I (Baseline) | Phase II (RAG + XAI) | Δ |
|---|---|---|---|
| Hallucination Score ↓ | 0.810 | 0.097 | −88.0% |
| Faithfulness Score ↑ | 0.818 | 0.955 | +16.7% |
| Answer Relevancy ↑ | 0.891 | 0.905 | +1.6% |
| Hallucination Pass Rate ↑ | 15.2% | 89.9% | +74.7 pp |
| Faithfulness Pass Rate ↑ | 92.4% | 100.0% | +7.6 pp |
| Claim Recall | — | 81.6% | — |
| Context Precision | — | 83.8% | — |

---

## Repository Structure

```
arabic-legal-llm/
│
├── xai/
│   ├── perturbation_based_xai/       # RAG-Ex XAI framework (adapted for Arabic legal)
│   │   ├── ragex_framework/          # Core framework package
│   │   │   ├── explainer/            # GenericExplainer, GenericGeneratorExplainer
│   │   │   └── modules/
│   │   │       ├── comparator/       # LegalHybridComparator, EmbeddingComparator, etc.
│   │   │       ├── tokenizer/        # ArabicLegalTokenizer
│   │   │       └── perturber/        # LOO, LLM-based, ReOrder perturbers
│   │   ├── analysis/                 # RAG-Ex output files and figures
│   │   │   ├── extremes/             # Per-strategy extreme-case JSONs
│   │   │   └── filtered_tokenization/# Filtered tokenization outputs
│   │   ├── XAI_v5.ipynb              # Main XAI analysis notebook
│   │   ├── generated_answers.jsonl   # Answers used for XAI analysis
│   │   └── LOO_HUMAN_EVAL.html       # Leave-One-Out human evaluation view
│   │
│   └── post_hoc_verification/        # Claim-level post-hoc verification pipeline
│       └── post_hoc_ver.py           # 4-stage: extract → rank → verify → aggregate
│
├── rag/                              # Hybrid RAG pipeline
│   ├── retriever.py                  # BM25 + BGE-M3 + RRF + cross-encoder reranker
│   ├── generator.py                  # DeepSeek-Chat answer generation
│   ├── llm.py                        # LLM client wrapper
│   ├── RAGChecker.py                 # RAGChecker claim-recall & context-precision eval
│   ├── reconvert_RAGChecker.py       # RAGChecker format conversion utility
│   ├── ragchecker_input.json         # RAGChecker evaluation input
│   ├── ragchecker_output.json        # RAGChecker evaluation results
│   ├── najiz_top_sources.jsonl       # Retrieved top-k sources per query
│   └── generated_answers.jsonl       # RAG-generated answers
│
├── deepeval/                         # DeepEval evaluation pipeline
│   ├── evaluate.py                   # Phase II evaluation (with RAG context)
│   ├── evaluate_without_context.py   # Phase I evaluation (no context baseline)
│   ├── generator.py                  # Answer generator (with context)
│   ├── generator_without_context.py  # Answer generator (without context)
│   ├── judge.py                      # GPT-4o judge configuration
│   ├── llm.py                        # LLM client wrapper
│   ├── deepeval_results.jsonl        # Phase II DeepEval results
│   ├── without_context_deepeval_results.jsonl  # Phase I baseline results
│   ├── generated_answers.jsonl       # Phase II generated answers
│   ├── without_context_generated_answers.jsonl # Phase I baseline answers
│   ├── najiz_records.jsonl           # Najiz benchmark questions
│   └── najiz_top_sources.jsonl       # Top sources for evaluation
│
├── interface/
│   └── index.html                    # Results web interface
│                                     # Live: https://hdla22.github.io/Legal-Results-Interface/
│
├── requirements.txt
└── README.md
```

---

## Pipeline Architecture

### Phase II: RAG + XAI Framework

```
User Query
    │
    ▼
[Hybrid Retrieval]
  BM25 (lexical) + BGE-M3 (semantic)
    │
    ▼
[RRF Fusion + Cross-Encoder Reranking]
  bge-reranker-v2-m3
    │
    ▼
[Dynamic Top-k Selection + Deduplication]
  Score-driven filtering (k_max = 3)
    │
    ▼
[Answer Generation]
  DeepSeek-Chat (context-grounded prompt)
    │
    ├──▶ [Post-Hoc Claim-Level Verification]
    │      Extract claims → Rank evidence → Verify → Aggregate
    │
    └──▶ [Perturbation-Based XAI (RAG-Ex)]
           Perturb → Generate → Compare → Importance scores
```

### Knowledge Base
- **BOE** (Bureau of Experts): 507 documents → 16,469 article-level chunks (primary source)
- **MOJ** (Ministry of Justice): 71 documents → 6,120 article-level chunks (complementary)

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export OPENAI_API_KEY="your-gpt4o-key"       # For DeepEval judge
export DEEPSEEK_API_KEY="your-deepseek-key"   # For generation & XAI
```

### 3. Run RAG Pipeline
```bash
cd rag
python generator.py
```

### 4. Evaluate with DeepEval
```bash
cd deepeval
python evaluate.py                      # Phase II (with context)
python evaluate_without_context.py      # Phase I baseline
```

### 5. Run Post-Hoc Verification
```bash
cd xai/post_hoc_verification
python post_hoc_ver.py
```

### 6. Run XAI Analysis
Open `xai/perturbation_based_xai/XAI_v5.ipynb` in Jupyter.

---

## Corpus Sources
- **Bureau of Experts (BOE)**: https://www.boe.gov.sa
- **Ministry of Justice (MOJ)**: https://www.moj.gov.sa

---

## Results Interface
A live web interface displaying generated answers, retrieved sources, claim-level verification results, and XAI explanations is available at:

**https://hdla22.github.io/Legal-Results-Interface/**

---

## Paper & Code
- Full report: included in this repo (`final_report.pdf`)
- Phase I benchmark: https://github.com/hdla22/Arabic-Legal-LLM-Benchmark
- Phase I paper: https://throbbing-dew-9.linkyhost.com

---

## Citation
```
Almunyah, A., Almehmadi, A., Almaashi, F., Alabdulhadi, H., & Alamri, S. (2026).
Reducing Hallucinations in Arabic Legal LLMs Using Retrieval-Augmented Generation
and Explainable AI. University of Jeddah.
```

---

## Acknowledgements
We thank the University of Jeddah and the Department of Computer Science and Artificial Intelligence for their support. Special thanks to our supervisor **Dr. Safa Alsafari**, and to **Mr. Mazen Shuaib** for his valuable legal review and insights.

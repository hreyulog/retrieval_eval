# Retrieval Evaluation Toolkit Guide

This guide walks you through installing the RAG Evaluation Toolkit, preparing your first dataset, and running your first evaluation.

## Prerequisites

The RAG Evaluation Toolkit requires:

| Requirement | Version | Purpose |
|------------|---------|---------|
| Python | ≥ 3.9 | Core runtime environment |
| pip | Latest | Package installation |

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/hreyulog/retrieval_eval
cd retrieval_eval
```

### Step 2: Install the Package

```bash
pip install -e .
```

### Step 2.1: For arkts dataset you can run

```python
python download_arkts_ds.py
```


### Step 3: Run evaluation

```bash
evaluator eval-retrieval --dataset ./arktscodesearch --embedding_model google/embeddinggemma-300m
```

---

# Dataset Preparation

Required directory structure:

```
your_dataset/
├── queries.csv
├── index.jsonl
└── annotations.jsonl
```

## queries.csv example

```csv
query_id,query
1,"How to implement binary search in Python?"
2,"What is the difference between list and tuple?"
```

## index.jsonl example

```json
{"doc_id": "doc_001", "doc": "def binary_search(arr, target): ..."}
{"doc_id": "doc_002", "doc": "# Lists are mutable, tuples are immutable ..."}
```

## annotations.jsonl example

```json
{"query_id": "1", "doc_id2rels": {"doc_001": 1.0, "doc_003": 0.5}}
{"query_id": "2", "doc_id2rels": {"doc_002": 1.0}}
```

---

# Running a Retrieval Evaluation

Basic command:

```bash
evaluator eval_retrieval \
  --dataset path/to/your_dataset \
  --embedding_model model_identifier \
  --top_k 5
```



# Output Example

## Console output

```
INFO: recall@5: 0.8234
INFO: precision@5: 0.6543
INFO: mrr: 0.7891
INFO: ndcg@5: 0.8012
INFO: f1@5: 0.7289
```

## JSON output

```json
{
  "recall@5": 0.8234,
  "precision@5": 0.6543,
  "mrr": 0.7891,
  "ndcg@5": 0.8012,
  "f1@5": 0.7289,
  "retrieval_time_sec": 3.2456
}
```

---

# Supported Encoders

- sentence-transformers/all-MiniLM-L6-v2
- automodel:bert-base-uncased:mean
- api:text-embedding-ada-002
- clip:openai/clip-vit-base-patch32

---

## Citation / Preprint


```bibtex
@misc{he2026arktscodesearchopensourcearktsdataset,
      title={ArkTS-CodeSearch: A Open-Source ArkTS Dataset for Code Retrieval}, 
      author={Yulong He and Artem Ermakov and Sergey Kovalchuk and Artem Aliev and Dmitry Shalymov},
      year={2026},
      eprint={2602.05550},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2602.05550}, 
}



## Pyserini BM25 backend

This toolkit can optionally run retrieval with **Pyserini BM25** (Lucene) instead of dense embeddings.

1) Install Pyserini:

```bash
pip install pyserini
```

2) Build (or download) a Pyserini Lucene index for your corpus.

3) Run retrieval evaluation with the Pyserini backend:

```bash
evaluator eval-retrieval   --dataset <DATASET_PATH_OR_NAME>   --embedding_model <IGNORED_FOR_BM25>   --top_k 5   --backend pyserini_bm25   --pyserini_index <PATH_TO_LUCENE_INDEX>
```

Notes:
- The Lucene index document IDs must match the dataset doc IDs used during evaluation.
- `--embedding_model` is still required by the CLI signature but is not used when `--backend pyserini_bm25`.

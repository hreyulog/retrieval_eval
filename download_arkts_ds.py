import json
import csv
from datasets import load_dataset
import os
# =========================
# Hugging Face dataset
# =========================

DATASET_NAME = "hreyulog/arkts-code-docstring"
SPLIT = "test"   # train / validation / test

dataset = load_dataset(DATASET_NAME, split=SPLIT)

# =========================
# 输出文件
# =========================

queries_path = "queries.csv"
annotations_path = "annotations.jsonl"
index_path = "index.jsonl"

queries = []
annotations = []
index_docs = []

# =========================
# 构建数据
# =========================

for i, data in enumerate(dataset):
    docstring = (data.get("docstring") or "").strip()
    code = (data.get("function") or "").strip()

    if not docstring or not code:
        continue

    query_id = i
    doc_id = i

    # ---- queries.csv ----
    queries.append({
        "query": docstring,
        "query_id": query_id
    })

    # ---- annotations.jsonl ----
    annotations.append({
        "query_id": query_id,
        "doc_id2rels": {
            str(doc_id): 1.0
        }
    })

    # ---- index.jsonl ----
    index_docs.append({
        "doc": code,
        "doc_id": doc_id
    })

# =========================
# 写文件
# =========================
OUTPUT_DIR = "arktscodesearch"
os.makedirs(OUTPUT_DIR, exist_ok=True)

queries_path = os.path.join(OUTPUT_DIR, "queries.csv")
annotations_path = os.path.join(OUTPUT_DIR, "annotations.jsonl")
index_path = os.path.join(OUTPUT_DIR, "index.jsonl")

# 1. queries.csv
with open(queries_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["query", "query_id"])
    writer.writeheader()
    writer.writerows(queries)

# 2. annotations.jsonl
with open(annotations_path, "w", encoding="utf-8") as f:
    for item in annotations:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 3. index.jsonl
with open(index_path, "w", encoding="utf-8") as f:
    for item in index_docs:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Done! Generated queries / annotations / index from Hugging Face dataset.")

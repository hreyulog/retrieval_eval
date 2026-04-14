"""
embedding_model_pretrain.py
Fine-tune embedding models for code retrieval tasks.
"""

import json
import random
from pathlib import Path
from typing import List

import typer
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import numpy as np

app = typer.Typer()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def load_jsonl(path: str) -> List[dict]:
    """Load JSONL file, one JSON object per line."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError as e:
                print(f"Skipped invalid JSON line: {line}\nError: {e}")
    return records


def build_examples_from_records(records: List[dict]) -> List[InputExample]:
    """
    Convert each record to InputExample pairs.
    Uses docstring as query and function as positive example.
    Bidirectional pairs are added to improve symmetric matching.
    """
    examples = []
    for r in records:
        caption = r.get("docstring", "").strip()
        desc = r.get("function", "").strip()
        if not caption or not desc:
            continue
        examples.append(InputExample(texts=[caption, desc]))
        examples.append(InputExample(texts=[desc, caption]))
    return examples

def prepare_sentence_transformer(model_name_or_path: str, max_seq_length: int, device: str):
    model = SentenceTransformer(model_name_or_path, device=device)
    model = model.to(device)
    for child in model._modules.values():
        child.to(device)
    try:
        model.max_seq_length = max_seq_length
    except Exception:
        pass
    return model

def train_model(model, train_examples: List[InputExample], output_dir: str, batch_size: int, epochs: int, lr: float):
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = max(100, int(len(train_dataloader) * epochs * 0.1))
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        steps_per_epoch=None,
        optimizer_params={'lr': lr},
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=output_dir
    )



@app.command()
def main(
    train_json: str = typer.Option(..., "--train-json", "-t", help="Path to training JSONL file"),
    model_name: str = typer.Option("BAAI/bge-base-en-v1.5", "--model", "-m", help="Model name or path"),
    output_dir: str = typer.Option("output_model", "--output", "-o", help="Output directory for fine-tuned model"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Batch size for training"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    lr: float = typer.Option(2e-5, "--lr", help="Learning rate"),
    max_seq_length: int = typer.Option(256, "--max-seq-length", help="Maximum sequence length"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device to use (cuda/cpu)"),
):
    if not Path(train_json).exists():
        raise FileNotFoundError(f"Training file {train_json} not found.")
    records = load_jsonl(train_json)
    print(f"Loaded {len(records)} records from {train_json}.")

    train_examples = build_examples_from_records(records)
    print(f"Built {len(train_examples)} training examples (pairs).")

    print("Loading model...")
    model = prepare_sentence_transformer(model_name, max_seq_length, device)
    print("Model loaded.")

    print("Start finetuning...")
    train_model(model, train_examples, output_dir, batch_size, epochs, lr)
    print("Finetuning finished. Model saved to", output_dir)

    model.save(output_dir)
    print("Saved final model to", output_dir)


if __name__ == "__main__":
    app()

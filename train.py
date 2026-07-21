"""
Training entry point.

Usage (Colab or terminal, after `pip install -r requirements.txt`):
    python train.py --config configs/concat_fusion.yaml
    python train.py --config configs/cross_attn_fusion.yaml

Run both configs to produce the comparison numbers for the novelty
evaluation (see evaluate.py).
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from models.fusion import VQAFusionModel
from data.clevr_dataset import CLEVRVQADataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_tokenizer(data_root: str):
    """
    Builds (or loads) the CLEVR word-level tokenizer. Vocab is learned
    from the training split's questions + answers, then reused for val
    so token ids stay consistent across splits.
    """
    from data.tokenizer import CLEVRTokenizer, build_tokenizer_from_clevr

    vocab_path = Path(data_root) / "vocab.json"
    train_questions_path = Path(data_root) / "questions" / "train_questions.json"

    tokenizer = CLEVRTokenizer()
    if vocab_path.exists():
        tokenizer.load(str(vocab_path))
    else:
        tokenizer = build_tokenizer_from_clevr(str(train_questions_path), save_path=str(vocab_path))

    return tokenizer


def train(config_path: str):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = get_tokenizer(cfg["data_root"])
    cfg["vocab_size"] = tokenizer.vocab_size  # override config value with actual learned vocab size

    train_ds = CLEVRVQADataset(
        cfg["data_root"], "train", tokenizer, image_size=cfg["image_size"], max_len=cfg["max_seq_len"]
    )
    val_ds = CLEVRVQADataset(
        cfg["data_root"], "val", tokenizer, image_size=cfg["image_size"], max_len=cfg["max_seq_len"]
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    model = VQAFusionModel(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        max_seq_len=cfg["max_seq_len"],
        fusion=cfg["fusion"],
        vision_backbone=cfg["vision_backbone"],
        freeze_vision=cfg["freeze_vision"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"]
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            question_ids = batch["question_ids"].to(device)
            answer_ids = batch["answer_ids"].to(device)

            logits = model(images, question_ids)
            # align logits/answer length; adjust slicing to your tokenization scheme
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), answer_ids.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                question_ids = batch["question_ids"].to(device)
                answer_ids = batch["answer_ids"].to(device)
                logits = model(images, question_ids)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), answer_ids.reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        print(f"Epoch {epoch+1}/{cfg['epochs']} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")

    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Done. Model + history saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)

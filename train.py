"""
Training entry point.

Usage:
    python train.py --config configs/concat_fusion.yaml
    python train.py --config configs/cross_attn_fusion.yaml
    python train.py --config configs/no_vision_baseline.yaml

Uses differential learning rates: any unfrozen pretrained vision layers
(e.g. ResNet's layer4, when freeze_vision="partial") get a smaller LR
than the rest of the model. This is standard fine-tuning practice —
pretrained weights already encode useful structure, so they should be
nudged gently rather than updated at the same aggressive rate as
randomly-initialized layers (the decoder, projection, etc.), which need
larger updates to learn from scratch. Using one shared LR for both risks
either being too slow for the decoder or too disruptive for the
pretrained vision features.
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
    from data.tokenizer import CLEVRTokenizer, build_tokenizer_from_clevr

    vocab_path = Path(data_root) / "vocab.json"
    train_questions_path = Path(data_root) / "questions" / "train_questions.json"

    tokenizer = CLEVRTokenizer()
    if vocab_path.exists():
        tokenizer.load(str(vocab_path))
    else:
        tokenizer = build_tokenizer_from_clevr(str(train_questions_path), save_path=str(vocab_path))

    return tokenizer


def build_optimizer(model: nn.Module, base_lr: float, vision_lr_scale: float = 0.1):
    """
    Splits parameters into two groups:
      - unfrozen pretrained vision layers (e.g. ResNet layer4): base_lr * vision_lr_scale
      - everything else (decoder, projection, embeddings): base_lr

    vision_lr_scale=0.1 means the vision fine-tuning LR is 10x smaller
    than the rest of the model's LR — a common starting point for
    fine-tuning pretrained backbones.
    """
    vision_params = []
    other_params = []

    vision_encoder = getattr(model, "vision_encoder", None)
    vision_param_ids = set()
    if vision_encoder is not None:
        vision_param_ids = {id(p) for p in vision_encoder.parameters()}

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in vision_param_ids:
            vision_params.append(p)
        else:
            other_params.append(p)

    param_groups = [{"params": other_params, "lr": base_lr}]
    if vision_params:
        param_groups.append({"params": vision_params, "lr": base_lr * vision_lr_scale})
        print(f"Optimizer: {len(other_params)} params @ lr={base_lr}, "
              f"{len(vision_params)} vision params @ lr={base_lr * vision_lr_scale}")
    else:
        print(f"Optimizer: {len(other_params)} params @ lr={base_lr} (no trainable vision params)")

    return torch.optim.AdamW(param_groups)


def train(config_path: str):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = get_tokenizer(cfg["data_root"])
    cfg["vocab_size"] = tokenizer.vocab_size

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
        dropout=cfg.get("dropout", 0.1),
    ).to(device)

    optimizer = build_optimizer(model, base_lr=cfg["lr"], vision_lr_scale=cfg.get("vision_lr_scale", 0.1))
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            question_ids = batch["question_ids"].to(device)
            answer_ids = batch["answer_ids"].to(device)

            logits = model(images, question_ids)
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), out_dir / "model_best.pt")

    torch.save(model.state_dict(), out_dir / "model.pt")
    history["best_val_loss"] = best_val_loss
    history["best_epoch"] = best_epoch
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Done. Final model saved to {out_dir}/model.pt")
    print(f"Best model (val_loss={best_val_loss:.4f} at epoch {best_epoch}) saved to {out_dir}/model_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)

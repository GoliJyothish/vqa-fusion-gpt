"""
Runs accuracy evaluation for a trained model and, if both fusion variants
have been trained, produces the concat-vs-cross-attention comparison table
that is the core novelty result of this project.

Usage:
    python evaluate.py --config configs/concat_fusion.yaml
    python evaluate.py --compare configs/concat_fusion.yaml configs/cross_attn_fusion.yaml
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.fusion import VQAFusionModel
from data.clevr_dataset import CLEVRVQADataset
from train import load_config, get_tokenizer


@torch.no_grad()
def evaluate_accuracy(cfg: dict, device: torch.device) -> float:
    tokenizer = get_tokenizer(cfg["data_root"])
    cfg["vocab_size"] = tokenizer.vocab_size
    val_ds = CLEVRVQADataset(
        cfg["data_root"], "val", tokenizer, image_size=cfg["image_size"], max_len=cfg["max_seq_len"]
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

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
    model.load_state_dict(torch.load(Path(cfg["output_dir"]) / "model.pt", map_location=device))
    model.eval()

    correct, total = 0, 0
    for batch in val_loader:
        images = batch["image"].to(device)
        question_ids = batch["question_ids"].to(device)
        answer_ids = batch["answer_ids"].to(device)

        logits = model(images, question_ids)
        preds = logits.argmax(dim=-1)

        mask = answer_ids != tokenizer.pad_id
        correct += ((preds == answer_ids) & mask).sum().item()
        total += mask.sum().item()

    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="single config to evaluate")
    parser.add_argument("--compare", nargs=2, metavar=("CONFIG_A", "CONFIG_B"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.compare:
        results = {}
        for path in args.compare:
            cfg = load_config(path)
            acc = evaluate_accuracy(cfg, device)
            results[cfg["fusion"]] = acc
            print(f"{cfg['fusion']}: accuracy = {acc:.4f}")

        Path("results").mkdir(exist_ok=True)
        with open("results/comparison.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nComparison table saved to results/comparison.json")

    elif args.config:
        cfg = load_config(args.config)
        acc = evaluate_accuracy(cfg, device)
        print(f"{cfg['fusion']}: accuracy = {acc:.4f}")

    else:
        parser.error("Provide --config or --compare")


if __name__ == "__main__":
    main()

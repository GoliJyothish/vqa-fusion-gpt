"""
Breaks down accuracy by question type (counting, color, yes/no, etc.) and
generates qualitative examples (image + question + predicted vs true answer)
for both fusion strategies. Produces the kind of detailed analysis that
goes well beyond a single overall accuracy number.

Usage:
    python analyze_results.py --compare configs/concat_fusion.yaml configs/cross_attn_fusion.yaml
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.fusion import VQAFusionModel
from data.clevr_dataset import CLEVRVQADataset
from train import load_config, get_tokenizer


def classify_question(question: str) -> str:
    """
    Lightweight rule-based question-type classifier based on CLEVR's
    common question phrasings. Not as precise as using the ground-truth
    'program' field, but requires no extra parsing and is transparent
    to explain in a report.
    """
    q = question.lower().strip()
    if q.startswith("how many"):
        return "counting"
    if q.startswith("is there") or q.startswith("are there"):
        return "existence"
    if re.search(r"\b(bigger|smaller|larger|same size|same shape|same color|same material)\b", q):
        return "comparison"
    if q.startswith("what color"):
        return "color"
    if q.startswith("what shape") or q.startswith("what is the shape"):
        return "shape"
    if q.startswith("what size"):
        return "size"
    if q.startswith("what material") or "made of" in q:
        return "material"
    if q.startswith("is ") or q.startswith("are ") or q.startswith("does "):
        return "yes/no (attribute)"
    return "other"


@torch.no_grad()
def run_analysis(cfg: dict, device: torch.device, checkpoint: str = "model_best.pt", n_examples: int = 8):
    tokenizer = get_tokenizer(cfg["data_root"])
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
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(torch.load(Path(cfg["output_dir"]) / checkpoint, map_location=device))
    model.eval()

    # Reload raw questions (for text + type classification) aligned with dataset order
    with open(Path(cfg["data_root"]) / "questions" / "val_questions.json") as f:
        raw_questions = json.load(f)["questions"]

    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    examples = []
    idx = 0

    for batch in val_loader:
        images = batch["image"].to(device)
        question_ids = batch["question_ids"].to(device)
        answer_ids = batch["answer_ids"].to(device)

        logits = model(images, question_ids)
        preds = logits.argmax(dim=-1)

        bsz = images.shape[0]
        for i in range(bsz):
            q_text = raw_questions[idx]["question"]
            q_type = classify_question(q_text)

            mask = answer_ids[i] != tokenizer.pad_id
            correct = ((preds[i] == answer_ids[i]) & mask).sum().item()
            total = mask.sum().item()
            is_fully_correct = correct == total and total > 0

            type_total[q_type] += 1
            if is_fully_correct:
                type_correct[q_type] += 1

            if len(examples) < n_examples:
                pred_text = tokenizer.decode(preds[i].tolist())
                true_text = tokenizer.decode(answer_ids[i].tolist())
                examples.append({
                    "question": q_text,
                    "question_type": q_type,
                    "predicted": pred_text,
                    "true": true_text,
                    "correct": is_fully_correct,
                })

            idx += 1

    breakdown = {
        qtype: {
            "accuracy": type_correct[qtype] / type_total[qtype] if type_total[qtype] else 0.0,
            "n": type_total[qtype],
        }
        for qtype in sorted(type_total)
    }

    return breakdown, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", nargs=2, metavar=("CONFIG_A", "CONFIG_B"))
    parser.add_argument("--checkpoint", default="model_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for path in args.compare:
        cfg = load_config(path)
        breakdown, examples = run_analysis(cfg, device, checkpoint=args.checkpoint)
        results[cfg["fusion"]] = {"breakdown": breakdown, "examples": examples}

        print(f"\n=== {cfg['fusion']} — accuracy by question type ===")
        for qtype, stats in breakdown.items():
            print(f"  {qtype:20s} {stats['accuracy']*100:5.1f}%  (n={stats['n']})")

        print(f"\n=== {cfg['fusion']} — sample predictions ===")
        for ex in examples:
            status = "correct" if ex["correct"] else "wrong"
            print(f"  [{status}] ({ex['question_type']}) Q: {ex['question']}")
            print(f"           predicted: '{ex['predicted']}' | true: '{ex['true']}'")

    Path("results").mkdir(exist_ok=True)
    with open("results/analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull breakdown + examples saved to results/analysis.json")


if __name__ == "__main__":
    main()

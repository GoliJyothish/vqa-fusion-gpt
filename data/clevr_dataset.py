"""
CLEVR dataset + dataloader.

Expects a scoped CLEVR subset laid out as:
    data/clevr_subset/
        images/
            train/*.png
            val/*.png
        questions/
            train_questions.json
            val_questions.json

Download the full CLEVR dataset from https://cs.stanford.edu/people/jcjohns/clevr/
and take a subset (e.g. first 2-5k images + matching questions) to keep
training fast on Colab.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class CLEVRVQADataset(Dataset):
    def __init__(self, root: str, split: str, tokenizer, image_size: int = 128, max_len: int = 32):
        """
        Args:
            root: path to data/clevr_subset
            split: "train" or "val"
            tokenizer: object with .encode(str) -> List[int] and
                .pad_id / .vocab_size attributes (reuse your GPT project's
                tokenizer here).
        """
        self.root = Path(root)
        self.split = split
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(self.root / "questions" / f"{split}_questions.json") as f:
            data = json.load(f)
        self.questions = data["questions"]  # each: {image_filename, question, answer}

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.questions)

    def _encode(self, text: str):
        ids = self.tokenizer.encode(text)[: self.max_len]
        ids = ids + [self.tokenizer.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.questions[idx]
        img_path = self.root / "images" / self.split / item["image_filename"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        question_ids = self._encode(item["question"])
        answer_ids = self._encode(item["answer"])

        return {
            "image": image,
            "question_ids": question_ids,
            "answer_ids": answer_ids,
        }

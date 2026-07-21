"""
Simple word-level tokenizer, built from scratch (no external tokenizer
libraries), for CLEVR-style questions and answers.

Why word-level instead of BPE: CLEVR uses a small, fixed vocabulary
(colors, shapes, sizes, materials, numbers, yes/no, question words).
A word-level tokenizer covers this vocabulary directly, is trivial to
explain in an interview, and avoids the extra complexity of subword
merges that BPE needs.

Usage:
    tokenizer = CLEVRTokenizer()
    tokenizer.build_vocab(all_questions_and_answers)   # list of raw strings
    ids = tokenizer.encode("What color is the cube?")
    text = tokenizer.decode(ids)
"""

import json
import re
from pathlib import Path


class CLEVRTokenizer:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self._build_special_tokens()

    def _build_special_tokens(self):
        specials = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        for i, tok in enumerate(specials):
            self.word2id[tok] = i
            self.id2word[i] = tok

    @property
    def pad_id(self) -> int:
        return self.word2id[self.PAD_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    @staticmethod
    def _split_words(text: str):
        # lowercase, split on word characters — good enough for CLEVR's
        # simple templated question/answer strings
        text = text.lower().strip()
        return re.findall(r"[a-z0-9]+|[^\sa-z0-9]", text)

    def build_vocab(self, texts):
        """
        texts: iterable of raw strings (questions and/or answers) to learn
        the vocabulary from. Call this once, on your training split only,
        before encoding anything.
        """
        vocab = set()
        for text in texts:
            vocab.update(self._split_words(text))

        for word in sorted(vocab):
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx] = word

        print(f"Vocab built: {self.vocab_size} tokens "
              f"({self.vocab_size - 4} words + 4 special tokens)")

    def encode(self, text: str, add_special_tokens: bool = True):
        words = self._split_words(text)
        ids = [self.word2id.get(w, self.word2id[self.UNK_TOKEN]) for w in words]
        if add_special_tokens:
            ids = [self.word2id[self.BOS_TOKEN]] + ids + [self.word2id[self.EOS_TOKEN]]
        return ids

    def decode(self, ids, skip_special_tokens: bool = True):
        words = []
        specials = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN}
        for i in ids:
            word = self.id2word.get(int(i), self.UNK_TOKEN)
            if skip_special_tokens and word in specials:
                continue
            words.append(word)
        return " ".join(words)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.word2id, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            self.word2id = json.load(f)
        self.id2word = {v: k for k, v in self.word2id.items()}


def build_tokenizer_from_clevr(questions_json_path: str, save_path: str = None) -> CLEVRTokenizer:
    """
    Convenience function: builds a tokenizer's vocab directly from a CLEVR
    train_questions.json file (uses both the question text and the answer
    text so the vocab covers everything the model needs to read/produce).
    """
    with open(questions_json_path) as f:
        data = json.load(f)

    texts = []
    for q in data["questions"]:
        texts.append(q["question"])
        if "answer" in q:
            texts.append(str(q["answer"]))

    tokenizer = CLEVRTokenizer()
    tokenizer.build_vocab(texts)

    if save_path:
        tokenizer.save(save_path)
        print(f"Saved vocab to {save_path}")

    return tokenizer


if __name__ == "__main__":
    # Quick standalone build + sanity check:
    #   python data/tokenizer.py
    tok = build_tokenizer_from_clevr(
        "data/clevr_subset/questions/train_questions.json",
        save_path="data/clevr_subset/vocab.json",
    )
    sample = "What color is the cube to the right of the yellow sphere?"
    ids = tok.encode(sample)
    print("Sample encode:", ids)
    print("Sample decode:", tok.decode(ids))

# VQA-Fusion: Multimodal Question Answering with a Custom GPT Decoder

## Problem Statement
Visual Question Answering (VQA) requires a model to answer a natural language
question about an image by jointly understanding visual content and linguistic
intent. Introduced by Antol et al. (VQA: Visual Question Answering, ICCV 2015),
most VQA systems follow a common pipeline: a vision encoder extracts image
features, a language encoder processes the question, and a fusion mechanism
combines both before an answer is decoded.

## Dataset
This project uses a scoped subset of **CLEVR** (Johnson et al., CVPR 2017), a
diagnostic dataset with synthetic images and template-generated questions
covering counting, comparison, attribute identification, and spatial reasoning.
CLEVR's controlled, low-bias design makes it well suited to studying reasoning
ability with a small model, rather than requiring open-vocabulary, real-world
VQA at scale.

## Novelty / Contribution
Most VQA systems fuse visual and language features using a pretrained,
off-the-shelf language model. This project instead uses a **GPT-style decoder
built and trained from scratch**, and studies **two fusion strategies** for
injecting visual features into it:

1. **Token concatenation** — projected image features are prepended to the
   question token embeddings, so the decoder attends to both as one sequence.
2. **Cross-attention** — a cross-attention layer is added inside the decoder
   blocks, letting text tokens attend directly to image features at each layer.

The project compares these two strategies under tight parameter constraints
(1–10M parameters), evaluating which fusion mechanism a small, from-scratch
decoder learns more effectively — a comparison not made in the base paper,
which assumes large pretrained components on both sides.

## Architecture
```
Image ──▶ Frozen Vision Encoder (pretrained CNN/ViT) ──▶ Feature Projection ──┐
                                                                               ├─▶ Fusion ──▶ Custom GPT Decoder ──▶ Answer
Question ─▶ Tokenizer ─▶ Token Embeddings ────────────────────────────────────┘
```

## Repo Structure
```
vqa-fusion-gpt/
├── README.md
├── data/                 # CLEVR subset prep + dataset/dataloader code
├── models/
│   ├── vision_encoder.py # frozen pretrained CNN/ViT feature extractor
│   ├── gpt_decoder.py    # custom GPT decoder (adapted from LLM-from-scratch project)
│   └── fusion.py         # concatenation + cross-attention fusion variants
├── configs/              # training configs per fusion strategy
├── train.py
├── evaluate.py           # runs both fusion variants, reports comparison table
├── notebooks/            # Colab experiment notebooks
└── results/              # loss curves, accuracy tables, plots
```

## Base Papers
- Antol et al., *VQA: Visual Question Answering*, ICCV 2015
- Johnson et al., *CLEVR: A Diagnostic Dataset for Compositional Language and
  Elementary Visual Reasoning*, CVPR 2017

## Status
🚧 In progress — see `results/` for the latest comparison numbers as they land.

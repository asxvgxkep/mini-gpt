# MiniGPT

A GPT-style decoder-only language model implemented from scratch with PyTorch.

This project builds a small language model from the ground up, including tokenizer, Transformer architecture, training pipeline, checkpoint saving, and inference.

## Features

- Decoder-only Transformer architecture
- Multi-head self-attention
- SentencePiece tokenizer
- PyTorch training pipeline
- Epoch-level checkpoint saving
- CPU inference support

## Model Configuration

- Parameters: 97.54M
- Layers: 12
- Attention heads: 12
- Hidden dimension: 768
- Context length: 256

## Dataset

- Dataset: TinyStories
- Token count: 23.6M

## Training Result

Training environment:

- GPU: NVIDIA RTX 4090D
- Framework: PyTorch
- Epochs: 20

Final training loss:

```
epoch 20, loss=1.4606
```

## Generation Example

Prompt:

```
Once upon a time
```

Output:

```
Once upon a time, there was a little girl named Lily. She loved to play outside in her backyard. One day, she saw a mole digging holes in the ground...
```

## Run Inference

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate text:

```bash
python -m inference.generate
```

## Project Structure

```
mini-gpt/
├── model/          # GPT model implementation
├── train/          # training pipeline
├── tokenizer/      # SentencePiece tokenizer
├── inference/      # text generation
├── configs/        # model configurations
└── mini_gpt.pt     # trained model weights
```
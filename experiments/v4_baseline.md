# MiniGPT v4 Baseline Experiment

> This historical record was reconstructed retrospectively from the original training notes. It was added after the run to preserve the project's experiment history.

## Model

- Parameters: approximately 97.54M
- Layers: 12
- Attention heads: 12
- Hidden dimension: 768
- Context length: 256
- Vocabulary size: 8,000
- Dropout: 0.1

## Dataset

- Tokens: 23,587,586
- Source: the first 100,000,000 characters of TinyStories

## Training

- Hardware: cloud NVIDIA RTX 4090D
- Batch size: 32
- Learning rate: `3e-4`
- Epochs: 20

## Recorded training loss

| Epoch | Training loss |
| ---: | ---: |
| 1 | 3.0862 |
| 2 | 2.3002 |
| 3 | 2.0813 |
| 4 | 1.9617 |
| 5 | 1.8803 |
| 6 | 1.8188 |
| 7 | 1.7694 |
| 8 | 1.7279 |
| 9 | 1.6921 |
| 10 | 1.6612 |
| 11 | 1.6329 |
| 12 | 1.6072 |
| 13 | 1.5843 |
| 14 | 1.5630 |
| 15 | 1.5430 |
| 16 | 1.5246 |
| 17 | 1.5071 |
| 18 | 1.4907 |
| 19 | 1.4752 |
| 20 | 1.4606 |

Final recorded training loss: **1.4606**.

## Evaluation limits

Only training loss was recorded, and no held-out validation set was used. The training-loss sequence alone cannot establish whether or how much the model overfit.

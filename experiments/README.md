# MiniGPT Experiment Evolution

The early experiment documents in this directory were reconstructed retrospectively from the original training logs and notes, then added later to preserve the project history. The Markdown files should not be read as records committed at the time the experiments ran.

## Run summary

| Run | Params | Data | Batch | LR | Epochs | Validation | Recorded result |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| [v0.2](v02.md) | ~33.54M | Earlier subset; exact size unverified | 8 | `1e-3` | 10 | No held-out result recorded | Final train loss 4.3568 |
| [v0.3](v03.md) | 33.54M | 23,587,586 tokens from the first 100M TinyStories characters | 32 | `3e-4` | 20 | Not used | Final train loss 1.6541 |
| [v4 baseline](v4_baseline.md) | ~97.54M | 23,587,586 tokens from the first 100M TinyStories characters | 32 | `3e-4` | 20 | Not used | Final train loss 1.4606 |
| [Final full-data run](../README.md#final-training-result) | 97.54M | Full TinyStories corpus; 90% train / 10% validation | 32 |`3e-4` (epochs 1-10); `1e-4` (epochs 11-15) | 15 | 10% held out | Epoch 15: train 1.3002, validation 1.2654 |

The sequence records the progression from an early 33.54M-parameter run, to a revised 33.54M training setup, to a 97.54M baseline, and finally to 97.54M full-corpus training with held-out validation. Multiple settings changed between runs, so the results do not isolate any single cause for the later loss improvements.

For the final run, the exact best validation loss was `1.2653516990689389`. Epochs 11-15 loaded the model weights from epoch 10, but did not restore optimizer state: AdamW was reinitialized and the learning rate changed to `1e-4`. This was a weight-resumed continuation, not a strict optimizer-state resume. The full epoch record is in [`logs/train_full.log`](../logs/train_full.log).

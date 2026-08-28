from pathlib import Path
import re

import matplotlib.pyplot as plt


LOG_PATH = Path("logs/train_full.log")
OUTPUT_PATH = Path("experiments/full_loss_curve.png")

pattern = re.compile(
    r"epoch=(\d+), "
    r"train_loss=([\d.]+), "
    r"val_loss=([\d.]+)"
)

epochs = []
train_losses = []
val_losses = []

with LOG_PATH.open(
    "r",
    encoding="utf-8"
) as f:
    for line in f:
        match = pattern.fullmatch(
            line.strip()
        )

        if match is None:
            continue

        epoch, train_loss, val_loss = (
            match.groups()
        )

        epochs.append(
            int(epoch)
        )

        train_losses.append(
            float(train_loss)
        )

        val_losses.append(
            float(val_loss)
        )


if not epochs:
    raise RuntimeError(
        f"No training metrics found in {LOG_PATH}"
    )


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Train loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "MiniGPT Full TinyStories Training"
)

plt.xticks(epochs)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"saved {OUTPUT_PATH}"
)
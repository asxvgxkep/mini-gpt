import matplotlib.pyplot as plt


epochs = list(range(1, 21))

losses = [
    3.0862,
    2.3002,
    2.0813,
    1.9617,
    1.8803,
    1.8188,
    1.7694,
    1.7279,
    1.6921,
    1.6612,
    1.6329,
    1.6072,
    1.5843,
    1.5630,
    1.5430,
    1.5246,
    1.5071,
    1.4907,
    1.4752,
    1.4606,
]


plt.plot(
    epochs,
    losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MiniGPT Training Loss")

plt.grid(True)

plt.savefig(
    "experiments/loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

print("saved experiments/loss_curve.png")
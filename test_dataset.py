from train.dataset import TextDataset


tokens = [
    1,2,3,4,5,6
]


dataset = TextDataset(
    tokens,
    context_length=3
)


for x,y in dataset:

    print(
        x.tolist(),
        y.tolist()
    )
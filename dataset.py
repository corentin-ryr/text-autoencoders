import torch
from torch.utils.data import DataLoader, Dataset

from utils import ToyVocab

class BinaryToyDataset(Dataset):
    def __init__(self, numClasses=5, sampleLength=50, samplePerClass=100) -> None:
        super().__init__()
        self.rng = torch.Generator().manual_seed(0)
        vocab = ToyVocab()

        self.samples, self.labels = [], []
        for i in range(numClasses):
            classSamples = torch.randint(2, (sampleLength,), generator=self.rng).repeat(100, 1)
            swapMask = torch.rand(classSamples.shape, generator=self.rng) < 0.2
            classSamples = torch.where(swapMask, 1 - classSamples, classSamples)

            self.samples.append(classSamples)
            self.labels.append(torch.ones(samplePerClass) * i)

        self.samples = torch.concat(self.samples) + vocab.nspecial
        self.labels = torch.concat(self.labels)


    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    dataset = BinaryToyDataset()

    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    for data in dataloader:
        print(data)

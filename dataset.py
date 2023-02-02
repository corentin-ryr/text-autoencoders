import random
import torch
from torch.utils.data import Dataset, DataLoader
from utils import ToyVocab

from datasets import load_dataset

class BinaryToyDataset(Dataset):
    def __init__(self, numClasses=5, sampleLength=50, samplePerClass=100, vocab=None) -> None:
        super().__init__()
        self.rng = torch.Generator().manual_seed(0)
        if not vocab: vocab = ToyVocab(2) 

        self.samples, self.labels = [], []
        for i in range(numClasses):
            classSamples = torch.randint(vocab.vocab_length - vocab.nspecial, (sampleLength,), generator=self.rng).repeat(100, 1)
            swapMask = torch.rand(classSamples.shape, generator=self.rng) < 0.2

            classSamples = torch.where(swapMask, torch.randint(vocab.vocab_length - vocab.nspecial, size=classSamples.shape), classSamples)

            self.samples.append(classSamples)
            self.labels.append(torch.ones(samplePerClass) * i)

        self.samples = torch.concat(self.samples) + vocab.nspecial
        self.labels = torch.concat(self.labels)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class BinaryDuplicateToyDataset(Dataset):
    def __init__(self, sampleLength=50, numberSamples=400, proportionDuplicate=0.15, mode:str = "sampling", nonMatchToMatchRatio=2, vocab=None) -> None:
        """Generate a toy dataset for contrastive learning.

        Args:
            sampleLength (int, optional): Length of each samples. Defaults to 50.
            numberSamples (int, optional): Total number of unique samples (the duplicates come on top of that). Defaults to 400.
            proportionDuplicate (float, optional): Percentage of duplicate in the final dataset (approximation). Defaults to 0.15.
            mode (str, optional): Can be 'sampling' (the dataset returns just the sample), 'pairs' (returns pairs that are either duplicate or not). Defaults to "sampling".
        """
        super().__init__()
        self.rng = torch.Generator().manual_seed(0)
        if not vocab: vocab = ToyVocab()
        self.mode = mode

        self.samples = []
        self.exactMatch = []
        for i in range(numberSamples):
            sample = torch.randint(vocab.vocab_length - vocab.nspecial, (sampleLength,), generator=self.rng)
            self.samples.append(sample)

            if torch.rand(1, generator=self.rng) < proportionDuplicate:
                swapMask = torch.rand(sample.shape, generator=self.rng) < 0.15
                duplicateSample = torch.where(swapMask, torch.randint(vocab.vocab_length - vocab.nspecial, size=sample.shape), sample)
                
                self.samples.append(duplicateSample)
                self.exactMatch.append( (len(self.samples) - 1, len(self.samples) - 2) )
        
        self.samples = torch.stack(self.samples) + vocab.nspecial
        self.nonMatchToMatchRatio = nonMatchToMatchRatio

    def setMode(self, mode): self.mode = mode

    def __getitem__(self, index):
        if self.mode == "sampling": return self.samples[index], torch.tensor(0)

        if self.mode == "pairs":
            if index < len(self.exactMatch):
                sample1 = self.samples[self.exactMatch[index][0]]
                sample2 = self.samples[self.exactMatch[index][1]]
                return sample1, sample2, torch.tensor(1)
            else:
                sample1Idx = torch.randint(len(self.samples) - 1, (1,), generator=self.rng).item()

                sample1Matches = [match[1] for match in self.exactMatch if match[0] == sample1Idx]
                sample1Matches += [match[0] for match in self.exactMatch if match[1] == sample1Idx]
                sample1Matches += [sample1Idx]

                allowedSamples = set(list(range(len(self.samples)))).difference(sample1Matches)

                sample2Idx = list(allowedSamples)[torch.randint(len(allowedSamples), (1,), generator=self.rng).item()]

                return self.samples[sample1Idx], self.samples[sample2Idx], torch.tensor(0)

    def __len__(self):
        if self.mode == "sampling": return len(self.samples)
        if self.mode == "pairs": return int(len(self.exactMatch) * self.nonMatchToMatchRatio)

class SickDataset(Dataset):
    def __init__(self, set="train", mode="sampling") -> None:
        super().__init__()
        self.set = set
        self.mode = mode

        sickDataset = load_dataset("sick")
        self.train = sickDataset["train"]
        self.test = sickDataset["test"]
    
    def __getitem__(self, index):
        if self.set == "train":
            sample = self.train[index]
        if self.set == "test":
            sample = self.test[index]

        if self.mode == "sampling":
            return sample["sentence_A"] if random.random() < 0.5 else sample["sentence_B"]
        if self.mode == "pairs":
            return sample["sentence_A"], sample["sentence_B"], torch.tensor(0 if sample["relatedness_score"] < 3.5 else 1)
        
        raise ValueError(f"{self.set} is not a correct value for mode.")

    def __len__(self):
        if self.set == "train": return self.train.num_rows
        if self.set == "test": return self.test.num_rows

        raise ValueError(f"{self.set} is not a correct value for set.")

    def setMode(self, mode):
        self.mode = mode

if __name__ == "__main__":
    sickDataset = SickDataset()

    sickDataset.setMode("pairs")
    dataloader = DataLoader(sickDataset, batch_size=3, shuffle=True)

    print(next(iter(dataloader)))
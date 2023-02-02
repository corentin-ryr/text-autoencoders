import os
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchmetrics import AUC
from torchmetrics.classification import BinaryPrecisionRecallCurve
from tqdm import tqdm

def simpleVizualization(trainer, dataloader, device="cpu"):
    allEncodings, allLabels = None, None
    for samples, labels in tqdm(dataloader, desc="Computing encodings for simple visualization"):
        encodings = trainer.encode(samples.to(device)).to("cpu")

        if allEncodings is not None:
            allEncodings = torch.cat((allEncodings, encodings))
            allLabels = torch.cat((allLabels, labels))
        else:
            allEncodings = encodings
            allLabels = labels

    fig, ax = plt.subplots()
    for g in np.unique(allLabels):
        idx = np.array(np.where(allLabels == g))
        ax.scatter(allEncodings[idx, 0], allEncodings[idx, 1], s=2)

    trainer.add_image_tensorboard("simple", fig)


def pcaOn2Dim(trainer, dataloader, device="cpu"):
    print()
    pca = PCA(n_components=2)

    allEncodings, allLabels = None, None
    for samples, labels in tqdm(dataloader, desc="Computing encodings for PCA projection"):
        encodings = trainer.encode(samples.to(device)).to("cpu")

        if allEncodings is not None:
            allEncodings = torch.cat((allEncodings, encodings))
            allLabels = torch.cat((allLabels, labels))
        else:
            allEncodings = encodings
            allLabels = labels

    pca.fit(allEncodings)
    print(f"Variance explained by the first two compoenents: {pca.explained_variance_ratio_}")

    pcaEncodings = pca.transform(allEncodings)

    fig, ax = plt.subplots()
    for g in np.unique(allLabels):
        idx = np.where(allLabels == g)
        ax.scatter(pcaEncodings[idx, 0], pcaEncodings[idx, 1], s=2)

    trainer.add_image_tensorboard("pca", fig)


def tsneOn2Dim(trainer, dataloader, device="cpu"):
    print()
    tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=3)

    allEncodings, allLabels = None, None
    for samples, labels in tqdm(dataloader, desc="Computing encodings for t-SNE projection"):
        encodings = trainer.encode(samples.to(device)).to("cpu")

        if allEncodings is not None:
            allEncodings = torch.cat((allEncodings, encodings))
            allLabels = torch.cat((allLabels, labels))
        else:
            allEncodings = encodings
            allLabels = labels

    pcaEncodings = tsne.fit_transform(allEncodings)
    print(f"KL divergence after {tsne.n_iter_}: {tsne.kl_divergence_}")

    fig, ax = plt.subplots()
    for g in np.unique(allLabels):
        idx = np.where(allLabels == g)
        ax.scatter(pcaEncodings[idx, 0], pcaEncodings[idx, 1], s=1)

    trainer.add_image_tensorboard("tsne", fig)


class Visualizer(ABC):
    def __init__(self, trainer, dataloader, device: str = "cpu") -> None:
        self.trainer = trainer
        self.dataloader = dataloader
        self.savePath = "outputs"
        self.device = device

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

    def __call__(self) -> None:
        pass

    def _show_save_fig(self, tag, fig: plt.Figure):
        self.trainer.add_image_tensorboard(tag, fig)


class NeihgborInDistance(Visualizer):
    def __call__(self, distance: float) -> None:

        allEncodings = {}
        for samples, _ in tqdm(self.dataloader, desc="Computing encodings"):
            encodings = self.trainer.encode(samples.to(self.device)).to("cpu")

            for idx, encoding in enumerate(encodings):
                allEncodings[idx] = encoding

        numberNeighborInDistance = []

        for encoding in allEncodings:
            numberNeighbors = 0
            for encoding1 in allEncodings:
                if torch.max(torch.abs(allEncodings[encoding] - allEncodings[encoding1])) < distance:
                    numberNeighbors += 1

            numberNeighborInDistance.append(numberNeighbors)

        average = sum(numberNeighborInDistance) / len(numberNeighborInDistance)
        print(f"Average number of neighbors closer than {distance:.2f}: {average}")
        print(f"Or {(average / len(allEncodings) * 100):.2f}% of the samples")

        return average / len(allEncodings)


class PlotProximity(Visualizer):
    def __call__(self, percentages) -> None:
        distances = []
        labels = []
        yValues = []

        prc = BinaryPrecisionRecallCurve()

        for sample0, sample1, label in tqdm(self.dataloader, desc="Proximity visualization"):
            # Encode the two sentences in sample
            encoding0 = self.trainer.encode(sample0.to(self.device)).to("cpu")
            encoding1 = self.trainer.encode(sample1.to(self.device)).to("cpu")
            distance = torch.max(torch.abs(encoding0 - encoding1), dim=1)[0].tolist()

            distances += distance
            labels += label.tolist()
        prc.update(1 - (torch.tensor(distances) / max(distances)), torch.tensor(labels))

        distances = np.array(distances)
        labels = np.array(labels)
        yValues = np.random.random((len(distances)))

        # Plot the distance with a noise in y, the distance in x and a different color if they are a duplicate
        cdict = {0: "red", 1: "green"}
        legend = ["Non duplicate", "Duplicate"]

        fig, ax = plt.subplots()
        for g in np.unique(labels):
            ix = np.where(labels == g)
            ax.scatter(distances[ix], yValues[ix], c=cdict[g], label=legend[g], s=1)

        ax.get_yaxis().set_visible(False)
        ax.set_xlabel("L-inf Distance between pair")
        ax.legend()
        ax.set_title("Distances for duplicate and non-duplicate pairs")

        self.trainer.add_image_tensorboard("distances", fig)

        fig, ax = plt.subplots()
        for g in np.unique(labels):
            ix = np.where(labels == g)
            n, x, _ = ax.hist(
                distances[ix], bins=np.linspace(0, max(distances), 100), histtype="step", density=True, alpha=0.5, color=cdict[g]
            )
            if len(np.unique(distances[ix])) > 1:
                density = stats.gaussian_kde(distances[ix])
                ax.plot(x, density(x), label=legend[g], c=cdict[g])

        ax.legend()
        ax.set_ylabel("Density of pairs")
        ax.set_xlabel("Linf Distance between pair")
        ax.set_title("Histogram of the distance between pairs")

        self.trainer.add_image_tensorboard("histogram", fig)

        (precision, recall, thresholds) = prc.compute()

        # F1 score
        f1 = torch.max(2 * precision * recall / (precision + recall))
        print(f"F1 score {f1}")
        self.trainer.writer.add_text("f1", str(f1))

        auc = AUC()
        aucrp = auc(recall, precision).item()
        print(f"Area under precision recall curve: {aucrp:.3f}")

        fig, ax = plt.subplots()
        ax.plot(recall, precision, marker="x")
        ax.set_title("Precision recall curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid()

        self.trainer.add_image_tensorboard("precision-recall", fig)

        distancePercentages = []
        fig, axes = plt.subplots(1, len(percentages))
        for idx, percentage in enumerate(percentages):
            # Distance for {percentage} of the duplicates
            sorted_distances = np.sort(distances[labels == 1])
            linear = np.linspace(0, 1, len(sorted_distances))

            firstIndiceAbove90 = np.argmax(linear >= percentage)
            distance90Percent = sorted_distances[firstIndiceAbove90]
            print(f"{percentage*100:.2f}% distances under {distance90Percent}")
            distancePercentages.append(distance90Percent)

            axes[idx].plot(sorted_distances, linear)
            axes[idx].vlines([sorted_distances[firstIndiceAbove90]], 0, 1, colors=["red"], label="90% threshold")
            axes[idx].set_title("Percentage of duplicate pair under distance")
            axes[idx].set_xlabel("Distance")
            axes[idx].set_ylabel("Number of pairs")
            axes[idx].legend()

        self.trainer.add_image_tensorboard("dup-under-distance", fig)

        return distancePercentages


class SRCC(Visualizer):
    def __call__(self) -> None:

        similarities = []
        truth = []
        for sample1, sample2, label in tqdm(self.dataloader, desc="Computing SRCC"):

            linfSim = torch.exp(
                -torch.max(
                    torch.abs(
                        self.trainer.encode(sample1.to(self.device)).to("cpu") - self.trainer.encode(sample2.to(self.device)).to("cpu")
                    ),
                    dim=1,
                )[0]
            )
            cosineSim = F.cosine_similarity(
                self.trainer.encode(sample1.to(self.device)).to("cpu"), self.trainer.encode(sample2.to(self.device)).to("cpu")
            ).squeeze()
            similarities.append(linfSim)
            truth.append(label)

        similarities = torch.concat(similarities)
        truth = torch.concat(truth)

        srcc = spearmanr(similarities, truth)
        self.trainer.writer.add_text("SRCC", str(srcc.correlation * 100))
        print(f"Spearmen Ranking correlation coefficient {srcc.correlation * 100:.2f}")
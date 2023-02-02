import argparse
import distutils
import os
from abc import ABC
from typing import Tuple
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torchmetrics import AUC
from torchmetrics.classification import BinaryPrecisionRecallCurve
from tqdm import tqdm

# Tokenizer and vocab ================================================

class Tokenizer(ABC):
    def __init__(self) -> None:
        super().__init__()

        self.pad = 0
        self.go = 0
        self.eos = 0
        self.blank = 0
        self.nspecial = 4

        self.word2idx = {}
        self.idx2word = []

        self.vocab_length = 0

class ToyVocab(Tokenizer):
    def __init__(self, numWords=2) -> None:
        super().__init__()

        self.pad = 0
        self.go = 1
        self.eos = 2
        self.blank = 3

        self.nspecial = 4

        self.word2idx = {}
        self.idx2word = []

        self.vocab_length = self.nspecial + numWords

# Noise ========================================================================================

def word_shuffle(vocab, x, k):  # slight shuffle such that |sigma[i]-i| <= k
    base = torch.arange(x.size(0), dtype=torch.float).repeat(x.size(1), 1).t()
    inc = (k + 1) * torch.rand(x.size())
    inc[x == vocab.go] = 0  # do not shuffle the start sentence symbol
    inc[x == vocab.pad] = k + 1  # do not shuffle end paddings
    # inc[x == vocab.eos] = k+1  # do not shuffle eos
    _, sigma = (base + inc).sort(dim=0)
    return x[sigma, torch.arange(x.size(1))]


def word_drop(vocab, x, p):  # drop words with probability p
    x_ = []
    for i in range(x.size(0)):
        words = x[i, :].tolist()
        keep = np.random.rand(len(words)) > p
        keep[0] = True  # do not drop the start sentence symbol
        keep[words.index(vocab.eos)] = True  # do not drop the eos
        sent = [w for j, w in enumerate(words) if keep[j]]
        sent += [vocab.pad] * (len(words) - len(sent))
        x_.append(sent)
    return torch.LongTensor(x_).contiguous().to(x.device)


def word_blank(vocab, x, p):  # blank words with probability p
    blank = (torch.rand(x.size(), device=x.device) < p) & (x != vocab.go) & (x != vocab.pad) & (x != vocab.eos)
    x_ = x.clone()
    x_[blank] = vocab.blank
    return x_


def word_substitute(vocab, x, p):  # substitute words with probability p
    keep = (torch.rand(x.size(), device=x.device) > p) | (x == vocab.go) | (x == vocab.pad) | (x == vocab.eos)
    x_: torch.Tensor = x.clone()
    x_.random_(vocab.nspecial, vocab.vocab_length)
    x_[keep] = x[keep]
    return x_

def noisy(vocab: Tokenizer, x: torch.Tensor, drop_prob: float, blank_prob: float, sub_prob: float, shuffle_dist: int):
    """Takes a tensor of word indices and applies noise with the given parameters.

    Args:
        vocab (Vocab): The vocab object
        x (torch.Tensor): tensor of shape length of sentence X batch size
        drop_prob (float): Probability of erasing a word from the sentence
        blank_prob (float): Probability of replacing a word by a blank symbol
        sub_prob (float): Probability of replacing a word by another random word in the vocab
        shuffle_dist (int): Distance from the original position of a word and its new position

    Returns:
        torch.Tensor: The noisy tensor of the same shape as the input
    """
    if shuffle_dist > 0:
        x = word_shuffle(vocab, x, shuffle_dist)
    if drop_prob > 0:
        x = word_drop(vocab, x, drop_prob)  # ok
    if blank_prob > 0:
        x = word_blank(vocab, x, blank_prob)  # ok
    if sub_prob > 0:
        x = word_substitute(vocab, x, sub_prob)  # ok
    return x


# Scheduler ===================================================================================

class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


# Term frequency ====================================================================

def compute_term_frequency(dataloader, vocabLength):

    weights = torch.zeros((vocabLength))
    nbWords = 0

    for idx, data in enumerate(tqdm(dataloader)):
        source, *_ = data
        source = source.flatten()

        for token in source:
            weights[token] += 1
            nbWords += 1

    weights = nbWords / (vocabLength * (weights + 1))

    return weights


# Visualization =========================================================================


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

    def _show_save_fig(self, tag, fig:plt.Figure):
        self.trainer.add_image_tensorboard(tag, fig)

class NeihgborInDistance(Visualizer):
    def __call__(self, distance:float) -> None:

        allEncodings = {}
        for samples, _ in tqdm(self.dataloader, desc="Computing encodings"):
            encodings = self.trainer.encode(samples.to(self.device)).to("cpu")

            for idx, encoding in enumerate(encodings):
                allEncodings[idx] = encoding
        
        numberNeighborInDistance = []

        for encoding in allEncodings:
            numberNeighbors = 0
            for encoding1 in allEncodings:
                if torch.max(torch.abs(allEncodings[encoding] - allEncodings[encoding1])) < distance: numberNeighbors += 1
            
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
            n, x, _ = ax.hist(distances[ix], bins=np.linspace(0, max(distances), 100), histtype="step", density=True, alpha=0.5, color=cdict[g])
            if len(np.unique(distances[ix])) > 1:
                density = stats.gaussian_kde(distances[ix])
                ax.plot(x, density(x), label=legend[g], c=cdict[g])

        ax.legend()
        ax.set_ylabel("Density of pairs")
        ax.set_xlabel("Linf Distance between pair")
        ax.set_title("Histogram of the distance between pairs")
        
        self.trainer.add_image_tensorboard("histogram", fig)

        (precision, recall, thresholds) = prc.compute()

        #F1 score
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

            linfSim = torch.exp(-torch.max(torch.abs(self.trainer.encode(sample1.to(self.device)).to("cpu") - self.trainer.encode(sample2.to(self.device)).to("cpu")), dim=1)[0])
            cosineSim = F.cosine_similarity(self.trainer.encode(sample1.to(self.device)).to("cpu"), self.trainer.encode(sample2.to(self.device)).to("cpu")).squeeze()
            similarities.append(linfSim)
            truth.append(label)

        similarities = torch.concat(similarities)
        truth = torch.concat(truth)
     
        srcc = spearmanr(similarities, truth)
        self.trainer.writer.add_text("SRCC", str(srcc.correlation * 100))
        print(f"Spearmen Ranking correlation coefficient {srcc.correlation * 100:.2f}")


# Arg parser ===============================================================================

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training", type=str)
    parser.add_argument("--model-param", type=str)
    parser.add_argument("--num-epoch", type=int)
    parser.add_argument("--use-gpu", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--use-noise", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--lr", type=float)
    parser.add_argument("--run-name", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# Various ===========================================================================================
def get_device(useGPU):
    if torch.cuda.is_available() and useGPU:
        return torch.device("cuda:0")

    if torch.backends.mps.is_available() and useGPU:
        return torch.device("mps")

    return torch.device("cpu")

# Sigmoid annealing =================================================================================
def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L 

# Probability sampling ==============================================================================

def priorSampling(sampleShape:Tuple[int], prior:str="gaussian"):
    if prior == "gaussian":
        return torch.randn(sampleShape)
    if prior == "molUniform":
        means = torch.rand(sampleShape) - 0.5
        return torch.randn(sampleShape) * 0.1 + means
    if prior == "uniform":
        return torch.rand(sampleShape) - 0.5




# Grad viz ===========================================================================================

class InteractiveGradFlow():

    def __init__(self) -> None:
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        plt.subplots_adjust(bottom=0.3, top=1)
        self.axes = plt.gca()
        self.axes.set_title("Gradient flow")

    def update_plot(self, named_parameters):

        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())

        self.axes.clear()

        self.axes.bar(np.arange(len(max_grads)), max_grads,
                      alpha=0.1, lw=1, color="c")
        self.axes.bar(np.arange(len(max_grads)), ave_grads,
                      alpha=0.1, lw=1, color="b")
        self.axes.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        self.axes.set_xticks(range(0, len(ave_grads), 1),
                             layers, rotation="vertical")
        self.axes.set_xlim(left=0, right=len(ave_grads))
        # zoom in on the lower gradient regions
        self.axes.set_ylim(bottom=-0.001, top=0.02)
        self.axes.set_xlabel("Layers")
        self.axes.set_ylabel("average gradient")
        self.axes.set_title("Gradient flow")
        self.axes.grid(True)
        self.axes.legend([Line2D([0], [0], color="c", lw=4),
                          Line2D([0], [0], color="b", lw=4),
                          Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

        plt.draw()
        plt.pause(1e-20)
import argparse
import distutils
from abc import ABC
from typing import Tuple
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from tqdm import tqdm

# Tokenizer and vocab ================================================

class ToyVocab():
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


def noisy(vocab: ToyVocab, x: torch.Tensor, drop_prob: float, blank_prob: float, sub_prob: float, shuffle_dist: int):
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
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 1.0 / (1.0 + np.exp(-(v * 12.0 - 6.0)))
            v += step
            i += 1
    return L


# Probability sampling ==============================================================================


def priorSampling(sampleShape: Tuple[int], prior: str = "gaussian"):
    if prior == "gaussian":
        return torch.randn(sampleShape)
    if prior == "molUniform":
        means = torch.rand(sampleShape) - 0.5
        return torch.randn(sampleShape) * 0.1 + means
    if prior == "uniform":
        return torch.rand(sampleShape) - 0.5


# Grad viz ===========================================================================================


class InteractiveGradFlow:
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

        self.axes.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        self.axes.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        self.axes.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        self.axes.set_xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        self.axes.set_xlim(left=0, right=len(ave_grads))
        # zoom in on the lower gradient regions
        self.axes.set_ylim(bottom=-0.001, top=0.02)
        self.axes.set_xlabel("Layers")
        self.axes.set_ylabel("average gradient")
        self.axes.set_title("Gradient flow")
        self.axes.grid(True)
        self.axes.legend(
            [Line2D([0], [0], color="c", lw=4), Line2D([0], [0], color="b", lw=4), Line2D([0], [0], color="k", lw=4)],
            ["max-gradient", "mean-gradient", "zero-gradient"],
        )

        plt.draw()
        plt.pause(1e-20)

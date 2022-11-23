import numpy as np
import torch
from abc import ABC
from tqdm import tqdm

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
import os
import distutils
import argparse

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

    def __init__(self) -> None:
        super().__init__()

        self.pad = 0
        self.go = 1
        self.eos = 2
        self.blank = 3
        
        self.nspecial = 4

        self.word2idx = {}
        self.idx2word = []

        self.vocab_length = 6

# Noise ========================================================================================


def word_shuffle(vocab, x, k):   # slight shuffle such that |sigma[i]-i| <= k
    base = torch.arange(x.size(0), dtype=torch.float).repeat(x.size(1), 1).t()
    inc = (k+1) * torch.rand(x.size())
    inc[x == vocab.go] = 0     # do not shuffle the start sentence symbol
    inc[x == vocab.pad] = k+1  # do not shuffle end paddings
    # inc[x == vocab.eos] = k+1  # do not shuffle eos
    _, sigma = (base + inc).sort(dim=0)
    return x[sigma, torch.arange(x.size(1))]

def word_drop(vocab, x, p):     # drop words with probability p
    x_ = []
    for i in range(x.size(0)):
        words = x[i, :].tolist()
        keep = np.random.rand(len(words)) > p
        keep[0] = True  # do not drop the start sentence symbol
        keep[words.index(vocab.eos)] = True  # do not drop the eos
        sent = [w for j, w in enumerate(words) if keep[j]]
        sent += [vocab.pad] * (len(words)-len(sent))
        x_.append(sent)
    return torch.LongTensor(x_).contiguous().to(x.device)

def word_blank(vocab, x, p):     # blank words with probability p
    blank = (torch.rand(x.size(), device=x.device) < p) & \
        (x != vocab.go) & (x != vocab.pad) & (x != vocab.eos)
    x_ = x.clone()
    x_[blank] = vocab.blank
    return x_

def word_substitute(vocab, x, p):     # substitute words with probability p
    keep = (torch.rand(x.size(), device=x.device) > p) | \
        (x == vocab.go) | (x == vocab.pad) | (x == vocab.eos)
    x_:torch.Tensor = x.clone()
    x_.random_(vocab.nspecial, vocab.vocab_length)
    x_[keep] = x[keep]
    return x_

def noisy(vocab:Tokenizer, x:torch.Tensor, drop_prob:float, blank_prob:float, sub_prob:float, shuffle_dist:int):
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
        x = word_drop(vocab, x, drop_prob) # ok
    if blank_prob > 0:
        x = word_blank(vocab, x, blank_prob) # ok
    if sub_prob > 0:
        x = word_substitute(vocab, x, sub_prob) # ok
    return x




# Scheduler ===================================================================================


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
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
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
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
        source, _ = data
        source = source.flatten()

        for token in source:
            weights[token] += 1
            nbWords += 1

    weights = nbWords / (vocabLength * (weights + 1))

    return weights


# Visualization =========================================================================

def simpleVizualization(trainer, dataloader, showFig, savePath, device="cpu"):
    allEncodings, allLabels = None, None
    for samples, labels in tqdm(dataloader, desc="Projection PCA"):
        encodings = trainer.encode(samples.to(device)).to("cpu")

        if allEncodings is not None:
            allEncodings = torch.cat((allEncodings, encodings))
            allLabels = torch.cat((allLabels, labels))
        else:
            allEncodings = encodings
            allLabels = labels

    fig, ax = plt.subplots()
    for g in np.unique(allLabels):
        idx = np.where(allLabels == g)
        ax.scatter(allEncodings[idx, 0], allEncodings[idx, 1], s=2)

    plt.savefig(os.path.join(savePath, "simple.png"))
    if showFig: 
        plt.show()

def pcaOn2Dim(trainer, dataloader, showFig, savePath, device="cpu"):
    print()
    pca = PCA(n_components=2)

    allEncodings, allLabels = None, None
    for samples, labels in tqdm(dataloader, desc="Projection PCA"):
        encodings = trainer.encode(samples.to(device)).to("cpu")

        if allEncodings is not None:
            allEncodings = torch.cat((allEncodings, encodings))
            allLabels = torch.cat((allLabels, labels))
        else:
            allEncodings = encodings
            allLabels = labels

    pca.fit(allEncodings)
    print(
        f"Variance explained by the first two compoenents: {pca.explained_variance_ratio_}")

    pcaEncodings = pca.transform(allEncodings)

    fig, ax = plt.subplots()
    for g in np.unique(allLabels):
        idx = np.where(allLabels == g)
        ax.scatter(pcaEncodings[idx, 0], pcaEncodings[idx, 1], s=2)

    plt.savefig(os.path.join(savePath, "pca.png"))
    if showFig:
        plt.show()


def tsneOn2Dim(trainer, dataloader, showFig, savePath, device="cpu"):
    print()
    tsne = TSNE(n_components=2, learning_rate='auto',
                init='random', perplexity=3)

    allEncodings, allLabels = None, None
    for samples, labels in tqdm(dataloader, desc="Projection PCA"):
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

    plt.savefig(os.path.join(savePath, "tsne.png"))
    if showFig:
        plt.show()

# Arg parser ===============================================================================

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training", type=str)
    parser.add_argument("--model-param", type=str)
    parser.add_argument("--num-epoch", type=int)
    parser.add_argument("--use-gpu", type=lambda x:bool(distutils.util.strtobool(x)))
    parser.add_argument("--use-noise", type=lambda x:bool(distutils.util.strtobool(x)))
    parser.add_argument("--lr", type=float)

    # parse args
    args = parser.parse_args()

    # return args
    return args
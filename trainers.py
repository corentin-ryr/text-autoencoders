from datetime import datetime
import os

import torch
import torch.nn.functional as F
from utils import noisy, compute_term_frequency, Tokenizer, frange_cycle_sigmoid

from mixers.trainers.abstractTrainers import Trainer
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR



class AutoencoderTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        device,
        ignore_padding: bool,
        vocab: Tokenizer = None,
        traindataset: Dataset = None,
        testdataset: Dataset = None,
        evaldataset: Dataset = None,
        batch_size: int = 256,
        epochs=150,
        collate_fn=None,
        lr=0.001,
        shuffle: bool = True,
        denoising: bool = False,
        displayLoss: bool = True,
        runName: str = None,
    ) -> None:
        """Class to train a autoencoder model. It works for recurrent and non recurrent models. It can apply the variational and adversarial training procedures.

        Args:
            model (nn.Module): The torch model of the autoencoder (must have a forward method that returns the predictions, mu and ligvar) and an encode method that returns mu and logvar (the mean and the log variance of the latent space)
            device (_type_): The device on which to train
            vocab (_type_): The vocabulary class
            rnn (bool): Wether the model is recurrent or not (the input)
            traindataset (Dataset, optional): Training dataset. The data is a tensor of indices in the vocabulary. Defaults to None.
            testdataset (Dataset, optional): Testing dataset.The data is a tensor of indices in the vocabulary. Defaults to None.
            evaldataset (Dataset, optional): Evaluation dataset. The data is a tensor of indices in the vocabulary. Defaults to None.
            batch_size (int, optional): The batch size. Defaults to 256.
            epochs (int, optional): The number of epochs. Defaults to 150.
            collate_fn (_type_, optional): The collate function for the dataloader. The output must fit the input of the network. Defaults to None.
            lr (float, optional): The initial learning rate. Defaults to 0.001.
            shuffle (bool, optional): Wether to shuffle the dataset at each epoch. Defaults to True.
            denoising (bool, optional): Wether to apply noise to the input. Defaults to False.
        """
        super().__init__(model, device, traindataset, testdataset, evaldataset, batch_size, collate_fn, shuffle=shuffle, num_workers=0)

        self.lr = lr
        self.vocab = vocab
        self.epochs = epochs
        self.ignore_padding = ignore_padding
        self.denoising = denoising
        self.display_loss = displayLoss
        self.metrics = {}


        runName = runName if runName else datetime.today().strftime("%Y-%m-%d-%H-%M")
        self.runDirectory = os.path.join("outputs", "runs", runName)
        self.writer = SummaryWriter(log_dir=self.runDirectory)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=self.epochs)

    def train(self):
        super().train()

        weights = compute_term_frequency(self.trainloader, self.vocab.vocab_length).to(self.device) if self.vocab else None
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0) if self.ignore_padding else nn.CrossEntropyLoss(weight=weights)

        for epoch in range(self.epochs):
            total_loss = 0
            self.model.train()

            for idx, data in enumerate(tqdm(self.trainloader, desc="Epoch {}".format(epoch))):
                source, _ = data
                target = torch.clone(source)

                if self.denoising and self.vocab:
                    drop_prob = 0
                    source = noisy(self.vocab, source, drop_prob, 0, 0.2, 0)

                    if drop_prob > 0:
                        lengths = torch.argwhere(source == self.vocab.eos)[:, 1] + 1
                        sortingIndices = torch.argsort(lengths, descending=True)
                        lengths = lengths[sortingIndices]
                        source = source[sortingIndices]
                        target = target[sortingIndices]

                source = source.to(self.device).detach()
                target = target.to(self.device).detach()

                self.optimizer.zero_grad()

                predictions = self.model(source, target)

                loss = criterion(torch.flatten(predictions, end_dim=1), torch.flatten(target))
                loss.backward()

                self.optimizer.step()

                total_loss += loss.item() * len(source)

            print(datetime.now().strftime("%H:%M:%S"), "Epoch", epoch, "loss:", total_loss / len(self.trainloader.dataset))

            self.metrics["Loss reconstruction"] = total_loss / len(self.trainloader.dataset)
            self.metrics["Learning rate"] = self.scheduler.get_last_lr()[0]
            self._handle_metrics(epoch)

            self.scheduler.step()
            if epoch % 100 == 99:
                self.save_model(checkpoint=True)
                self.validate(validateOnTrain=True, epoch=epoch)


    def validate(self, validateOnTrain: bool = False, epoch:int=None):

        correct_sentences, incorrect_sentences = 0, 0
        correct_symbols, all_symbols_target = 0, 0

        self.model.eval()
        for idx, data in enumerate(tqdm(self.trainloader if validateOnTrain else self.testloader, desc="Validation")):
            source, _ = data
            source = source.to(self.device)
            target = torch.clone(source).detach()

            z = self.encode(source)
            predicted_sentences = self.decode(z)

            symbolsBatch = torch.count_nonzero(target).item() if self.ignore_padding else target.numel()
            nbNonZeros = (
                torch.count_nonzero(torch.masked_select(torch.abs(predicted_sentences - target), target != 0)).item()
                if self.ignore_padding
                else torch.count_nonzero(predicted_sentences - target).item()
            )
            all_symbols_target += symbolsBatch
            correct_symbols += symbolsBatch - nbNonZeros

            for idx, sentence in enumerate(predicted_sentences):
                if torch.equal(sentence, target[idx]):
                    correct_sentences += 1
                else:
                    incorrect_sentences += 1

        print(f"Number of symbols: {all_symbols_target} | Number of sentences: {len(self.trainloader.dataset)} \n")

        print(f"Correctly predicted words    : {correct_symbols} ({(correct_symbols / all_symbols_target)*100:.2f}% of all symbols)")
        print(
            f"Correctly predicted sentences  : {correct_sentences} ({(correct_sentences / len(self.trainloader.dataset))*100:.2f}% of all sentences)"
        )

        if epoch:
            self.writer.add_scalar("Correct symbol ", correct_symbols / all_symbols_target, epoch)
            self.writer.add_scalar("Correct sentences ", correct_sentences / (correct_sentences + incorrect_sentences), epoch)



    def encode(self, input):
        self.model.eval()

        z = self.model.encode(input)

        if len(z.shape) > 2 and z.shape[0] > 1:
            z = torch.cat([z[i] for i in range(z.shape[0])], dim=1).unsqueeze(0)
        return z.squeeze(0).detach()

    def decode(self, z):
        self.model.eval()

        predictions = self.model.decode(z)
        _, predicted_tensor = predictions.topk(1)
        return predicted_tensor.squeeze()

    def save_model(self, object_to_save=None, savePath="checkpoints", checkpoint: bool = False):
        if not object_to_save:
            if checkpoint:
                object_to_save = {"model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict()}
            else:
                object_to_save = self.model.state_dict()

        super().save_model(object_to_save, savePath, checkpoint=checkpoint)

    def load_model(self, load_path):
        # Check if it is a checkpoint and if so setup the optimizer
        checkpoint = torch.load(load_path)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

    def _handle_metrics(self, epoch):
        metricsString = " | ".join([f"{key}:  {self.metrics[key]}" for key in self.metrics])
        print(datetime.now().strftime("%H:%M:%S"), "Epoch", epoch, metricsString)
        for key in self.metrics:
            self.writer.add_scalar(key, self.metrics[key], epoch)

    def add_image_tensorboard(self, tag, figure):
        self.writer.add_figure(tag, figure)
        self.writer.flush()

class AdversarialAutoEncoderTrainer(AutoencoderTrainer):
    def __init__(
        self,
        model: nn.Module,
        device,
        vocab,
        ignore_padding: bool,
        traindataset: Dataset = None,
        testdataset: Dataset = None,
        evaldataset: Dataset = None,
        batch_size: int = 256,
        epochs=150,
        collate_fn=None,
        lr=0.001,
        shuffle: bool = True,
        denoising: bool = False,
        displayLoss: bool = False,
    ) -> None:
        """Class to train a autoencoder model. It works for recurrent and non recurrent models. It can apply the variational and adversarial training procedures.

        Args:
            model (nn.Module): The torch model of the autoencoder (must have a forward method that returns the predictions, mu and ligvar) and an encode method that returns mu and logvar (the mean and the log variance of the latent space)
            device (_type_): The device on which to train
            vocab (_type_): The vocabulary class
            rnn (bool): Wether the model is recurrent or not (the input)
            traindataset (Dataset, optional): Training dataset. The data is a tensor of indices in the vocabulary. Defaults to None.
            testdataset (Dataset, optional): Testing dataset.The data is a tensor of indices in the vocabulary. Defaults to None.
            evaldataset (Dataset, optional): Evaluation dataset. The data is a tensor of indices in the vocabulary. Defaults to None.
            batch_size (int, optional): The batch size. Defaults to 256.
            epochs (int, optional): The number of epochs. Defaults to 150.
            collate_fn (_type_, optional): The collate function for the dataloader. The output must fit the input of the network. Defaults to None.
            lr (float, optional): The initial learning rate. Defaults to 0.001.
            shuffle (bool, optional): Wether to shuffle the dataset at each epoch. Defaults to True.
            procedure (str, optional): Can be standard, variational or adversarial. Defaults to "standard".
            denoising (bool, optional): Wether to apply noise to the input. Defaults to False.
        """
        super().__init__(
            model=model,
            device=device,
            vocab=vocab,
            ignore_padding=ignore_padding,
            traindataset=traindataset,
            testdataset=testdataset,
            evaldataset=evaldataset,
            batch_size=batch_size,
            epochs=epochs,
            collate_fn=collate_fn,
            lr=lr,
            shuffle=shuffle,
            denoising=denoising,
            displayLoss=displayLoss,
        )

        self.z_dim = self.model.latentSize
        self.D = nn.Sequential(nn.Linear(self.z_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()).to(device)

        self.optD = optim.Adam(self.D.parameters(), lr=self.lr)
        self.schedulerD = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=self.epochs)

    def train(self):

        weights = compute_term_frequency(self.trainloader, self.vocab.vocab_length).to(self.device) if self.vocab else None
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0) if self.ignore_padding else nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            total_loss_ae = 0
            total_loss_d = 0
            self.model.train()

            for idx, data in enumerate(tqdm(self.trainloader, desc="Epoch {}".format(epoch))):
                source, _ = data
                target = torch.clone(source).detach()

                if self.denoising:
                    source = noisy(self.vocab, source, 0, 0, 0.2, 0)
                source = source.to(self.device).detach()
                target = target.to(self.device).detach()

                z = self.model.encode(source)
                predictions = self.model.decode(z, target)

                zn = torch.randn_like(z)
                zeros = torch.zeros(len(z), 1, device=self.device)
                ones = torch.ones(len(z), 1, device=self.device)

                loss_d = F.binary_cross_entropy(self.D(z.detach()), zeros) + F.binary_cross_entropy(self.D(zn), ones)
                loss_adv = F.binary_cross_entropy(self.D(z), ones)
                loss_rec = criterion(torch.flatten(predictions, end_dim=1), torch.flatten(target))
                loss: torch.Tensor = loss_rec + 0.1 * loss_adv

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.optD.zero_grad()
                loss_d.backward()
                self.optD.step()

                total_loss_ae += loss_rec.item() * len(source)
                total_loss_d += loss_d.item() * len(source)

            self.metrics["Loss reconstruction"] = total_loss_ae / len(self.trainloader.dataset)
            self.metrics["Loss discriminator"] = total_loss_d / len(self.trainloader.dataset)
            self.metrics["Learning rate"] = self.scheduler.get_last_lr()[0]
            self._handle_metrics(epoch)

            self.scheduler.step()
            self.schedulerD.step()
            if epoch % 100 == 99:
                self.save_model(checkpoint=True)
                self.validate(validateOnTrain=True, epoch=epoch)
        
    def encode(self, input):
        self.model.eval()

        z = self.model.encode(input)
        return z.squeeze(0).detach()

class VariationalAutoEncoderTrainer(AutoencoderTrainer):
    def __init__(
        self,
        model: nn.Module,
        device,
        ignore_padding: bool,
        vocab=None,
        traindataset: Dataset = None,
        testdataset: Dataset = None,
        evaldataset: Dataset = None,
        batch_size: int = 256,
        epochs=150,
        collate_fn=None,
        lr=0.001,
        shuffle: bool = True,
        denoising: bool = False,
        displayLoss: bool = False,
    ) -> None:
        """Class to train a autoencoder model. It works for recurrent and non recurrent models. It can apply the variational and adversarial training procedures.

        Args:
            model (nn.Module): The torch model of the autoencoder (must have a forward method that returns the predictions, mu and ligvar) and an encode method that returns mu and logvar (the mean and the log variance of the latent space)
            device (_type_): The device on which to train
            vocab (_type_): The vocabulary class
            rnn (bool): Wether the model is recurrent or not (the input)
            traindataset (Dataset, optional): Training dataset. The data is a tensor of indices in the vocabulary. Defaults to None.
            testdataset (Dataset, optional): Testing dataset.The data is a tensor of indices in the vocabulary. Defaults to None.
            evaldataset (Dataset, optional): Evaluation dataset. The data is a tensor of indices in the vocabulary. Defaults to None.
            batch_size (int, optional): The batch size. Defaults to 256.
            epochs (int, optional): The number of epochs. Defaults to 150.
            collate_fn (_type_, optional): The collate function for the dataloader. The output must fit the input of the network. Defaults to None.
            lr (float, optional): The initial learning rate. Defaults to 0.001.
            shuffle (bool, optional): Wether to shuffle the dataset at each epoch. Defaults to True.
            procedure (str, optional): Can be standard, variational or adversarial. Defaults to "standard".
            denoising (bool, optional): Wether to apply noise to the input. Defaults to False.
        """
        super().__init__(
            model=model,
            device=device,
            vocab=vocab,
            ignore_padding=ignore_padding,
            traindataset=traindataset,
            testdataset=testdataset,
            evaldataset=evaldataset,
            batch_size=batch_size,
            epochs=epochs,
            collate_fn=collate_fn,
            lr=lr,
            shuffle=shuffle,
            denoising=denoising,
            displayLoss=displayLoss,
        )

        self.sigmoidAnnealing = frange_cycle_sigmoid(0, 1, self.epochs, n_cycle=1, ratio=0.7) * 0.1

        import matplotlib.pyplot as plt

        plt.plot(self.sigmoidAnnealing)
        plt.grid(visible=True)
        plt.show()

    def train(self):
        weights = compute_term_frequency(self.trainloader, self.vocab.vocab_length).to(self.device) if self.vocab else None
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0) if self.ignore_padding else nn.CrossEntropyLoss(weight=weights)

        for epoch in range(self.epochs):
            total_loss_rec = 0
            total_loss_kl = 0
            self.model.train()

            for idx, data in enumerate(tqdm(self.trainloader, desc="Epoch {}".format(epoch))):
                source, _ = data
                target = torch.clone(source).detach()

                if self.denoising and self.vocab:
                    source = noisy(self.vocab, source, 0, 0, 0.2, 0)

                source = source.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()

                predictions, mu, logvar = self.model(source, target)

                lambdaKL = self.sigmoidAnnealing[epoch]

                loss_rec = criterion(torch.flatten(predictions, end_dim=1), torch.flatten(target))
                loss_kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
                (loss_rec + lambdaKL * loss_kl).backward()
                self.optimizer.step()

                total_loss_rec += loss_rec.item() * len(source)
                total_loss_kl += loss_kl.item() * len(source)

            self.metrics["Loss reconstruction"] = total_loss_rec / len(self.trainloader.dataset)
            self.metrics["Loss variational"] = total_loss_kl / len(self.trainloader.dataset)
            self.metrics["Learning rate"] = self.scheduler.get_last_lr()[0]
            self._handle_metrics(epoch)

            self.scheduler.step()
            if epoch % 100 == 99:
                self.save_model(checkpoint=True)
                self.validate(validateOnTrain=True)


    def encode(self, input):
        self.model.eval()

        _, mu, _ = self.model.encode(input)
        return mu.squeeze(0).detach()

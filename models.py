import math

import torch
from torch import nn
from mixers.models.MLPMixer.mlpmixer import MixerBlock

class AutoEncoderMixerToSeq(nn.Module):
    def __init__(self, vocab, sentenceLength=100, embeddingSize=50, mixerHiddenSize=256, decoderHiddenSize=512, num_layers=3, latentSize=32) -> None:
        super().__init__()

        self.mixerHiddenSize = mixerHiddenSize
        self.decoderHiddenSize = decoderHiddenSize
        self.sentenceLength = sentenceLength
        self.latentSize = latentSize
        self.vocab = vocab

        self.embedding = nn.Embedding(vocab.vocab_length, embedding_dim=embeddingSize)
        self.pos_encoder = PositionalEncoding(embeddingSize, max_len=sentenceLength)

        mixerBlocks = []
        for _ in range(num_layers):
            mixerBlocks.append(MixerBlock(embeddingSize, sentenceLength, self.mixerHiddenSize, self.mixerHiddenSize))
        self.encoder = nn.Sequential(*mixerBlocks)

        self.relu = nn.ReLU()
        self.reductionLayer = nn.Linear(embeddingSize, latentSize)

        self.z2cn = nn.Linear(latentSize, self.decoderHiddenSize)
        self.z2hn = nn.Linear(latentSize, self.decoderHiddenSize)

        self.decoder = nn.LSTM(
            input_size=latentSize,
            hidden_size=self.decoderHiddenSize,
            batch_first=True,
        )

        self.proj = nn.Linear(self.decoderHiddenSize, vocab.vocab_length)

    def encode(self, x):
        wordEmb = self.embedding(x)
        wordEmb = self.pos_encoder(wordEmb)

        z = self.encoder(wordEmb)
        z = self.reductionLayer(z)
        return torch.mean(z, dim=1)

    def decode(self, z, target=None):
        hn = self.z2hn(z).unsqueeze(0)
        cn = self.z2hn(z).unsqueeze(0)

        inp = z.unsqueeze(1).repeat(1, self.sentenceLength, 1)

        predictions, _ = self.decoder(inp, (hn, cn))
        logits = self.proj(predictions.squeeze(dim=1))
        return logits



    def forward(self, source, target):
        """Feed x into the autoencoder

        Args:
            x (list[list[int]]): Batch of token indices
            target (list[list[int]], optional): Target of the autoencoder. Only set for teacher forcing. Defaults to None.
        """
        z = self.encode(source)
        symbols = self.decode(z, target)

        return symbols


class VariationalAutoEncoderMixerToSeq(AutoEncoderMixerToSeq):
    def __init__(self, vocab, sentenceLength=100, embeddingSize=50, mixerHiddenSize=256, decoderHiddenSize=512, num_layers=3, latentSize=32) -> None:
        super().__init__(
            vocab=vocab,
            sentenceLength=sentenceLength,
            embeddingSize=embeddingSize,
            mixerHiddenSize=mixerHiddenSize,
            decoderHiddenSize=decoderHiddenSize,
            num_layers=num_layers,
            latentSize=latentSize
        )

        self.z2mu = nn.Linear(latentSize, latentSize)
        self.z2logvar = nn.Linear(latentSize, latentSize)

    def encode(self, x):
        wordEmb = self.embedding(x)
        wordEmb = self.pos_encoder(wordEmb)

        z = self.encoder(wordEmb)
        z = self.reductionLayer(z)
        z = torch.mean(z, dim=1)

        mu = self.z2mu(z)
        logvar = self.z2logvar(z)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        return z, mu, logvar

    def forward(self, source, target):
        """Feed x into the autoencoder

        Args:
            x (list[list[int]]): Batch of token indices
            target (list[list[int]], optional): Target of the autoencoder. Only set for teacher forcing. Defaults to None.
        """
        z, mu, logvar = self.encode(source)
        symbols = self.decode(z, target)

        return symbols, mu, logvar


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = torch.transpose(x, 0, 1) + self.pe[: x.size(1)]
        return torch.transpose(self.dropout(x), 0, 1)

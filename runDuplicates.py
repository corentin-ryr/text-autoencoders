from mixers.utils.helper import get_device
from torch.utils.data import DataLoader

from dataset import BinaryDuplicateToyDataset
from models import AutoEncoderMixerToSeq
from trainers import AdversarialAutoencoderTrainer, AutoencoderTrainer
from utils import (NeihgborInDistance, PlotProximity, ToyVocab, get_device,
                   parse_args, simpleVizualization, SRCC)

batch_size = 128
savePath = "outputs"
sentenceLength = 50

vocab = ToyVocab(numWords=10)
trainDataset = BinaryDuplicateToyDataset(mode="sampling", vocab=vocab, numberSamples=2000, nonMatchToMatchRatio=2)


def trainingAutoencoder(args):
    device = get_device(useGPU=args.use_gpu)

    model = AutoEncoderMixerToSeq(
        vocab, sentenceLength=sentenceLength, embeddingSize=64, mixerHiddenSize=128, decoderHiddenSize=256, latentSize=2, num_layers=1
    )

    if args.training == "ae":
        trainer = AutoencoderTrainer(
            model=model,
            device=device,
            ignore_padding=False,
            traindataset=trainDataset,
            testdataset=trainDataset,
            batch_size=batch_size,
            epochs=args.num_epoch,
            lr=args.lr,
            denoising=args.use_noise,
            vocab=vocab,
        )
    if args.training == "aae":
        trainer = AdversarialAutoencoderTrainer(
            model=model,
            device=device,
            ignore_padding=False,
            traindataset=trainDataset,
            testdataset=trainDataset,
            batch_size=batch_size,
            epochs=args.num_epoch,
            lr=args.lr,
            denoising=args.use_noise,
            vocab=vocab,
        )

    if args.model_param:
        trainer.load_model(args.model_param)

    print(args)
    trainer.summarize_model()

    # trainer.train()
    # trainer.save_model()

    trainer.validate()

    simpleVizualization(trainer, DataLoader(trainDataset, batch_size), device=device)

    # vizDataset = BinaryDuplicateToyDataset(mode="pairs", vocab=vocab, numberSamples=2000)
    trainDataset.setMode("pairs")
    distancePercentages = PlotProximity(trainer, DataLoader(trainDataset, batch_size), device=device)()

    trainDataset.setMode("sampling")
    for distance in distancePercentages:
        NeihgborInDistance(trainer, DataLoader(trainDataset, batch_size), device=device)(distance)

    trainDataset.setMode("pairs")
    SRCC(trainer, DataLoader(trainDataset, batch_size))()


# run script
if __name__ == "__main__":
    args = parse_args()

    if not args.training:
        args.training = "aae"
    if not args.model_param:
        args.model_param = ""
    if not args.num_epoch:
        args.num_epoch = 5000
    if not args.lr:
        args.lr = 0.005
    if args.use_gpu is None:
        args.use_gpu = False
    if args.use_noise is None:
        args.use_noise = True

    trainingAutoencoder(args)

from mixers.utils.helper import get_device
from torch.utils.data import DataLoader

from dataset import BinaryDuplicateToyDataset
from models import AutoEncoderMixerToSeq
from trainers import AdversarialAutoencoderTrainer, AutoencoderTrainer
from utils import ToyVocab, parse_args
from visualization import NeihgborInDistance, PlotProximity, get_device, simpleVizualization, SRCC
import matplotlib.pyplot as plt

batch_size = 128
savePath = "outputs"
sentenceLength = 50

vocab = ToyVocab(numWords=10)
trainDataset = BinaryDuplicateToyDataset(mode="sampling", vocab=vocab, numberSamples=2000, nonMatchToMatchRatio=2)


def trainingAutoencoder(args):
    device = get_device(useGPU=args.use_gpu)

    if args.run_name.startswith("mixer_small"):
        model = AutoEncoderMixerToSeq(
            vocab,
            sentenceLength=sentenceLength,
            embeddingSize=4,
            mixerHiddenSize=16,
            decoderHiddenSize=256,
            latentSize=3,
            num_layers=3,
            encoderType="mixer",
        )
    if args.run_name.startswith("transformer_small"):
        model = AutoEncoderMixerToSeq(
            vocab,
            sentenceLength=sentenceLength,
            embeddingSize=16,
            mixerHiddenSize=32,
            decoderHiddenSize=256,
            latentSize=3,
            num_layers=3,
            encoderType="transformer",
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
            runName=args.run_name,
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
            runName=args.run_name,
        )

    if args.model_param:
        trainer.load_model(args.model_param)

    print(args)
    trainer.summarize_model()

    trainer.train()
    trainer.save_model()

    trainer.validate()

    vizDataset = BinaryDuplicateToyDataset(mode="sampling", vocab=vocab, numberSamples=2000)
    simpleVizualization(trainer, DataLoader(vizDataset, batch_size), device=device)

    vizDataset.setMode("pairs")
    percentages = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    distancePercentages = PlotProximity(trainer, DataLoader(vizDataset, batch_size), device=device)(percentages)

    vizDataset.setMode("sampling")
    percentageOfSamples = []
    for distance in distancePercentages:
        percentageOfSamples.append(NeihgborInDistance(trainer, DataLoader(vizDataset, batch_size), device=device)(distance))

    fig, ax = plt.subplots()
    ax.plot(percentages, percentageOfSamples)
    ax.set_title("Percentage of sample for different recall values.")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Percentage of samples")
    trainer.add_image_tensorboard("percentage-recall", fig)

    vizDataset.setMode("pairs")
    SRCC(trainer, DataLoader(vizDataset, batch_size))()

    trainer.writer.flush()


# run script
if __name__ == "__main__":
    args = parse_args()

    if not args.training:
        args.training = "aae"
    if not args.model_param:
        args.model_param = ""
    if not args.num_epoch:
        args.num_epoch = 3000
    if not args.lr:
        args.lr = 0.003
    if args.use_gpu is None:
        args.use_gpu = False
    if args.use_noise is None:
        args.use_noise = True
    if args.run_name is None:
        args.run_name = "mixer_small_6"

    trainingAutoencoder(args)

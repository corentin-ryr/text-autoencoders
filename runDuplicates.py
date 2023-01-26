from mixers.utils.helper import get_device
from torch.utils.data import DataLoader

from dataset import BinaryDuplicateToyDataset
from models import AutoEncoderMixerToSeq
from trainers import AdversarialAutoencoderTrainer, AutoencoderTrainer
from utils import NeihgborInDistance, PlotProximity, ToyVocab, get_device, parse_args, simpleVizualization

batch_size = 128
savePath = "outputs"
sentenceLength = 50

vocab = ToyVocab(numWords=10)
trainDataset = BinaryDuplicateToyDataset(mode="sampling", vocab=vocab, numberSamples=2000)



# model = AutoEncoderMixerToSeq(
#     vocab, sentenceLength=sentenceLength, embeddingSize=4, mixerHiddenSize=128, decoderHiddenSize=256, latentSize=2, num_layers=1, encoderType="mixer"
# )
model = AutoEncoderMixerToSeq(
    vocab, sentenceLength=sentenceLength, embeddingSize=16, mixerHiddenSize=2048, decoderHiddenSize=256, latentSize=2, num_layers=1, encoderType="transformer"
)


def classicAutoencoder(args):
    device = get_device(useGPU=args.use_gpu)

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
        displayLoss=True,
        vocab=vocab,
    )

    if args.model_param:
        trainer.load_model(args.model_param)

    print(args)
    trainer.summarize_model()

    trainer.train()
    trainer.save_model()

    trainer.validate()

    simpleVizualization(trainer, DataLoader(BinaryDuplicateToyDataset(mode="sampling", vocab=vocab, numberSamples=2000), batch_size), device=device)

    vizDataset = BinaryDuplicateToyDataset(mode="pairs", vocab=vocab, numberSamples=2000)
    distance90Percent = PlotProximity(trainer, DataLoader(vizDataset, batch_size), showFig=True, device=device)()

    vizDataset.setMode("sampling")
    NeihgborInDistance(trainer, DataLoader(vizDataset, batch_size), showFig=True, device=device)(distance90Percent)


def adversarialAutoencoder(args):
    device = get_device(useGPU=args.use_gpu)

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
        displayLoss=True,
        vocab=vocab,
    )

    if args.model_param:
        trainer.load_model(args.model_param)

    print(args)
    trainer.summarize_model()

    trainer.train()
    trainer.save_model()

    trainer.validate()

    simpleVizualization(trainer, DataLoader(BinaryDuplicateToyDataset(mode="sampling", vocab=vocab, numberSamples=200), batch_size), device=device)

    vizDataset = BinaryDuplicateToyDataset(mode="pairs", vocab=vocab, numberSamples=2000)
    distance90Percent = PlotProximity(trainer, DataLoader(vizDataset, batch_size), showFig=True, device=device)()

    vizDataset.setMode("sampling")
    NeihgborInDistance(trainer, DataLoader(vizDataset, batch_size), showFig=True, device=device)(distance90Percent)


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

    if args.training == "ae":
        classicAutoencoder(args)

    elif args.training == "aae":
        adversarialAutoencoder(args)

    else:
        print("Please select from the three training procedures: ae, vae and aae")

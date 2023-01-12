from mixers.utils.helper import get_device

from dataset import BinaryToyDataset
from trainers import AutoencoderTrainer, VariationalAutoEncoderTrainer, AdversarialAutoEncoderTrainer
from models import AutoEncoderMixerToSeq, VariationalAutoEncoderMixerToSeq
from utils import ToyVocab
from utils import parse_args
from utils import pcaOn2Dim, tsneOn2Dim, simpleVizualization, get_device

from torch.utils.data import DataLoader


batch_size = 128
savePath = "outputs"
sentenceLength = 50

vocab = ToyVocab()
trainDataset = BinaryToyDataset(samplePerClass=100, sampleLength=sentenceLength)


def classicAutoencoder(args):
    device = get_device(useGPU=args.use_gpu)

    model = AutoEncoderMixerToSeq(
        vocab, sentenceLength=sentenceLength, embeddingSize=16, mixerHiddenSize=128, decoderHiddenSize=256, latentSize=2, num_layers=1
    )

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
    trainer.save_model(savePath=savePath)

    trainer.validate()

    # tsneOn2Dim(trainer, DataLoader(trainDataset, batch_size), showFig=False, savePath=savePath, device=device)
    # pcaOn2Dim(trainer, DataLoader(trainDataset, batch_size), showFig=False, savePath=savePath, device=device)
    simpleVizualization(trainer, DataLoader(trainDataset, batch_size), showFig=False, savePath=savePath, device=device)


def variationalAutoencoder(args):
    device = get_device(useGPU=args.use_gpu)

    model = VariationalAutoEncoderMixerToSeq(
        vocab, sentenceLength=sentenceLength, embeddingSize=4, mixerHiddenSize=128, decoderHiddenSize=256, latentSize=2, num_layers=1
    )

    trainer = VariationalAutoEncoderTrainer(
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

    if args.model_param: trainer.load_model(args.model_param)

    print(args)
    trainer.summarize_model()

    trainer.train()
    trainer.save_model(savePath=savePath)

    trainer.validate()

    # tsneOn2Dim(trainer, DataLoader(trainDataset, batch_size), showFig=True, savePath=savePath, device=device)
    # pcaOn2Dim(trainer, DataLoader(trainDataset, batch_size), showFig=True, savePath=savePath, device=device)
    simpleVizualization(trainer, DataLoader(trainDataset, batch_size), showFig=True, savePath=savePath, device=device)


def adversarialAutoencoder(args):
    device = get_device(useGPU=args.use_gpu)

    model = AutoEncoderMixerToSeq(
        vocab, sentenceLength=sentenceLength, embeddingSize=4, mixerHiddenSize=128, decoderHiddenSize=256, latentSize=2, num_layers=1
    )

    trainer = AdversarialAutoEncoderTrainer(
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
    trainer.save_model(savePath=savePath)

    trainer.validate()

    # tsneOn2Dim(trainer, DataLoader(trainDataset, batch_size), showFig=False, savePath=savePath, device=device)
    # pcaOn2Dim(trainer, DataLoader(trainDataset, batch_size), showFig=False, savePath=savePath, device=device)
    simpleVizualization(trainer, DataLoader(trainDataset, batch_size), showFig=False, savePath=savePath, device=device)


# run script
if __name__ == "__main__":
    args = parse_args()

    if not args.training:
        args.training = "aae"
    if not args.model_param:
        args.model_param = ""
    if not args.num_epoch:
        args.num_epoch = 4000
    if not args.lr:
        args.lr = 0.003
    if args.use_gpu is None:
        args.use_gpu = False
    if args.use_noise is None:
        args.use_noise = True

    if args.training == "ae":
        classicAutoencoder(args)

    elif args.training == "vae":
        variationalAutoencoder(args)

    elif args.training == "aae":
        adversarialAutoencoder(args)

    else:
        print("Please select from the three training procedures: ae, vae and aae")

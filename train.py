import os
import argparse
import segmentation_models_pytorch as smp
import torch

from omegaconf import OmegaConf

from src.data_loader import DataGenerator
from src.training import Trainer


parser = argparse.ArgumentParser(description="Train a segmentation model")
parser.add_argument("--config",
                    type=str,
                    default="config.yaml",
                    help="Path to the config file")
args = parser.parse_args()


def main(config_path):
    config = OmegaConf.load(config_path)
    batch_size = config.training.batch_size
    CLASSES = config.training.classes
    encoder_name = config.training.encoder_name
    train_dir = os.path.join('data', 'train', 'train')
    val_dir = os.path.join('data', 'valid', 'valid')

    train_loader = DataGenerator(train_dir,
                                 phase="train",
                                 batch_size=batch_size,
                                 shuffle=True).load_data()
    val_loader = DataGenerator(val_dir,
                               phase="val",
                               batch_size=len(os.listdir(val_dir)),
                               shuffle=False).load_data()

    # try:
    #     device = torch.device("cuda:0")
    #     print('run with gpu')
    # except:
    device = torch.device("cpu")
    print(f"Using {device} device")

    activation = 'sigmoid' if CLASSES == 1 else 'softmax'
    model = smp.Unet(
        encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=CLASSES,                     # model output classes (number of classes in your dataset)
        activation=activation
    ).to(device)
    model = model.float()

    epoch_num = config.training.epochs
    learning_rate = config.training.initial_learning_rate
    loss = config.training.loss

    steps = epoch_num  #int(len(train_slices)/batch_size)+1

    if loss == "cross_entropy":
        loss_function = torch.nn.CrossEntropyLoss()
    elif loss == "focal_loss":
        loss_function = smp.losses.FocalLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    trainer = Trainer(
        train_generator=train_loader,
        val_generator=val_loader,
        num_classes=CLASSES,
        model=model,
        device=device,
        optimizer=optimizer,
        loss_function=loss_function,
        scheduler=scheduler,
        run_dir=config.dirs.run
    )

    trainer.fit(epochs=epoch_num)


if __name__ == '__main__':
    main(args.config)

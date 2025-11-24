import segmentation_models_pytorch as smp

from omegaconf import OmegaConf


class ModelBuilder:
    def __init__(self, config: OmegaConf):
        self.config = config

    def build_model(self):
        # Build and return the model based on the configuration
        classes = self.config.dataset.classes
        encoder_name = self.config.training.encoder_name
        activation = 'sigmoid' if len(classes) == 1 else 'softmax'
        model = smp.Unet(
            encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=len(classes),                     # model output classes (number of classes in your dataset)
            activation=activation
        )
        return model

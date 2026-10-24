import segmentation_models_pytorch as smp


def create_model(pretrained=False):
    encoder_weights = "imagenet" if pretrained else None

    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=4,
        activation=None,
    )
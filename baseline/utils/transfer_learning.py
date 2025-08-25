import logging
import torch


def freeze_encoder(model):
    """Freeze the full ResNet encoder"""
    for param in model.encoder.parameters():
        param.requires_grad = False
    logging.info(" Encoder frozen (training head + classifier only)")


def unfreeze_last_layers(model, num_blocks=2):
    """
    Unfreeze the last `num_blocks` residual blocks of encoder.layer4.
    Since encoder is Sequential(..., layer4, avgpool),
    we need encoder[-2] to access layer4.
    """
    layer4 = model.encoder[-2]

    # Take last N residual blocks inside layer4
    for block in list(layer4.children())[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    logging.info(
        f"Unfroze last {num_blocks} residual blocks of encoder.layer4"
    )


def get_optimizer(model, base_lr=1e-3, encoder_lr=1e-4, weight_decay=1e-4):
    # Separate parameter groups
    head_params = list(model.head.parameters()) + list(
        model.classifier.parameters()
    )
    encoder_params = list(model.encoder.parameters())

    optimizer = torch.optim.Adam(
        [
            {"params": head_params, "lr": base_lr},
            {"params": encoder_params, "lr": encoder_lr},
        ],
        weight_decay=weight_decay,
    )

    return optimizer

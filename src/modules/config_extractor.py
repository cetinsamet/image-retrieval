import torchvision


# Config for the models that are supported by the extractor
MODEL_CONFIG = {
    "resnet18": {
        "weights": torchvision.models.ResNet18_Weights.DEFAULT,
        "model": torchvision.models.resnet18,
        "feat_layer": "flatten",
        "feat_dims": 512,
    },
    "resnet34": {
        "weights": torchvision.models.ResNet34_Weights.DEFAULT,
        "model": torchvision.models.resnet34,
        "feat_layer": "flatten",
        "feat_dims": 512,
    },
    "resnet50": {
        "weights": torchvision.models.ResNet50_Weights.DEFAULT,
        "model": torchvision.models.resnet50,
        "feat_layer": "flatten",
        "feat_dims": 2048,
    },
    "resnet101": {
        "weights": torchvision.models.ResNet101_Weights.DEFAULT,
        "model": torchvision.models.resnet101,
        "feat_layer": "flatten",
        "feat_dims": 2048,
    },
    "resnet152": {
        "weights": torchvision.models.ResNet152_Weights.DEFAULT,
        "model": torchvision.models.resnet152,
        "feat_layer": "flatten",
        "feat_dims": 2048,
    },
    "vit_b_16": {
        "weights": torchvision.models.ViT_B_16_Weights.DEFAULT,
        "model": torchvision.models.vit_b_16,
        "feat_layer": "getitem_5",
        "feat_dims": 768,
    },
    "vit_b_32": {
        "weights": torchvision.models.ViT_B_32_Weights.DEFAULT,
        "model": torchvision.models.vit_b_32,
        "feat_layer": "getitem_5",
        "feat_dims": 768,
    },
    "vit_l_16": {
        "weights": torchvision.models.ViT_L_16_Weights.DEFAULT,
        "model": torchvision.models.vit_l_16,
        "feat_layer": "getitem_5",
        "feat_dims": 1024,
    },
    "vit_l_32": {
        "weights": torchvision.models.ViT_L_32_Weights.DEFAULT,
        "model": torchvision.models.vit_l_32,
        "feat_layer": "getitem_5",
        "feat_dims": 1024,
    },
    "vit_h_14": {
        "weights": torchvision.models.ViT_H_14_Weights.DEFAULT,
        "model": torchvision.models.vit_h_14,
        "feat_layer": "getitem_5",
        "feat_dims": 1280,
    },
}

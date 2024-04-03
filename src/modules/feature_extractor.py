import torchvision.models.feature_extraction
import torchvision
import os

from .config_extractor import MODEL_CONFIG

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class FeatureExtractor:
    """Class for extracting features from images using a pre-trained model"""

    def __init__(self, base_model):
        # set the base model
        self.base_model = base_model
        # get the number of features
        self.feat_dims = MODEL_CONFIG[base_model]["feat_dims"]
        # initialize the image transformations
        self.model, self.transforms = self.init_model(base_model)
        self.model.eval()  # set the model to evaluation mode

    def init_model(self, base_model):
        """Initialize the  model for feature extraction

        Args:
            base_model: str, the name of the base model

        Returns:
            model: torch.nn.Module, the feature extraction model
            transforms: torchvision.transforms.Compose, the image transformations
        """
        if base_model not in MODEL_CONFIG:
            raise ValueError(f"Invalid base model: {base_model}")

        # get the model and weights
        weights = MODEL_CONFIG[base_model]["weights"]
        model = torchvision.models.feature_extraction.create_feature_extractor(
            MODEL_CONFIG[base_model]["model"](weights=weights),
            [MODEL_CONFIG[base_model]["feat_layer"]],
        )
        # get the image transformations
        transforms = weights.transforms()
        return model, transforms

    def extract_features(self, img):
        """Extract features from an image

        Args:
            img: PIL.Image, the input image

        Returns:
            output: torch.Tensor, the extracted features
        """
        # apply transformations
        x = self.transforms(img)
        # add batch dimension
        x = x.unsqueeze(0)
        # output now has the features corresponding to input x
        output = self.model(x)[MODEL_CONFIG[self.base_model]["feat_layer"]]
        return output

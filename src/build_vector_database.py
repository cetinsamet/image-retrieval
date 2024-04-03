# Description:
#   This script is used to build the vector database for the images in the dataset.
#   The script uses the FeatureExtractor class to extract the features from the images and saves them to a Faiss index.
#
# Usage:
#   To use this script, you can run the following commands:
#       python3 build_vector_database.py
#       python3 build_vector_database.py --feat_extractor vit_l_32
#       python3 build_vector_database.py --feat_extractor resnet101
#
from tqdm import tqdm
import argparse
import faiss
import torch
import PIL
import os

from modules import FeatureExtractor
from config import *


def main(args=None):
    # initialize the feature extractor with the base model specified in the arguments
    feature_extractor = FeatureExtractor(base_model=args.feat_extractor)
    # initialize the vector database indexing
    index = faiss.IndexFlatIP(feature_extractor.feat_dims)
    # get the list of images in sorted order
    image_list = sorted(os.listdir(IMAGES_DIR))

    with torch.no_grad():
        # iterate over the images and add their extracted features to the index
        for img_filename in tqdm(image_list):
            # load image
            img = PIL.Image.open(os.path.join(IMAGES_DIR, img_filename)).convert("RGB")
            # extract features
            output = feature_extractor.extract_features(img)
            # keep only batch dimension
            output = output.view(output.size(0), -1)
            # normalize the output since we are using the inner product as the similarity measure (cosine similarity)
            output = output / output.norm(p=2, dim=1, keepdim=True)
            # add to the index
            index.add(output.numpy())

    # save the index
    index_filepath = os.path.join(DATA_DIR, f"db_{args.feat_extractor}.index")
    faiss.write_index(index, index_filepath)


if __name__ == "__main__":
    # parse arguments
    args = argparse.ArgumentParser()
    args.add_argument(
        "--feat_extractor",
        type=str,
        default="vit_l_32",
        choices=FEATURE_EXTRACTOR_MODELS,
    )
    args = args.parse_args()

    # run the main function
    main(args)

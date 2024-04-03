# Description:
#   This script is used to query the index for similar images to a set of random images.
#   The script uses the FeatureExtractor class to extract the features from the images and the Faiss index to search for similar images.
#
# Usage:
#
#   To use this script, you can run the following commands: (You MUST define a feat_extractor since indexings are different for each model)
#       python3 search_query.py --feat_extractor resnet50
#       python3 search_query.py --feat_extractor resnet101
#       python3 search_query.py --feat_extractor resnet50 --n 5
#       python3 search_query.py --feat_extractor resnet50 --k 20
#       python3 search_query.py --feat_extractor resnet50 --n 10 --k 12
#
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import faiss
import PIL
import os

from modules import FeatureExtractor
from config import *


def select_random_images(n, image_list):
    """Select n random images from the image list.

    Args:
        n (int): The number of images to select.
        image_list (list[str]): The list of image file names.

    Returns:
        list[PIL.Image]: The list of selected images.
    """
    selected_indices = np.random.randint(len(image_list), size=n)
    img_filenames = [image_list[i] for i in selected_indices]
    images = [
        PIL.Image.open(os.path.join(IMAGES_DIR, img_filename))
        for img_filename in img_filenames
    ]
    return images


def plot_query_results(query_img, similar_imgs, distances, out_filepath):
    """Plot the query image and the similar images side by side. Save the plot to the specified file path.

    Args:
        query_img (PIL.Image): The query image.
        similar_imgs (list[PIL.Image]): The list of similar images.
        distances (list[float]): The list of distances of the similar images.
        out_filepath (str): The file path to save the plot.

    Returns:
        None
    """
    # initialize the figure
    fig, axes = plt.subplots(3, args.k // 2, figsize=(20, 10))
    # plot the query image
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title("Query Image")
    axes[0, 0].axis("off")
    # do not draw the remaining pots in the first row
    for i in range(1, args.k // 2):
        axes[0, i].axis("off")
    # plot the similar images
    for i, (img, dist) in enumerate(zip(similar_imgs, distances)):
        axes[i // (args.k // 2) + 1, i % (args.k // 2)].imshow(img)
        axes[i // (args.k // 2) + 1, i % (args.k // 2)].set_title(f"{dist:.4f}")
        axes[i // (args.k // 2) + 1, i % (args.k // 2)].axis("off")
    # remove the remaining axes
    plt.tight_layout()
    # save the plot
    plt.savefig(out_filepath, bbox_inches="tight", dpi=200)


def main(args=None):

    # set the random seed for reproducibility
    np.random.seed(args.seed)

    # load the vector database index
    index_filepath = os.path.join(DATA_DIR, f"db_{args.feat_extractor}.index")
    index = faiss.read_index(index_filepath)

    # initialize the feature extractor with the base model specified in the arguments
    feature_extractor = FeatureExtractor(base_model=args.feat_extractor)

    # get the list of images in sorted order since the index is built in the same order
    image_list = sorted(os.listdir(IMAGES_DIR))
    # select n random images
    query_images = select_random_images(args.n, image_list)

    with torch.no_grad():
        # iterate over the selected/query images
        for query_idx, img in enumerate(query_images, start=1):
            # output now has the features corresponding to input x
            output = feature_extractor.extract_features(img)
            # keep only batch dimension
            output = output.view(output.size(0), -1)
            # normalize
            output = output / output.norm(p=2, dim=1, keepdim=True)
            # search for similar images
            D, I = index.search(output.numpy(), args.k)

            # get the similar images
            similar_images = [
                PIL.Image.open(os.path.join(IMAGES_DIR, image_list[index]))
                for index in I[0]
            ]
            # plot the query results and save the plot
            query_results_folderpath = f"{RESULTS_DIR}/results_{args.feat_extractor}"
            os.makedirs(query_results_folderpath, exist_ok=True)
            query_results_filepath = f"{query_results_folderpath}/query_{query_idx:03}.jpg"
            plot_query_results(
                img, similar_images, D[0], out_filepath=query_results_filepath
            )


if __name__ == "__main__":
    # parse arguments
    args = argparse.ArgumentParser()
    args.add_argument(
        "--feat_extractor",
        type=str,
        choices=FEATURE_EXTRACTOR_MODELS,
        required=True,
    )
    args.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of random images to select",
    )
    args.add_argument(
        "--k",
        type=int,
        default=12,
        help="Number of similar images to retrieve",
    )
    args.add_argument(
        "--seed",
        type=int,
        default=777,
        help="Random seed for reproducibility",
    )

    args = args.parse_args()

    # run the main function
    main(args)

import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


def prepare_dataset(dataset: COCO, is_sepcial: int = 1000):
    """
    Build a DataFrame of annotations enriched with image dimensions
    and normalised bounding-box heights.

    Parameters
    ----------
    dataset : COCO
        A loaded COCO dataset object.
    is_sepcial : int, optional
        Category-filter flag (default 1000 = no filter).
          - 1000 : keep all annotations (no filtering)
          - 10   : keep all categories (explicit no-op)
          - 2    : keep only annotations whose category_id == 2
          - else : keep only annotations whose category_id == 1
    """
    # Build annotation and image DataFrames from the COCO object
    df = pd.DataFrame(dataset.anns)
    df = df.T
    images = pd.DataFrame(dataset.imgs)
    images = images.T

    cp = df.copy()
    cp["width"] = np.zeros(len(cp))
    cp["height"] = np.zeros(len(cp))
    cp["occlusion"] = np.zeros(len(cp))

    # Propagate each image's dimensions to all of its annotation rows
    for idx, row in images.iterrows():
        rows = cp.loc[df["image_id"] == row["id"]]
        cp.loc[df["image_id"] == row["id"], "width"] = row["width"] * np.ones(len(rows))
        cp.loc[df["image_id"] == row["id"], "height"] = row["height"] * np.ones(len(rows))

    # Compute absolute and normalised bounding-box heights (bbox[3] is the height)
    cp["bb_height"] = [bb[3] for bb in cp["bbox"].tolist()]
    cp["bb_height_norm"] = [bb[3] for bb in cp["bbox"].tolist()] / cp["height"]

    # Remove annotations whose normalised height exceeds 1 (physically invalid boxes)
    cp = cp.loc[cp["bb_height_norm"] <= 1]

    # Apply category filter according to the is_sepcial flag
    if is_sepcial != 1000:
        if is_sepcial == 10:
            # All categories are kept — explicit no-op for KAIST multi-class loading
            pass
        elif is_sepcial == 2:
            # Keep only the category matching the flag value (e.g. Lynred class mapping)
            cp = cp.loc[cp["category_id"] == is_sepcial]
        else:
            # Default: keep only the "person" class (category_id == 1)
            cp = cp.loc[cp["category_id"] == 1]

    return cp


def parse():
    """Parse command-line arguments for dataset annotation file paths."""
    parser = argparse.ArgumentParser(
        description="Plot normalised bounding-box height distributions across datasets."
    )
    parser.add_argument(
        "--LYNRED__path",
        type=str,
        default="/scratch2/clear/larbez/Backup_datasets/dataset_lynred_final/metadata/IR_train.json",
        help="Path to the LYNRED annotations JSON file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    LYNRED_train_ir = COCO("/scratch2/clear/larbez/Backup_datasets/dataset_lynred_final/metadata/IR_train.json")
    LYNRED_train_rgb = COCO("/scratch2/clear/larbez/Backup_datasets/dataset_lynred_final/metadata/RGB_train.json")
    LYNRED_test_rgb = COCO("/scratch2/clear/larbez/Backup_datasets/dataset_lynred_final/metadata/RGB_test.json")
    LYNRED_test_ir = COCO("/scratch2/clear/larbez/Backup_datasets/dataset_lynred_final/metadata/IR_test.json")


        
    df_lynred_ir = prepare_dataset(LYNRED_train_ir)
    df_lynred_rgb = prepare_dataset(LYNRED_train_rgb)
    df_lynred_test_rgb = prepare_dataset(LYNRED_test_rgb)
    df_lynred_test_ir = prepare_dataset(LYNRED_test_ir)



    plt.figure(figsize=[7,7])

    df_lynred_rgb['bb_height'].hist(bins=1000 ,density=True, cumulative=True,histtype="step", linewidth=2,label=f"train RGB, total = {len(df_lynred_rgb['bb_height'])}")
    df_lynred_ir['bb_height'].hist(bins=1000 ,density=True,cumulative=True,histtype="step", linewidth=2, label=f'train IR, total = {len(df_lynred_ir["bb_height"])}')
    df_lynred_test_ir['bb_height'].hist(bins=1000 ,density=True, cumulative=True,histtype="step",linewidth=2, label=f"test IR, total = {len(df_lynred_test_ir['bb_height'])}")
    df_lynred_test_rgb['bb_height'].hist(bins=1000  ,density=True,cumulative=True,histtype="step",linewidth=2, label=f'test RGB, total = {len(df_lynred_test_rgb["bb_height"])}')
    
    plt.xlim([0,200])
    plt.xlabel('Box height in px')
    plt.ylabel("Percentage of boxes in the dataset")
    plt.legend(fontsize='large')
    plt.title('Annotation size distribution across our dataset')
    plt.figsave("./Figure5B.png")
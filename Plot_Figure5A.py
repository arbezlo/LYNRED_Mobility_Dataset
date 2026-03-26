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
        "--M3FD_path",
        type=str,
        default="/scratch2/clear/larbez/RGBXFusion/dataset/M3FD/meta/m3fd-train.json",
        help="Path to the M3FD annotations JSON file.",
    )
    parser.add_argument(
        "--LYNRED_path",
        type=str,
        default="/scratch2/clear/larbez/Backup_datasets/dataset_lynred_final/metadata/IR_train.json",
        help="Path to the LYNRED annotations JSON file.",
    )
    parser.add_argument(
        "--FLIR_path",
        type=str,
        default="/scratch2/clear/larbez/Backup_datasets/FLIR_ADAS_v2/images_rgb_val/coco.json",
        help="Path to the FLIR annotations JSON file.",
    )
    parser.add_argument(
        "--FLIR_a_path",
        type=str,
        default="/scratch2/clear/larbez/RGBXFusion/dataset/FLIR_Aligned/meta/rgb/NEW_flir_train.json",
        help="Path to the FLIR Aligned annotations JSON file.",
    )
    parser.add_argument(
        "--LLVIP_path",
        type=str,
        default="/scratch2/clear/larbez/Backup_datasets/LLVIP/visible_train_.json",
        help="Path to the LLVIP annotations JSON file.",
    )
    parser.add_argument(
        "--KAIST_path",
        type=str,
        default="/scratch2/clear/larbez/datasets/KAIST_AMFD_COCO/anno/sanitized/lwir_train.json",
        help="Path to the KAIST annotations JSON file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    # Load each dataset using its corresponding annotation file path
    COCO_train = COCO(args.COCO_path)
    M3FD_train = COCO(args.M3FD_path)
    LYNRED_train = COCO(args.LYNRED_path)
    FLIR_train = COCO(args.FLIR_path)
    FLIR_a_train = COCO(args.FLIR_a_path)
    LLVIP_train = COCO(args.LLVIP_path)
    KAIST_train = COCO(args.KAIST_path)

    # Build annotation DataFrames for each dataset
    df_m3fd = prepare_dataset(M3FD_train)
    df_lynred = prepare_dataset(LYNRED_train, 2)   # category_id == 2 (Lynred class mapping)
    df_flir = prepare_dataset(FLIR_train)
    df_llvip = prepare_dataset(LLVIP_train)
    df_flira = prepare_dataset(FLIR_a_train)
    df_KAIST = prepare_dataset(KAIST_train, 10)    # is_sepcial=10: keep all categories

    # Remap all KAIST category IDs to 1 (person) for a unified cross-dataset comparison
    df_KAIST["category_id"] = np.ones(len(df_KAIST))

    # Plot the cumulative distribution of normalised bounding-box heights for each dataset : FIGURE 5.A
    plt.figure(figsize=[7,7])
    df_KAIST['bb_height'].hist(bins=1000,density=True,  cumulative=True,histtype="step",linewidth=2, label=f"KAIST, total = {len(df_KAIST['bb_height'])}")
    df_flira['bb_height'].hist(bins=1000 ,density=True, cumulative=True,histtype="step", linewidth=2, label=f"FLIR Aligned, total = {len(df_flira['bb_height'])}")
    df_flir['bb_height'].hist(bins=1000  ,density=True,cumulative=True,histtype="step", linewidth=2, label=f'FLIR, total = {len(df_flir["bb_height"])}')
    df_m3fd['bb_height'].hist(bins=1000  ,density=True,cumulative=True,histtype="step", linewidth=2, label=f"M3FD, total = {len(df_m3fd['bb_height'])}")
    df_llvip['bb_height'].hist(bins=1000, density=True,cumulative=True,histtype="step", linewidth=2, label=f'LLVIP, total = {len(df_llvip["bb_height"])}')
    df_lynred['bb_height'].hist(bins=1000,density=True, cumulative=True,histtype="step", linewidth=2, label=f'Ours, total = {len(df_lynred["bb_height"])}')
    plt.xlim([0,250])
    plt.xlabel('Box height in px')
    plt.xticks(range(0,250,25))
    plt.ylabel("Percentage of boxes in the dataset")
    plt.legend(loc=(0.65,0.6), fontsize="medium")
    plt.title('Box size distribution across existing datasets in the IR test sets')
    plt.savefig('./Figure5A.png')
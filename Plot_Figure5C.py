import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


def prepare_dataset(dataset: COCO, is_special: int = 1000):
    """
    Build a DataFrame of annotations enriched with image dimensions
    and normalised bounding-box heights.

    Parameters
    ----------
    dataset : COCO
        A loaded COCO dataset object.
    is_special : int, optional
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

    # Apply category filter according to the is_special flag
    if is_special != 1000:
        if is_special == 10:
            # All categories are kept — explicit no-op for KAIST multi-class loading
            pass
        elif is_special == 2:
            # Keep only the category matching the flag value (e.g. Lynred class mapping)
            cp = cp.loc[cp["category_id"] == is_special]
        else:
            # Default: keep only the "person" class (category_id == 1)
            cp = cp.loc[cp["category_id"] == 1]

    return cp


def parse():
    """Parse command-line arguments for dataset annotation file paths."""
    parser = argparse.ArgumentParser(
        description="Plot normalised bounding-box height distributions across IR datasets."
    )

    parser.add_argument(
        "--LYNREDIR_path",
        type=str,
        default="/scratch2/clear/larbez/Backup_datasets/dataset_lynred_final/metadata/IR_test.json",
        help="Path to the LYNRED IR annotations JSON file.",
    )
    parser.add_argument(
        "--FLIRIR_path",
        type=str,
        default="/scratch2/clear/larbez/Backup_datasets/FLIR_ADAS_v2/images_thermal_val/coco.json",
        help="Path to the FLIR IR annotations JSON file.",
    )
    parser.add_argument(
        "--FLIRIR_a_path",
        type=str,
        default="/scratch2/clear/larbez/RGBXFusion/dataset/FLIR_Aligned/meta/thermal/NEW_flir_test.json",
        help="Path to the FLIR Aligned IR annotations JSON file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    # Load each IR dataset using its corresponding annotation file path
    LYNREDIR_test = COCO(args.LYNREDIR_path)
    FLIRIR_test = COCO(args.FLIRIR_path)
    FLIRIR_a_test = COCO(args.FLIRIR_a_path)

    # Build annotation DataFrames for each dataset
    df_lynred_ir = prepare_dataset(LYNREDIR_test)
    df_flir_ir = prepare_dataset(FLIRIR_test)
    df_flir_a_ir = prepare_dataset(FLIRIR_a_test)

    # --- Lynred: filter to non-occluded persons only ---
    # Occlusion info is stored under the 'attributes' dict for this dataset
    df = df_lynred_ir.copy()
    df["occlusion_attr"] = df["attributes"].apply(lambda x: x.get("occlusion"))
    df_lynred_ir = df.loc[df["occlusion_attr"] == "none"]

    # --- FLIR IR: filter to non-occluded persons only ---
    # Occlusion info is stored under the 'extra_info' dict for this dataset
    df = df_flir_ir.copy()
    df["occlusion_attr"] = df["extra_info"].apply(lambda x: x.get("occluded"))
    df_flir_ir = df.loc[df["occlusion_attr"] == "no_(fully_visible)"]

    # Plot cumulative distributions of absolute bounding-box heights (in pixels)
    plt.figure(figsize=[6, 6])
    df_flir_a_ir["bb_height"].hist(bins=1000, density=True, cumulative=True, histtype="step", linewidth=2, color="tab:orange", label=f"FLIR Aligned, total = {len(df_flir_a_ir['bb_height'])}")
    df_flir_ir["bb_height"].hist(bins=1000,   density=True, cumulative=True, histtype="step", linewidth=2, color="tab:green",  label=f"FLIR IR, total = {len(df_flir_ir['bb_height'])}")
    df_lynred_ir["bb_height"].hist(bins=1000,  density=True, cumulative=True, histtype="step", linewidth=2, color="tab:brown",  label=f"Ours IR, total = {len(df_lynred_ir['bb_height'])}")

    plt.xlim([0, 200])
    plt.xlabel("Box height in px")

    # Vertical reference lines marking the thresholds @ 50m for teh PICO sensor and the FLIR sensor used in FLIR ADAS
    plt.axvline(29, ls=":", color="crimson")
    plt.text(31, 0.9, "Ours: PICO", color="darkred")
    plt.axvline(27, ls=":", color="crimson")
    plt.text(10, 0.9, "FLIR", color="darkred", ha="left", va="top")

    plt.ylabel("Percentage of boxes in the dataset")
    plt.legend(fontsize="large")
    plt.title('Annotation size distribution for the non occluded "person" objects')
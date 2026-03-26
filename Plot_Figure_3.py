import argparse
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Color palette (colorblind-friendly, ordered by index)
# ---------------------------------------------------------------------------
FULL_PALETTE = [
    "#88CCEE",  # 0  Light Blue
    "#CC6677",  # 1  Soft Red
    "#DDCC77",  # 2  Faded Mustard
    "#117733",  # 3  Dark Desaturated Green
    "#332288",  # 4  Indigo
    "#AA4499",  # 5  Desaturated Magenta
    "#44AA99",  # 6  Turquoise
    "#999933",  # 7  Olive
    "#882255",  # 8  Deep Burgundy
    "#661100",  # 9  Brown
    "#6699CC",  # 10 Dusty Blue
    "#888888",  # 11 Medium Grey
    "#E69F00",  # 12 Orange
    "#56B4E9",  # 13 Sky Blue
    "#009E73",  # 14 Desaturated Teal
    "#F0E442",  # 15 Lemon Yellow
    "#0072B2",  # 16 Blue
    "#D55E00",  # 17 Rust
    "#CC79A7",  # 18 Pink
    "#999999",  # 19 Light Grey
]

# Fixed color index per category value for reproducible cross-plot consistency
DICT_COLORS_SEASON = {"spring": 1, "summer": 2, "autumn": 0, "winter": 3}
DICT_COLORS_TEMP = {
    " < 10°C": 0,
    " [17.9°C ;\n 22.2°C]": 1,
    " [22.5°C ;\n 24.8°C]": 2,
    ">28°C": 4,
}

# Mapping from raw temperature values (°C) to display ranges
COLUMN_GROUP_MAP = {
    3.7:  " < 10°C",
    4.9:  " < 10°C",
    5.4:  " < 10°C",
    5.7:  " < 10°C",
    17.9: " [17.9°C ;\n 22.2°C]",
    20.6: " [17.9°C ;\n 22.2°C]",
    22.2: " [17.9°C ;\n 22.2°C]",
    22.5: " [22.5°C ;\n 24.8°C]",
    23.3: " [22.5°C ;\n 24.8°C]",
    24.8: " [22.5°C ;\n 24.8°C]",
    28.0: ">28°C",
    30.0: ">28°C",
}


def import_dataframe(path: str, type_data: str) -> pd.DataFrame:
    """
    Load a JSON annotation file and return a specific sub-table as a DataFrame.

    Parameters
    ----------
    path : str
        Path to the JSON file.
    type_data : str
        Top-level key to extract from the JSON (e.g. 'images', 'annotations').
    """
    file = open(path, "r")
    jason = json.load(file)
    return pd.DataFrame(jason[type_data])


def plot_pie_chart(df: pd.DataFrame, metadata: str) -> None:
    """
    Draw a pie chart showing the distribution of a metadata column.

    Colors are assigned deterministically via per-metadata color dictionaries
    so that the same category always maps to the same color across subplots.
    For unknown metadata columns a sequential palette is used as fallback.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the `metadata` column and a 'tamb' column
        (used here only for counting rows per group via groupby).
    metadata : str
        Column name to group by and visualise.
    """
    # Aggregate: count rows per category
    data = df.groupby(metadata)["tamb"].count()
    labels = data.index
    values = data.values

    # Build legend labels that include percentage per slice
    total = values.sum()
    legend_labels = [
        f"{label} - {value / total * 100:.0f}%"
        for label, value in zip(labels, values)
    ]

    # Select the appropriate color mapping for the metadata column
    if metadata == "Season":
        color_dict = DICT_COLORS_SEASON
    elif metadata == "Temperature range":
        color_dict = DICT_COLORS_TEMP
    elif metadata == "Sequence id":
        # Sequential indices for sequence IDs (no fixed semantic color needed)
        color_dict = {label: i for i, label in enumerate(labels)}
    else:
        # Fallback: assign colors in palette order, no semantic mapping
        wedges, _ = plt.pie(values, colors=FULL_PALETTE)
        plt.legend(
            wedges, legend_labels,
            title=metadata, loc="center left", bbox_to_anchor=(1, 0.5),
            title_fontsize="large", fontsize="large",
        )
        plt.tight_layout()
        return

    # Map each label to its designated palette color
    colors = [FULL_PALETTE[color_dict[label]] for label in labels]

    wedges, _ = plt.pie(values, colors=colors)
    plt.legend(
        wedges, legend_labels,
        title=metadata, loc="center left", bbox_to_anchor=(1, 0.5),
        title_fontsize="large", fontsize="large",
    )


def parse():
    """Parse command-line arguments for input annotation paths and output figure paths."""
    parser = argparse.ArgumentParser(
        description="Plot metadata distribution pie charts for the Lynred dataset splits."
    )

    parser.add_argument(
        "--ALL_data_path",
        type=str,
        default="/scratch2/clear/larbez/Backup_datasets/dataset_lynred_final/metadata/instances_default_vis.json",
        help="Path to the full dataset annotations JSON file.",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="/scratch2/clear/larbez/Backup_datasets/dataset_lynred_final/metadata/IR_train.json",
        help="Path to the training split annotations JSON file.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="/scratch2/clear/larbez/Backup_datasets/dataset_lynred_final/metadata/IR_test.json",
        help="Path to the test split annotations JSON file.",
    )
    parser.add_argument(
        "--train_output_path",
        type=str,
        default="/home/larbez/Documents/random_visu/alltrain.png",
        help="Output path for the training split pie chart figure.",
    )
    parser.add_argument(
        "--test_output_path",
        type=str,
        default="/home/larbez/Documents/random_visu/alltest.png",
        help="Output path for the test split pie chart figure.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    # --- Load image metadata from each split ---
    df_train = import_dataframe(args.train_data_path, "images")
    df_test = import_dataframe(args.test_data_path, "images")

    # Concatenate splits to build a combined view, then load the full reference set
    df_All = pd.concat((df_train, df_test))
    df_ALL = import_dataframe(args.ALL_data_path, "images")

    # --- Reduce to the columns needed for visualisation ---
    essential = ["season", "time_of_day", "tamb", "sequence_id"]
    df_All_reduced = df_All[essential + ["split"]]
    df_train_reduced = df_train[essential]
    df_test_reduced = df_test[essential]

    # Separate the combined DataFrame back into splits using the 'split' flag
    train = df_All_reduced.loc[df_All_reduced["split"] == "train"]
    test = df_All_reduced.loc[df_All_reduced["split"] == "test"]

    # Map raw temperature values to human-readable range labels
    train["Temperature range"] = train["tamb"].map(COLUMN_GROUP_MAP)
    test["Temperature range"] = test["tamb"].map(COLUMN_GROUP_MAP)

    # Rename columns for display-friendly axis and legend titles
    new_names = {
        "season": "Season",
        "time_of_day": "Time of day",
        "sequence_id": "Sequence id",
    }
    train.rename(columns=new_names, inplace=True)
    test.rename(columns=new_names, inplace=True)

    # --- Training split: four metadata pie charts ---
    plt.figure(figsize=[20, 4])
    plt.subplot(141); plot_pie_chart(train, "Sequence id");      plt.tight_layout()
    plt.subplot(142); plot_pie_chart(train, "Time of day");      plt.tight_layout()
    plt.subplot(143); plot_pie_chart(train, "Temperature range"); plt.tight_layout()
    plt.subplot(144); plot_pie_chart(train, "Season");           plt.tight_layout()
    plt.savefig(args.train_output_path)

    # --- Test split: four metadata pie charts ---
    plt.figure(figsize=[20, 4])
    plt.subplot(141); plot_pie_chart(test, "Sequence id")
    plt.subplot(142); plot_pie_chart(test, "Time of day")
    plt.subplot(143); plot_pie_chart(test, "Temperature range")
    plt.subplot(144); plot_pie_chart(test, "Season")
    plt.tight_layout()
    plt.savefig(args.test_output_path)
import shutil
from pathlib import Path
 
# Destination folder (hidden directory, created if it does not exist)
DEST = Path(".datasets")
DEST.mkdir(exist_ok=True)
 
# All JSON annotation files referenced across the three analysis scripts
JSON_FILES = [
    # --- IR datasets ---
    "/srv/storage/thoth1@storage4.grenoble.grid5000.fr/larbez/RGBXFusion/dataset/M3FD/meta/m3fd-train.json",
    "/srv/storage/thoth1@storage4.grenoble.grid5000.fr/larbez/Backup_datasets/FLIR_ADAS_v2/images_thermal_val/coco.json",
    "/srv/storage/thoth1@storage4.grenoble.grid5000.fr/larbez/RGBXFusion/dataset/FLIR_Aligned/meta/thermal/NEW_flir_train.json",
    "/srv/storage/thoth1@storage4.grenoble.grid5000.fr/larbez/Backup_datasets/LLVIP/visible_train_.json",
    "/srv/storage/thoth1@storage4.grenoble.grid5000.fr/larbez/datasets/KAIST_AMFD_COCO/anno/sanitized/lwir_train.json",

    "/srv/storage/thoth1@storage4.grenoble.grid5000.fr/larbez/Backup_datasets/dataset_lynred_final/metadata/IR_test.json",
    "/srv/storage/thoth1@storage4.grenoble.grid5000.fr/larbez/Backup_datasets/FLIR_ADAS_v2/images_thermal_val/coco.json",
    "/srv/storage/thoth1@storage4.grenoble.grid5000.fr/larbez/RGBXFusion/dataset/FLIR_Aligned/meta/thermal/NEW_flir_test.json",
 
    # --- Lynred metadata (dataset_metadata_pies.py) ---
    "/srv/storage/thoth1@storage4.grenoble.grid5000.fr/larbez/Backup_datasets/dataset_lynred_final/metadata/instances_default_vis.json",
    "/srv/storage/thoth1@storage4.grenoble.grid5000.fr/larbez/Backup_datasets/dataset_lynred_final/metadata/IR_train.json",
]
 
for src_str in JSON_FILES:
    src = Path(src_str)
    dest = DEST / src.name
 
    if not src.exists():
        print(f"[SKIP]  {src}  (not found)")
        continue
 
    if dest.exists():
        print(f"[SKIP]  {src.name}  (already in {DEST})")
        continue
 
    shutil.move(str(src), str(dest))
    print(f"[MOVED] {src}  →  {dest}")
 
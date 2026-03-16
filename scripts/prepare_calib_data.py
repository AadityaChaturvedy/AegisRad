import os
import shutil
import pandas as pd

MIMIC_ROOT = "/path/to/mimic-cxr-jpg"
OUTPUT_DIR = "data/calib_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

chexpert = pd.read_csv(f"{MIMIC_ROOT}/mimic-cxr-2.0.0-chexpert.csv")
split    = pd.read_csv(f"{MIMIC_ROOT}/mimic-cxr-2.0.0-split.csv")
meta     = pd.read_csv(f"{MIMIC_ROOT}/mimic-cxr-2.0.0-metadata.csv")

val      = split[split["split"] == "validate"]
frontal  = meta[meta["ViewPosition"].isin(["PA", "AP"])]
val_meta = val.merge(frontal, on="study_id")

conditions = [
    "No Finding",
    "Pneumonia",
    "Pleural Effusion",
    "Cardiomegaly",
    "Consolidation"
]

copied = 0
for condition in conditions:
    subset = chexpert[chexpert[condition] == 1.0]
    subset = subset.merge(val_meta, on="study_id").head(40)

    for _, row in subset.iterrows():
        pid = str(row.subject_id)
        src = (
            f"{MIMIC_ROOT}/files/"
            f"p{pid[:2]}/p{pid}/"
            f"s{row.study_id}/{row.dicom_id}.jpg"
        )
        dst = (
            f"{OUTPUT_DIR}/"
            f"{condition.replace(' ', '_')}_{row.dicom_id}.jpg"
        )
        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1
            print(f"[{copied}] Copied: {condition}")

        if copied >= 200:
            break

print(f"\n[AegisRad] Done. {copied} calibration images in {OUTPUT_DIR}/")
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"

metadata = pd.read_csv(DATA / "HAM10000_metadata.csv")

splits = {
    name: pd.read_csv(DATA / f"{name}.csv")
    for name in ["train", "validation", "test"]
}

for name, df in splits.items():
    print(f"{name}: {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")

# Check image overlap
for a, b in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]:
    overlap = set(splits[a]["image_id"]) & set(splits[b]["image_id"])
    print(f"Image overlap ({a}, {b}): {len(overlap)}")

# Map each image_id to its lesion_id using the metadata
image_to_lesion = metadata.set_index("image_id")["lesion_id"]

for name, df in splits.items():
    if "lesion_id" not in df.columns:
        splits[name]["lesion_id"] = splits[name]["image_id"].map(
            image_to_lesion
        )

    missing = splits[name]["lesion_id"].isna().sum()
    print(f"Missing lesion IDs in {name}: {missing}")

# Check lesion overlap
for a, b in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]:
    lesions_a = set(splits[a]["lesion_id"].dropna())
    lesions_b = set(splits[b]["lesion_id"].dropna())
    overlap = lesions_a & lesions_b
    print(f"Lesion overlap ({a}, {b}): {len(overlap)}")
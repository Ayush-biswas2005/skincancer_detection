from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"

df = pd.read_csv(DATA / "HAM10000_metadata.csv")

# Keep only rows whose image files exist.
image_dir = DATA / "images"
df = df[
    df["image_id"].map(
        lambda x: (image_dir / f"{x}.jpg").is_file()
    )
].dropna(subset=["lesion_id", "dx"]).copy()

# One lesion must belong to only one split.
first_split = GroupShuffleSplit(
    n_splits=1, test_size=0.15, random_state=42
)
trainval_idx, test_idx = next(
    first_split.split(df, groups=df["lesion_id"])
)

trainval = df.iloc[trainval_idx].copy()
test = df.iloc[test_idx].copy()

# Reserve approximately 15% of the full dataset for validation.
val_fraction = 0.15 / 0.85

second_split = GroupShuffleSplit(
    n_splits=1, test_size=val_fraction, random_state=43
)
train_idx, val_idx = next(
    second_split.split(trainval, groups=trainval["lesion_id"])
)

train = trainval.iloc[train_idx].copy()
val = trainval.iloc[val_idx].copy()

# Verify that lesion IDs do not cross splits.
for name_a, a, name_b, b in [
    ("train", train, "validation", val),
    ("train", train, "test", test),
    ("validation", val, "test", test),
]:
    overlap = set(a["lesion_id"]) & set(b["lesion_id"])
    print(f"Lesion overlap ({name_a}, {name_b}): {len(overlap)}")
    assert not overlap, "Lesion leakage detected!"

# Back up old CSVs before replacing them.
backup = DATA / "old_splits_backup"
backup.mkdir(exist_ok=True)

for name in ["train", "validation", "test"]:
    path = DATA / f"{name}.csv"
    if path.exists():
        path.replace(backup / f"{name}.csv")

# Save the new splits in the format train.py already expects.
for name, part in [
    ("train", train),
    ("validation", val),
    ("test", test),
]:
    part.to_csv(DATA / f"{name}.csv", index=False)
    print(f"{name}: {len(part)} images")
    print(part["dx"].value_counts().sort_index())

print("\nGroup-safe splits created successfully.")
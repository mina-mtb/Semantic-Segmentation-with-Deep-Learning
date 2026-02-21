import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import numpy as np
from sklearn.model_selection import KFold

SEED = 42
N_FOLDS = 5


# =====================================================================
# CONFIGURATION - Portable path discovery
import os
POSSIBLE_DATA_PATHS = [
    os.path.join(os.getcwd(), 'Potsdam-GeoTif'),
    os.path.join(os.getcwd(), 'data'),
    os.path.join(os.path.dirname(os.getcwd()), 'Potsdam-GeoTif'),
    os.path.join(os.path.dirname(os.getcwd()), 'data'),
    os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'Potsdam-GeoTif'),
    os.getcwd()
]
DATA_DIR = next((p for p in POSSIBLE_DATA_PATHS if os.path.exists(p)), 'data')
# =====================================================================

# Original local path:
# DATA_DIR = r'C:\Users\mina_\OneDrive\Documents\DESING_OF_AI_SYSTEMS\Semantic Segmentation with Deep Learning\PROJECT'
DATA_DIR = DATA_DIR # already discovered above

# ── Collect all .tif files ──────────────────────────────────────────
all_tifs = []
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith('.tif'):
            all_tifs.append(os.path.join(root, f))

print(f"Found {len(all_tifs)} .tif file(s): {[os.path.basename(f) for f in all_tifs]}")

# ── Simulate 5-fold split ───────────────────────────────────────────
# NOTE: With only 1 file, we duplicate it to demonstrate the split logic.
# When the full dataset arrives, just replace all_tifs with the actual list.

# For demonstration with 1 file, we create a "virtual" list by repeating
# to show the k-fold mechanics clearly.

if len(all_tifs) < N_FOLDS:
    # Duplicate the file references for demonstration
    demo_files = all_tifs * 20   # 20 copies = 20 "virtual" samples
    print(f"\nNote: Only {len(all_tifs)} file(s) found. Duplicating to {len(demo_files)} for demonstration.")
else:
    demo_files = all_tifs

np.random.seed(SEED)
np.random.shuffle(demo_files)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
demo_array = np.array(demo_files)

folds = []
for fold_idx, (_, fold_indices) in enumerate(kf.split(demo_array)):
    fold_files = demo_array[fold_indices].tolist()
    folds.append(fold_files)
    print(f"  Fold {fold_idx+1}: {len(fold_files)} samples")

# Fold assignment
train_files = folds[0] + folds[1] + folds[2]
val_files   = folds[3]
test_files  = folds[4]

print(f"\nSplit summary:")
print(f"  Training   (Folds 1+2+3): {len(train_files)} samples")
print(f"  Validation (Fold 4)      : {len(val_files)} samples")
print(f"  Test       (Fold 5)      : {len(test_files)} samples")

# Save fold_splits.json
splits = {
    "train": train_files,
    "val":   val_files,
    "test":  test_files,
    "all_folds": folds,
    "note": "Paths point to real files. Duplicated for demo with 1 sample file."
}

out_json = os.path.join(DATA_DIR, 'fold_splits.json')
with open(out_json, 'w') as fp:
    json.dump(splits, fp, indent=2)

print(f"\nfold_splits.json saved to: {out_json}")
print("=== Step 1 Complete ===")

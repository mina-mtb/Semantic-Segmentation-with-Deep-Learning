import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio

# ── CONFIG ─────────────────────────────────────────────────────────────

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
# SAMPLE_FILE = r'C:\Users\mina_\OneDrive\Documents\DESING_OF_AI_SYSTEMS\Semantic Segmentation with Deep Learning\PROJECT\0000000224-0000042784.tif'
SAMPLE_FILE = os.path.join(DATA_DIR, '0000000224-0000042784.tif')

CLASS_NAMES = [
    'Impervious surface', 'Building', 'Tree',
    'Low vegetation', 'Car', 'Clutter/Background'
]
CLASS_COLORS = [
    [255, 255, 255],  # white
    [0,   0,   255],  # blue
    [0,   255,   0],  # green
    [0,   255, 255],  # cyan
    [255, 255,   0],  # yellow
    [255,   0,   0],  # red
]

# ── READ FILE ──────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 1 - Reading Sample GeoTIFF File")
print("=" * 60)

with rasterio.open(SAMPLE_FILE) as src:
    data   = src.read()          # shape: (bands, H, W)
    meta   = src.meta
    bounds = src.bounds
    crs    = src.crs
    transform = src.transform

print(f"\nFile       : {os.path.basename(SAMPLE_FILE)}")
print(f"Data shape : {data.shape}  (bands, height, width)")
print(f"Data type  : {data.dtype}")
print(f"CRS        : {crs}")
print(f"Bounds     : {bounds}")
print(f"\nBand summary:")
for i, name in enumerate(['Red', 'Green', 'Blue', 'IR', 'Elevation', 'Labels']):
    band = data[i]
    print(f"  Band {i} ({name:12s}): min={band.min():8.2f}  max={band.max():8.2f}  mean={band.mean():8.2f}")

# ── EXTRACT BANDS ──────────────────────────────────────────────────────
def normalize(band):
    b_min, b_max = band.min(), band.max()
    if b_max == b_min:
        return np.zeros_like(band, dtype=np.float32)
    return (band - b_min).astype(np.float32) / (b_max - b_min)

red   = data[0].astype(np.float32)
green = data[1].astype(np.float32)
blue  = data[2].astype(np.float32)
elev  = data[4].astype(np.float32)
label = data[5].astype(np.int32)

rgb_image = np.stack([normalize(red), normalize(green), normalize(blue)], axis=-1)

# ── LABEL -> RGB ───────────────────────────────────────────────────────
def label_to_rgb(label_band, colors):
    h, w = label_band.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(colors):
        rgb[label_band == idx] = color
    return rgb

label_rgb = label_to_rgb(label, CLASS_COLORS)

unique_classes = np.unique(label)
print(f"\nClasses present in this tile: {unique_classes.tolist()}")
for c in unique_classes:
    pct = (label == c).sum() / label.size * 100
    print(f"  Class {c} ({CLASS_NAMES[c]:22s}): {pct:.2f}% of pixels")

# ── PLOT ──────────────────────────────────────────────────────────────
patches = [mpatches.Patch(color=[c/255 for c in CLASS_COLORS[i]], label=CLASS_NAMES[i])
           for i in range(len(CLASS_NAMES))]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Sample GeoTIFF: {os.path.basename(SAMPLE_FILE)}', fontsize=13, fontweight='bold')

# 1. RGB
axes[0].imshow(rgb_image)
axes[0].set_title('RGB Image\n(Bands: Red, Green, Blue)', fontsize=11)
axes[0].axis('off')

# 2. Elevation
im_e = axes[1].imshow(normalize(elev), cmap='terrain')
axes[1].set_title('Elevation Band\n(Band index 4)', fontsize=11)
axes[1].axis('off')
plt.colorbar(im_e, ax=axes[1], fraction=0.046, pad=0.04, label='Normalized Elevation')

# 3. Label
axes[2].imshow(label_rgb)
axes[2].set_title('Target Label Band\n(Band index 5 - Semantic Classes)', fontsize=11)
axes[2].axis('off')
axes[2].legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.9)

plt.tight_layout()
# Original local path:
# out_path = r'C:\Users\mina_\OneDrive\Documents\DESING_OF_AI_SYSTEMS\Semantic Segmentation with Deep Learning\PROJECT\step1_visualization.png'
out_path = os.path.join('outputs', 'step1_visualization.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\nVisualization saved to: step1_visualization.png")
print("\n=== DONE ===")

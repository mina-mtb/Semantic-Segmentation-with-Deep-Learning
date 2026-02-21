import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

# ─── Hyperparameters ─────────────────────────────────────────────────
SEED            = 42
BATCH_SIZE      = 2      # small for demo
EPOCHS          = 20
LEARNING_RATE   = 1e-4
NUM_CLASSES     = 6
INPUT_CHANNELS  = 5      # RGB + IR + Elevation
MODEL_SAVE_PATH = 'best_unet_model.keras'


# =====================================================================
# CONFIGURATION - Portable path discovery
import os
def discover_data_dir():
    possible = [
        os.path.join(os.getcwd(), 'Potsdam-GeoTif'),
        os.path.join(os.getcwd(), 'data'),
        os.path.join(os.getcwd(), 'PROJECT', 'Potsdam-GeoTif'),
        os.path.join(os.getcwd(), 'PROJECT', 'data'),
        os.getcwd()
    ]
    existing = [p for p in possible if os.path.exists(p)]
    for p in existing:
        try:
            if any(f.endswith('.tif') for f in os.listdir(p)): return p
        except: continue
    return existing[0] if existing else 'data'

DATA_DIR = discover_data_dir()
# =====================================================================

# Original local path:
# DATA_DIR    = r'C:\Users\mina_\OneDrive\Documents\DESING_OF_AI_SYSTEMS\Semantic Segmentation with Deep Learning\PROJECT'
SPLITS_JSON = os.path.join(DATA_DIR, 'fold_splits.json')

CLASS_NAMES  = ['Impervious surface','Building','Tree','Low vegetation','Car','Clutter/Background']
CLASS_COLORS = [[255,255,255],[0,0,255],[0,255,0],[0,255,255],[255,255,0],[255,0,0]]

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("  STEP 3 - U-Net Encoder-Decoder Training")
print("=" * 60)
print(f"TensorFlow: {tf.__version__}")

# ─── Data loading ─────────────────────────────────────────────────────
def normalize_band(band):
    b_min, b_max = band.min(), band.max()
    if b_max == b_min:
        return np.zeros_like(band, dtype=np.float32)
    return (band - b_min).astype(np.float32) / (b_max - b_min)

def load_sample(file_path, use_all_bands=True):
    with rasterio.open(file_path) as src:
        data = src.read()
    n = 5 if use_all_bands else 4
    features = data[:n].transpose(1, 2, 0).astype(np.float32)
    for c in range(features.shape[-1]):
        features[..., c] = normalize_band(features[..., c])
    label = data[5].astype(np.int32)
    return features, tf.keras.utils.to_categorical(label, num_classes=NUM_CLASSES), data

def augment_numpy(img, lbl):
    if np.random.rand() > 0.5:
        img = img[:,::-1,:]; lbl = lbl[:,::-1,:]
    if np.random.rand() > 0.5:
        img = img[::-1,:,:]; lbl = lbl[::-1,:,:]
    k = np.random.randint(0, 4)
    img = np.rot90(img, k).copy(); lbl = np.rot90(lbl, k).copy()
    return img, lbl

def make_dataset(file_list, augment=False, batch_size=2):
    all_X, all_y = [], []
    for fp in file_list:
        X, y, _ = load_sample(fp, use_all_bands=True)
        if augment:
            X, y = augment_numpy(X, y)
        all_X.append(X); all_y.append(y)
    X_arr = np.stack(all_X); y_arr = np.stack(all_y)
    ds = tf.data.Dataset.from_tensor_slices((X_arr, y_arr))
    if augment: ds = ds.shuffle(len(file_list), seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

with open(SPLITS_JSON) as fp:
    splits = json.load(fp)

train_files = splits['train']
val_files   = splits['val']
test_files  = splits['test']

print("\nLoading data...")
train_ds = make_dataset(train_files, augment=True,  batch_size=BATCH_SIZE)
val_ds   = make_dataset(val_files,   augment=False, batch_size=BATCH_SIZE)
test_ds  = make_dataset(test_files,  augment=False, batch_size=BATCH_SIZE)
print(f"  Train / Val / Test batches: {len(list(train_ds))} / {len(list(val_ds))} / {len(list(test_ds))}")

# ─── U-Net Model ──────────────────────────────────────────────────────
def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', name=f'{name}_c1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', name=f'{name}_c2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    return x

def build_unet(in_ch=5, nc=6, f=32):
    inputs = keras.Input(shape=(None, None, in_ch), name='input')
    # Encoder
    e1 = conv_block(inputs, f*1,  'enc1'); p1 = layers.MaxPooling2D(2, name='pool1')(e1)
    e2 = conv_block(p1,     f*2,  'enc2'); p2 = layers.MaxPooling2D(2, name='pool2')(e2)
    e3 = conv_block(p2,     f*4,  'enc3'); p3 = layers.MaxPooling2D(2, name='pool3')(e3)
    e4 = conv_block(p3,     f*8,  'enc4'); p4 = layers.MaxPooling2D(2, name='pool4')(e4)
    # Bottleneck
    b  = conv_block(p4,     f*16, 'bottleneck')
    # Decoder
    u4 = layers.UpSampling2D(2, name='up4')(b)
    u4 = layers.Concatenate(name='skip4')([u4, e4]); d4 = conv_block(u4, f*8, 'dec4')
    u3 = layers.UpSampling2D(2, name='up3')(d4)
    u3 = layers.Concatenate(name='skip3')([u3, e3]); d3 = conv_block(u3, f*4, 'dec3')
    u2 = layers.UpSampling2D(2, name='up2')(d3)
    u2 = layers.Concatenate(name='skip2')([u2, e2]); d2 = conv_block(u2, f*2, 'dec2')
    u1 = layers.UpSampling2D(2, name='up1')(d2)
    u1 = layers.Concatenate(name='skip1')([u1, e1]); d1 = conv_block(u1, f*1, 'dec1')
    out = layers.Conv2D(nc, 1, padding='same', activation='softmax', name='output')(d1)
    return keras.Model(inputs, out, name='UNet_SegModel')

unet = build_unet(in_ch=INPUT_CHANNELS, nc=NUM_CLASSES, f=32)
unet.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nModel: {unet.name}  |  Parameters: {unet.count_params():,}")
unet.summary()

# Architecture plot
try:
    keras.utils.plot_model(unet, to_file=os.path.join(DATA_DIR, 'step3_unet_architecture.png'),
                           show_shapes=True, show_layer_names=True, dpi=80)
    print("Architecture saved: step3_unet_architecture.png")
except Exception as e:
    print(f"Could not save architecture plot: {e}")

# ─── Training ─────────────────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...")
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(DATA_DIR, MODEL_SAVE_PATH),
        monitor='val_accuracy', save_best_only=True, mode='max', verbose=0
    )
]

history = unet.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                   callbacks=callbacks, verbose=1)

# ─── Training curves ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ep = range(1, len(history.history['loss'])+1)

ax1.plot(ep, history.history['loss'],     label='Train Loss', color='royalblue', lw=2)
ax1.plot(ep, history.history['val_loss'], label='Val Loss',   color='darkorange', lw=2, linestyle='--')
ax1.set_title('U-Net — Loss', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(ep, history.history['accuracy'],     label='Train Acc', color='forestgreen', lw=2)
ax2.plot(ep, history.history['val_accuracy'], label='Val Acc',   color='crimson',     lw=2, linestyle='--')
ax2.set_title('U-Net — Accuracy', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(DATA_DIR, 'step3_training_curves.png')
plt.savefig(curves_path, dpi=150); plt.close()
print("Training curves saved: step3_training_curves.png")

# ─── Evaluate on test set ─────────────────────────────────────────────
print("\nEvaluating on test set...")
best_model = keras.models.load_model(os.path.join(DATA_DIR, MODEL_SAVE_PATH))
test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
print(f"\n{'='*40}")
print(f"  TEST SET RESULTS (U-Net)")
print(f"  Loss    : {test_loss:.4f}")
print(f"  Accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"{'='*40}")

# ─── Prediction visualization ─────────────────────────────────────────
def label_to_rgb(label_band, colors):
    h, w = label_band.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, c in enumerate(colors):
        rgb[label_band == idx] = c
    return rgb

# Pick one test sample
test_fp = test_files[0]
X, y_oh, raw_data = load_sample(test_fp, use_all_bands=True)
gt_label  = raw_data[5].astype(np.int32)

pred_logits = best_model.predict(np.expand_dims(X, 0), verbose=0)[0]  # (H, W, 6)
pred_label  = np.argmax(pred_logits, axis=-1)

rgb_vis  = np.stack([normalize_band(raw_data[0].astype(np.float32)),
                     normalize_band(raw_data[1].astype(np.float32)),
                     normalize_band(raw_data[2].astype(np.float32))], axis=-1)
elev_vis = normalize_band(raw_data[4].astype(np.float32))
gt_rgb   = label_to_rgb(gt_label,   CLASS_COLORS)
pred_rgb = label_to_rgb(pred_label, CLASS_COLORS)

patches = [mpatches.Patch(color=[c/255 for c in CLASS_COLORS[i]], label=CLASS_NAMES[i])
           for i in range(NUM_CLASSES)]

fig, axes = plt.subplots(1, 4, figsize=(22, 6))
fig.suptitle('U-Net Prediction vs Ground Truth', fontsize=13, fontweight='bold')

axes[0].imshow(rgb_vis);  axes[0].set_title('RGB Image');         axes[0].axis('off')
im = axes[1].imshow(elev_vis, cmap='terrain');                     axes[1].set_title('Elevation Band'); axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
axes[2].imshow(gt_rgb);   axes[2].set_title('Ground Truth Label'); axes[2].axis('off')
axes[2].legend(handles=patches, loc='lower right', fontsize=6.5, framealpha=0.9)
axes[3].imshow(pred_rgb); axes[3].set_title('U-Net Prediction');  axes[3].axis('off')
axes[3].legend(handles=patches, loc='lower right', fontsize=6.5, framealpha=0.9)

plt.tight_layout()
pred_path = os.path.join(DATA_DIR, 'step3_prediction_visualization.png')
plt.savefig(pred_path, dpi=150, bbox_inches='tight'); plt.close()
print("Prediction visualization saved: step3_prediction_visualization.png")

# Save results
results = {
    "model": "UNet",
    "test_loss": round(float(test_loss), 4),
    "test_accuracy": round(float(test_acc), 4),
    "epochs_trained": len(history.history['loss']),
    "best_val_acc": round(float(max(history.history['val_accuracy'])), 4),
    "train_acc_final": round(float(history.history['accuracy'][-1]), 4),
    "train_loss_final": round(float(history.history['loss'][-1]), 4),
}
with open(os.path.join(DATA_DIR, 'step3_results.json'), 'w') as fp:
    json.dump(results, fp, indent=2)

print("\nResults saved to step3_results.json")
print("=== Step 3 Complete ===")

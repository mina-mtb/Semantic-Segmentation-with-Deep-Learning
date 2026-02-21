import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

# ─── Hyperparameters (document all assumptions) ───────────────────────
SEED            = 42
BATCH_SIZE      = 2      # small for demo with 1 file
EPOCHS          = 20
LEARNING_RATE   = 1e-3
NUM_CLASSES     = 6
INPUT_CHANNELS  = 4      # RGB + IR
MODEL_SAVE_PATH = 'best_simple_model.keras'


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
# DATA_DIR   = r'C:\Users\mina_\OneDrive\Documents\DESING_OF_AI_SYSTEMS\Semantic Segmentation with Deep Learning\PROJECT'
SPLITS_JSON = os.path.join(DATA_DIR, 'fold_splits.json')

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("  STEP 2 - Simple CNN Model Training")
print("=" * 60)
print(f"TensorFlow : {tf.__version__}")
print(f"GPUs found : {tf.config.list_physical_devices('GPU')}")

# ─── Data loading ──────────────────────────────────────────────────────
CLASS_NAMES = [
    'Impervious surface','Building','Tree',
    'Low vegetation','Car','Clutter/Background'
]

def normalize_band(band):
    b_min, b_max = band.min(), band.max()
    if b_max == b_min:
        return np.zeros_like(band, dtype=np.float32)
    return (band - b_min).astype(np.float32) / (b_max - b_min)

def load_sample(file_path, use_all_bands=False):
    with rasterio.open(file_path) as src:
        data = src.read()                          # (6, H, W)
    n = 5 if use_all_bands else 4
    features = data[:n].transpose(1, 2, 0).astype(np.float32)
    for c in range(features.shape[-1]):
        features[..., c] = normalize_band(features[..., c])
    label = data[5].astype(np.int32)
    label_oh = tf.keras.utils.to_categorical(label, num_classes=NUM_CLASSES)
    return features, label_oh

def augment_numpy(img, lbl):
    """Apply random horizontal/vertical flip and 90° rotation."""
    if np.random.rand() > 0.5:
        img  = img[:, ::-1, :]
        lbl  = lbl[:, ::-1, :]
    if np.random.rand() > 0.5:
        img  = img[::-1, :, :]
        lbl  = lbl[::-1, :, :]
    k = np.random.randint(0, 4)
    img = np.rot90(img, k)
    lbl = np.rot90(lbl, k)
    return img.copy(), lbl.copy()

def make_tf_dataset(file_list, augment=False, batch_size=2):
    all_X, all_y = [], []
    for fp in file_list:
        X, y = load_sample(fp, use_all_bands=False)
        if augment:
            X, y = augment_numpy(X, y)
        all_X.append(X)
        all_y.append(y)
    X_arr = np.stack(all_X, axis=0)
    y_arr = np.stack(all_y, axis=0)
    ds = tf.data.Dataset.from_tensor_slices((X_arr, y_arr))
    if augment:
        ds = ds.shuffle(len(file_list), seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Load splits
with open(SPLITS_JSON) as fp:
    splits = json.load(fp)

train_files = splits['train']
val_files   = splits['val']
test_files  = splits['test']

print(f"\nLoading data...")
train_ds = make_tf_dataset(train_files, augment=True,  batch_size=BATCH_SIZE)
val_ds   = make_tf_dataset(val_files,   augment=False, batch_size=BATCH_SIZE)
test_ds  = make_tf_dataset(test_files,  augment=False, batch_size=BATCH_SIZE)
print(f"  Train batches: {len(list(train_ds))}")
print(f"  Val   batches: {len(list(val_ds))}")
print(f"  Test  batches: {len(list(test_ds))}")

# ─── Model ────────────────────────────────────────────────────────────
def build_simple_model(in_ch=4, nc=6):
    inputs = keras.Input(shape=(None, None, in_ch), name='input')
    x = layers.Conv2D(32,  3, padding='same', activation='relu', name='conv1_1')(inputs)
    x = layers.BatchNormalization(name='bn1_1')(x)
    x = layers.Conv2D(32,  3, padding='same', activation='relu', name='conv1_2')(x)
    x = layers.BatchNormalization(name='bn1_2')(x)
    x = layers.Conv2D(64,  3, padding='same', activation='relu', name='conv2_1')(x)
    x = layers.BatchNormalization(name='bn2_1')(x)
    x = layers.Conv2D(64,  3, padding='same', activation='relu', name='conv2_2')(x)
    x = layers.BatchNormalization(name='bn2_2')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv3_1')(x)
    x = layers.BatchNormalization(name='bn3_1')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv3_2')(x)
    x = layers.BatchNormalization(name='bn3_2')(x)
    out = layers.Conv2D(nc, 1, padding='same', activation='softmax', name='output')(x)
    return keras.Model(inputs, out, name='SimpleCNN_SegModel')

model = build_simple_model(in_ch=INPUT_CHANNELS, nc=NUM_CLASSES)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nModel: {model.name}")
print(f"Parameters: {model.count_params():,}")
model.summary()

# Save architecture plot
try:
    keras.utils.plot_model(model, to_file=os.path.join(DATA_DIR, 'step2_model_architecture.png'),
                           show_shapes=True, show_layer_names=True, dpi=100)
    print("Architecture saved: step2_model_architecture.png")
except Exception as e:
    print(f"Could not plot model: {e}")

# ─── Training ─────────────────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...")
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(DATA_DIR, MODEL_SAVE_PATH),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ─── Training curves ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ep = range(1, len(history.history['loss']) + 1)

ax1.plot(ep, history.history['loss'],     label='Train Loss', color='royalblue', lw=2)
ax1.plot(ep, history.history['val_loss'], label='Val Loss',   color='darkorange', lw=2, linestyle='--')
ax1.set_title('Simple CNN — Loss', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(ep, history.history['accuracy'],     label='Train Acc', color='forestgreen', lw=2)
ax2.plot(ep, history.history['val_accuracy'], label='Val Acc',   color='crimson',     lw=2, linestyle='--')
ax2.set_title('Simple CNN — Accuracy', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(DATA_DIR, 'step2_training_curves.png')
plt.savefig(curves_path, dpi=150)
plt.close()
print(f"Training curves saved: step2_training_curves.png")

# ─── Evaluate on test set ─────────────────────────────────────────────
print("\nEvaluating on test set...")
best_model = keras.models.load_model(os.path.join(DATA_DIR, MODEL_SAVE_PATH))
test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)

print("\n" + "=" * 40)
print(f"  TEST SET RESULTS")
print(f"  Loss    : {test_loss:.4f}")
print(f"  Accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)")
print("=" * 40)

# Save results to JSON for LaTeX update
results = {
    "model": "SimpleCNN",
    "test_loss": round(float(test_loss), 4),
    "test_accuracy": round(float(test_acc), 4),
    "epochs_trained": len(history.history['loss']),
    "best_val_acc": round(float(max(history.history['val_accuracy'])), 4),
    "best_val_loss": round(float(min(history.history['val_loss'])), 4),
    "train_loss_final": round(float(history.history['loss'][-1]), 4),
    "train_acc_final":  round(float(history.history['accuracy'][-1]), 4),
}
with open(os.path.join(DATA_DIR, 'step2_results.json'), 'w') as fp:
    json.dump(results, fp, indent=2)
print("\nResults saved to step2_results.json")
print("=== Step 2 Complete ===")

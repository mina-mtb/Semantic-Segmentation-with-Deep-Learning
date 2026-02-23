import json
import os

def standardize_notebook(filepath, samples=100):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Simplified standardization logic
    # Cell 0: Header (keep)
    # Cell 1: Libraries (replace)
    # Cell 2: Description (keep)
    # Cell 3: Imports (remove redundant)
    # ... and so on.
    
    # Actually, a safer approach is to find the specific cells by their content or type
    # and inject the standard configuration.

    new_cells = []
    
    # Standard Import Block
    lib_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, random, json\n",
            "import numpy as np\n",
            "import matplotlib\n",
            "import matplotlib.pyplot as plt\n",
            "import matplotlib.patches as mpatches\n",
            "import rasterio\n",
            "import tensorflow as tf\n",
            "from tensorflow import keras\n",
            "from tensorflow.keras import layers\n",
            "from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping\n",
            "from sklearn.model_selection import KFold\n",
            "from IPython.display import Image, display\n",
            "\n",
            "matplotlib.use('Agg')  # headless mode\n",
            "print('All libraries loaded \\u2713')\n",
            "print('TensorFlow version:', tf.__version__)\n",
            "print('Rasterio version:  ', rasterio.__version__)"
        ]
    }

    # Standard Config Block
    config_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# \\u2500\\u2500 Reproducibility seed \\u2500\\u2500\n",
            "SEED = 42\n",
            "random.seed(SEED)\n",
            "np.random.seed(SEED)\n",
            "tf.random.set_seed(SEED)\n",
            "\n",
            "# \\u2500\\u2500 Global settings \\u2500\\u2500\n",
            "N_FOLDS      = 5\n",
            "NUM_CLASSES  = 6\n",
            f"NUM_SAMPLES  = {samples}   # For full training on Kaggle, use 5000\n",
            "BATCH_SIZE   = 8\n",
            "EPOCHS       = 20\n",
            "LEARNING_RATE = 1e-4\n",
            "OUTPUT_DIR   = './outputs'\n",
            "MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'best_model.h5')\n",
            "\n",
            "if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)\n",
            "\n",
            "CLASS_NAMES = [\n",
            "    'Impervious surface', 'Building', 'Tree',\n",
            "    'Low vegetation', 'Car', 'Clutter/Background'\n",
            "]\n",
            "CLASS_COLORS = [\n",
            "    [255, 255, 255],  # white\n",
            "    [0,   0,   255],  # blue\n",
            "    [0,   255,   0],  # green\n",
            "    [0,   255, 255],  # cyan\n",
            "    [255, 255,   0],  # yellow\n",
            "    [255,   0,   0],  # red\n",
            "]\n",
            "\n",
            "def discover_data_dir():\n",
            "    \"\"\"Auto-locate the folder containing .tif files.\"\"\"\n",
            "    candidates = [\n",
            "        '/kaggle/input/potsdam-geotif/Potsdam-GeoTif',  # Kaggle path\n",
            "        './PROJECT/Potsdam-GeoTif/Potsdam-GeoTif',\n",
            "        '../PROJECT/Potsdam-GeoTif/Potsdam-GeoTif',\n",
            "        './Potsdam-GeoTif/Potsdam-GeoTif',\n",
            "        './data'\n",
            "    ]\n",
            "    for p in candidates:\n",
            "        if os.path.isdir(p):\n",
            "            if any(f.endswith('.tif') for f in os.listdir(p)): return p\n",
            "    \n",
            "    # Fallback search\n",
            "    for root, _, files in os.walk('.'):\n",
            "        if any(f.endswith('.tif') for f in files): return root\n",
            "    return 'data'\n",
            "\n",
            "DATA_DIR = discover_data_dir()\n",
            "print(f\"Data directory: {DATA_DIR}\")\n",
            "print(f\"Seed initialized: {SEED}\")"
        ]
    }

    # Standard Data Discovery Block
    discovery_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def get_all_tif_files(data_dir):\n",
            "    \"\"\"Recursively find all .tif file paths.\"\"\"\n",
            "    tif_files = []\n",
            "    for root, _, files in os.walk(data_dir):\n",
            "        for f in files:\n",
            "            if f.endswith('.tif'):\n",
            "                tif_files.append(os.path.join(root, f))\n",
            "    return sorted(tif_files)\n",
            "\n",
            "all_files_raw = get_all_tif_files(DATA_DIR)\n",
            "all_files = all_files_raw[:NUM_SAMPLES]\n",
            "\n",
            "print(f'Total GeoTIFF files found: {len(all_files_raw)}')\n",
            "print(f'Files selected for this session: {len(all_files)}')\n",
            "print('Sample files:', [os.path.basename(f) for f in all_files[:5]])"
        ]
    }

    processed_lib = False
    processed_config = False
    processed_discovery = False

    for cell in nb['cells']:
        source = "".join(cell['source'])
        
        # Identify library cell (usually first code cell with tensorflow or pip)
        if cell['cell_type'] == 'code' and not processed_lib and ('import tensorflow' in source or 'pip install' in source):
            new_cells.append(lib_cell)
            processed_lib = True
            continue
            
        # Identify config cell (usually mentions SEED or DATA_DIR)
        if cell['cell_type'] == 'code' and not processed_config and ('SEED =' in source or 'DATA_DIR =' in source):
            new_cells.append(config_cell)
            processed_config = True
            continue

        # Identify discovery cell (contains get_all_tif_files)
        if cell['cell_type'] == 'code' and not processed_discovery and 'def get_all_tif_files' in source:
            new_cells.append(discovery_cell)
            processed_discovery = True
            continue

        # Filter out redundant redundant import cells
        if cell['cell_type'] == 'code' and ('import os' in source and 'import random' in source):
            continue
            
        new_cells.append(cell)

    nb['cells'] = new_cells
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Standardized {filepath}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <filepath> [samples]")
    else:
        path = sys.argv[1]
        samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        standardize_notebook(path, samples)

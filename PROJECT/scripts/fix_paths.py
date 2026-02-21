import json
import os
import re

def fix_notebook_paths(nb_path):
    if not os.path.exists(nb_path):
        print(f"Skipping {nb_path}, not found.")
        return
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    # Portable path snippet to insert
    portable_logic = [
        "import os\n",
        "POSSIBLE_DATA_PATHS = [\n",
        "    os.path.join(os.getcwd(), 'Potsdam-GeoTif'),\n",
        "    os.path.join(os.getcwd(), 'data'),\n",
        "    os.path.join(os.path.dirname(os.getcwd()), 'Potsdam-GeoTif'),\n",
        "    os.path.join(os.path.dirname(os.getcwd()), 'data'),\n",
        "    os.getcwd()\n",
        "]\n",
        "DATA_DIR = next((p for p in POSSIBLE_DATA_PATHS if os.path.exists(p)), 'data')\n",
        "print(f\"Data directory: {DATA_DIR}\")\n"
    ]

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            lines = cell['source']
            cell_modified = False
            
            # Find any assignment to DATA_DIR
            for i, line in enumerate(lines):
                if re.match(r'^\s*DATA_DIR\s*=', line):
                    # Found it! Replace with the portable snippet
                    # Check if we already have the logic here
                    if 'POSSIBLE_DATA_PATHS' in "".join(lines):
                        continue 
                    
                    # Replace this line and insert the rest
                    lines[i] = "DATA_DIR = next((p for p in POSSIBLE_DATA_PATHS if os.path.exists(p)), 'data')\n"
                    # Insert the setup before it
                    setup_block = [
                        "import os\n",
                        "POSSIBLE_DATA_PATHS = [\n",
                        "    os.path.join(os.getcwd(), 'Potsdam-GeoTif'),\n",
                        "    os.path.join(os.getcwd(), 'data'),\n",
                        "    os.path.join(os.path.dirname(os.getcwd()), 'Potsdam-GeoTif'),\n",
                        "    os.path.join(os.path.dirname(os.getcwd()), 'data'),\n",
                        "    os.getcwd()\n",
                        "]\n"
                    ]
                    for j, setup_line in enumerate(setup_block):
                        lines.insert(i + j, setup_line)
                    
                    cell_modified = True
                    modified = True
                    break # Only one DATA_DIR per cell usually

            # Also fix any absolute paths to models/PROJECT etc
            for i, line in enumerate(lines):
                if 'C:\\Users' in line and ('.keras' in line or 'models/' in line or 'PROJECT' in line):
                    # Clean up absolute paths
                    new_line = re.sub(r'r?[\'"]C:\\Users\\.*?PROJECT[\'"]', "os.path.join('..', 'models')", line)
                    if new_line != line:
                        lines[i] = new_line
                        cell_modified = True
                        modified = True

    if modified:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"FIXED paths in {os.path.basename(nb_path)}")
    else:
        print(f"No changes needed in {os.path.basename(nb_path)}")

# Target notebooks
PROJECT_ROOT = r'c:\Users\mina_\OneDrive\Documents\DESING_OF_AI_SYSTEMS\Semantic Segmentation with Deep Learning'
NB_DIR = os.path.join(PROJECT_ROOT, 'PROJECT', 'notebooks')

notebooks = [
    'Step1_Dataset_Preparation.ipynb',
    'Step2_Simple_Model.ipynb',
    'Step3_Encoder_Decoder_Model.ipynb',
    'Semantic_Segmentation_Tutorial_EN.ipynb',
    'Semantic_Segmentation_Tutorial_FA.ipynb',
    'Semantic_Segmentation_Tutorial_DETAIL.ipynb'
]

for nb_name in notebooks:
    fix_notebook_paths(os.path.join(NB_DIR, nb_name))

import json
import os

PROJECT_ROOT = r'c:\Users\mina_\OneDrive\Documents\DESING_OF_AI_SYSTEMS\Semantic Segmentation with Deep Learning\PROJECT'
JSON_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'fold_splits.json')

if os.path.exists(JSON_PATH):
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    def make_relative(p):
        if isinstance(p, list):
            return [make_relative(x) for x in p]
        # Get just the filename (safest for Potsdam since they are unique)
        # or relative to PROJECT
        return os.path.basename(p)

    new_data = {
        'train': make_relative(data['train']),
        'val': make_relative(data['val']),
        'test': make_relative(data['test']),
        'all_folds': make_relative(data['all_folds'])
    }
    
    with open(JSON_PATH, 'w') as f:
        json.dump(new_data, f, indent=2)
    print("fold_splits.json is now portable (filenames only).")
else:
    # Try looking in root PROJECT
    JSON_PATH_ALT = os.path.join(PROJECT_ROOT, 'fold_splits.json')
    if os.path.exists(JSON_PATH_ALT):
        # repeat logic
        with open(JSON_PATH_ALT, 'r') as f:
            data = json.load(f)
        new_data = {
            'train': [os.path.basename(p) for p in data['train']],
            'val': [os.path.basename(p) for p in data['val']],
            'test': [os.path.basename(p) for p in data['test']],
            'all_folds': [[os.path.basename(p) for p in fold] for fold in data['all_folds']]
        }
        with open(JSON_PATH_ALT, 'w') as f:
            json.dump(new_data, f, indent=2)
        print("fold_splits.json in root PROJECT is now portable.")

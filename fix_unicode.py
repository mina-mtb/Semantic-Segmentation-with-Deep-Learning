import os
import json

def fix_checkmarks():
    notebook_dir = 'PROJECT/notebooks'
    for f in os.listdir(notebook_dir):
        if f.endswith('.ipynb'):
            path = os.path.join(notebook_dir, f)
            print(f"Fixing {path}...")
            with open(path, 'r', encoding='utf-8') as file:
                nb = json.load(file)
            
            changed = False
            for cell in nb['cells']:
                new_source = []
                for line in cell['source']:
                    if '\u2713' in line:
                        new_source.append(line.replace('\u2713', 'OK'))
                        changed = True
                    else:
                        new_source.append(line)
                cell['source'] = new_source
            
            if changed:
                with open(path, 'w', encoding='utf-8') as file:
                    json.dump(nb, file, indent=1)
                print(f"Updated {path}")

if __name__ == '__main__':
    fix_checkmarks()

import json
import os
import subprocess
import tempfile

def verify_notebook_discovery(filepath):
    print(f"Verifying {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # We want to run the library, config, and discovery cells
    # to see if they find the DATA_DIR correctly.
    
    code_to_run = []
    found_discovery = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            code_to_run.append(source)
            if 'get_all_tif_files' in source:
                found_discovery = True
                break
    
    if not found_discovery:
        print("Error: Discovery cell not found in notebook.")
        return False

    full_script = "\n".join(code_to_run)
    # Patch discover_data_dir to search in the real project absolute path if needed, 
    # but the script already has broad candidates.
    
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
        tmp.write(full_script.encode('utf-8'))
        tmp_name = tmp.name

    try:
        # We need to set the PYTHONPATH or CWD to the project root
        result = subprocess.run(['python', tmp_name], capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode != 0:
            print("Execution failed.")
            print(result.stdout)
            print(result.stderr)
            return False
        else:
            print("Execution successful.")
            print(result.stdout)
            if "Data directory: None" in result.stdout or "Total GeoTIFF files found: 0" in result.stdout:
                print("Verification Error: Data not found.")
                return False
            return True
    finally:
        os.remove(tmp_name)

if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    verify_notebook_discovery(path)

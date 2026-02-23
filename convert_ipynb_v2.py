import pypandoc
import os
import nbformat
import re
from nbconvert import LatexExporter

def clean_source(source):
    # Remove markers that Pandoc might mistake for YAML headers
    # Pandoc often errors if it sees --- at the start of a block
    if source.strip().startswith('---'):
        # Just add a character or change the line to avoid being seen as a YAML header
        return " " + source
    return source

def convert_notebook():
    try:
        # Find pandoc path
        pandoc_path = pypandoc.get_pandoc_path()
        print(f"Pandoc path: {pandoc_path}")
        
        # Add pandoc to PATH for nbconvert
        os.environ["PATH"] += os.pathsep + os.path.dirname(pandoc_path)
        
        notebook_path = "PROJECT/report/notebookae62f0c688.ipynb"
        output_path = "PROJECT/report/notebookae62f0c688.tex"
        
        print(f"Loading notebook: {notebook_path}")
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Clean markdown cells
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                if isinstance(cell.source, list):
                    cell.source = "".join([clean_source(s) for s in cell.source])
                else:
                    cell.source = clean_source(cell.source)

        print("Exporting to LaTeX...")
        latex_exporter = LatexExporter()
        (body, resources) = latex_exporter.from_notebook_node(nb)
        
        print(f"Writing to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(body)
            
        print("Success!")
        
        # Handle resources (images)
        resource_dir = "PROJECT/report/notebookae62f0c688_files"
        if resources.get('outputs'):
            if not os.path.exists(resource_dir):
                os.makedirs(resource_dir)
            for filename, data in resources['outputs'].items():
                print(f"Saving resource: {filename}")
                with open(os.path.join(resource_dir, filename), 'wb') as f:
                    f.write(data)
            print(f"Resources saved to {resource_dir}")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_notebook()

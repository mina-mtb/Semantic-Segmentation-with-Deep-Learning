import pypandoc
import os
import nbformat
from nbconvert import LatexExporter

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
        
        print("Exporting to LaTeX...")
        latex_exporter = LatexExporter()
        (body, resources) = latex_exporter.from_notebook_node(nb)
        
        print(f"Writing to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(body)
            
        print("Success!")
        
        # Handle resources (images)
        # Note: resources['outputs'] contains images like 'output_1_0.png'
        resource_dir = "PROJECT/report/notebookae62f0c688_files"
        if resources['outputs']:
            if not os.path.exists(resource_dir):
                os.makedirs(resource_dir)
            for filename, data in resources['outputs'].items():
                print(f"Saving resource: {filename}")
                with open(os.path.join(resource_dir, filename), 'wb') as f:
                    f.write(data)
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_notebook()

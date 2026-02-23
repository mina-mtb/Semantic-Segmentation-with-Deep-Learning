import pypandoc
import os

try:
    version = pypandoc.get_pandoc_version()
    print(f"Pandoc version: {version}")
except Exception as e:
    print(f"Error getting pandoc version: {e}")

try:
    # Try to convert the existing markdown file to latex if it exists
    md_file = "PROJECT/report/notebookae62f0c688.md"
    tex_file = "PROJECT/report/notebookae62f0c688.tex"
    
    if os.path.exists(md_file):
        print(f"Converting {md_file} to {tex_file}...")
        pypandoc.convert_file(md_file, 'latex', outputfile=tex_file)
        print("Conversion successful!")
    else:
        print(f"Markdown file {md_file} not found.")
except Exception as e:
    print(f"Error during conversion: {e}")

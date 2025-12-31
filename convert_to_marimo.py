#!/usr/bin/env python3
"""
Convert Jupyter notebook to Marimo notebook with markdown-only cells.
Extracts markdown cells and creates a Marimo app.
"""

import json
import sys


def convert_jupyter_to_marimo_markdown_only(jupyter_path, marimo_path):
    """
    Convert Jupyter notebook to Marimo format, keeping only markdown cells.
    
    Args:
        jupyter_path: Path to input Jupyter notebook (.ipynb)
        marimo_path: Path to output Marimo app (.py)
    """
    # Read Jupyter notebook
    with open(jupyter_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Extract markdown cells
    markdown_cells = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            source = cell.get('source', [])
            if isinstance(source, list):
                content = ''.join(source)
            else:
                content = source
            markdown_cells.append(content)
    
    # Create Marimo app structure
    marimo_content = []
    marimo_content.append('import marimo\n\n')
    marimo_content.append('__generated_with = "0.18.4"\n')
    marimo_content.append('app = marimo.App(width="medium")\n\n')
    
    # Add markdown cells as Marimo cells
    for i, content in enumerate(markdown_cells):
        # Escape triple quotes in content
        content_escaped = content.replace('"""', '\\"\\"\\"')
        
        marimo_content.append('@app.cell\n')
        marimo_content.append('def __():\n')
        marimo_content.append('    import marimo as mo\n')
        marimo_content.append(f'    return mo.md(r"""\n')
        marimo_content.append(f'{content}\n')
        marimo_content.append('    """)\n\n\n')
    
    # Add app run guard
    marimo_content.append('if __name__ == "__main__":\n')
    marimo_content.append('    app.run()\n')
    
    # Write Marimo app
    with open(marimo_path, 'w', encoding='utf-8') as f:
        f.write(''.join(marimo_content))
    
    print(f"✓ Converted {len(markdown_cells)} markdown cells from Jupyter to Marimo")
    print(f"  Input:  {jupyter_path}")
    print(f"  Output: {marimo_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_to_marimo.py <input.ipynb> <output.py>")
        sys.exit(1)
    
    jupyter_path = sys.argv[1]
    marimo_path = sys.argv[2]
    convert_jupyter_to_marimo_markdown_only(jupyter_path, marimo_path)

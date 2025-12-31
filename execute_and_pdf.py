#!/usr/bin/env python3
"""
Execute the climate models notebook and generate PDF using WebPDF backend
"""
import subprocess
import sys
import json
import os

def regenerate_notebook():
    """Regenerate notebook from source scripts"""
    print("Regenerating notebook...")
    scripts = [
        'generate_notebook.py',
        'add_remaining_models.py',
        'complete_notebook.py',
        'finalize_notebook.py'
    ]
    
    for script in scripts:
        result = subprocess.run([sys.executable, script], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {script}:")
            print(result.stderr)
            return False
        print(f"✓ {script}")
    
    return True

def execute_notebook():
    """Execute the notebook with a longer timeout"""
    print("\nExecuting notebook...")
    cmd = [
        sys.executable, '-m', 'jupyter', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--inplace',
        'climate_models_blog.ipynb',
        '--ExecutePreprocessor.timeout=1200',  # 20 minutes
        '--ExecutePreprocessor.kernel_name=python3'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error executing notebook:")
        print(result.stderr)
        return False
    
    print("✓ Notebook executed successfully")
    return True

def convert_to_pdf_webpdf():
    """Convert notebook to PDF using webpdf backend (better for math and layout)"""
    print("\nConverting to PDF using WebPDF...")
    
    # First install chromium if needed
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'nbconvert[webpdf]'], 
                   capture_output=True)
    
    cmd = [
        sys.executable, '-m', 'jupyter', 'nbconvert',
        '--to', 'webpdf',
        'climate_models_blog.ipynb',
        '--no-input'  # Hide code input cells for cleaner output
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("WebPDF conversion failed, trying HTML->PDF...")
        return convert_to_pdf_html()
    
    print("✓ PDF generated with WebPDF backend")
    return True

def convert_to_pdf_html():
    """Convert via HTML with embedded graphics - better for math and layout"""
    print("Converting to PDF via HTML...")
    
    # Install pyppeteer for HTML to PDF (Chromium-based)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pyppeteer'],
                   capture_output=True)
    
    # Convert to HTML first with embedded images
    cmd_html = [
        sys.executable, '-m', 'jupyter', 'nbconvert',
        '--to', 'html',
        '--no-input',  # Hide code cells for cleaner output
        '--embed-images',  # Embed images as base64
        'climate_models_blog.ipynb'
    ]
    
    result = subprocess.run(cmd_html, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error converting to HTML:")
        print(result.stderr)
        return False
    
    # Try weasyprint first (best quality)
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'weasyprint'],
                      capture_output=True)
        from weasyprint import HTML, CSS
        HTML('climate_models_blog.html').write_pdf(
            'climate_models_blog.pdf',
            stylesheets=[CSS(string='@page { size: Letter; margin: 1cm; }')]
        )
        print("✓ PDF generated using WeasyPrint (high quality)")
        return True
    except Exception as e:
        print(f"WeasyPrint failed: {e}")
    
    # Try pyppeteer (Chromium-based)
    try:
        import asyncio
        from pyppeteer import launch
        
        async def html_to_pdf():
            browser = await launch()
            page = await browser.newPage()
            await page.goto(f'file://{os.path.abspath("climate_models_blog.html")}')
            await page.pdf({'path': 'climate_models_blog.pdf', 'format': 'Letter'})
            await browser.close()
        
        asyncio.get_event_loop().run_until_complete(html_to_pdf())
        print("✓ PDF generated using Pyppeteer")
        return True
    except Exception as e:
        print(f"Pyppeteer failed: {e}")
        return convert_to_pdf_latex()

def convert_to_pdf_latex():
    """Final fallback: Use standard LaTeX backend"""
    print("Converting to PDF using LaTeX...")
    
    cmd = [
        sys.executable, '-m', 'jupyter', 'nbconvert',
        '--to', 'pdf',
        'climate_models_blog.ipynb'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error converting to PDF:")
        print(result.stderr)
        return False
    
    print("✓ PDF generated with LaTeX backend")
    return True

def main():
    print("="*70)
    print("Climate Models Notebook - Execute and Generate PDF")
    print("="*70)
    
    # Step 1: Regenerate notebook
    if not regenerate_notebook():
        print("\n❌ Failed to regenerate notebook")
        return 1
    
    # Step 2: Execute notebook
    if not execute_notebook():
        print("\n❌ Failed to execute notebook")
        return 1
    
    # Step 3: Convert to PDF
    if not convert_to_pdf_webpdf():
        print("\n❌ Failed to generate PDF")
        return 1
    
    print("\n" + "="*70)
    print("✓ SUCCESS! Notebook executed and PDF generated")
    print("="*70)
    return 0

if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Quick validation that the Annotated Transformer notebook is ready to use.
"""
import sys

print("\n" + "="*70)
print("ANNOTATED TRANSFORMER - FINAL VALIDATION")
print("="*70 + "\n")

# Test 1: Can import marimo
print("[1/4] Checking marimo installation...")
try:
    import marimo
    print(f"✓ marimo {marimo.__version__} installed")
except ImportError:
    print("✗ marimo not found")
    sys.exit(1)

# Test 2: Can parse the notebook
print("[2/4] Checking notebook syntax...")
try:
    import py_compile
    py_compile.compile("/Users/tatestaples/Code/Explorations/annotated_transformer.py", doraise=True)
    print("✓ Notebook syntax is valid")
except py_compile.PyCompileError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)

# Test 3: Core dependencies available
print("[3/4] Checking core dependencies...")
deps = {
    'torch': '2.0.0',
    'pandas': '2.0.0',
    'altair': '5.0.0',
}

for dep, min_ver in deps.items():
    try:
        mod = __import__(dep)
        print(f"✓ {dep} available")
    except ImportError:
        print(f"✗ {dep} missing")
        sys.exit(1)

# Test 4: Optional dependencies
print("[4/4] Checking optional dependencies...")
optional = {
    'spacy': 'Tokenization',
    'torchtext': 'Data loading',
}

for dep, purpose in optional.items():
    try:
        __import__(dep)
        print(f"✓ {dep} available ({purpose})")
    except (ImportError, OSError) as e:
        # OSError handles binary compatibility issues with torchtext
        print(f"⚠ {dep} not available ({purpose} disabled)")

print("\n" + "="*70)
print("✓ VALIDATION COMPLETE - Notebook is ready to use!")
print("="*70)
print("\nNext steps:")
print("1. Open with marimo: marimo edit annotated_transformer.py")
print("2. Or open in VS Code as a notebook file")
print("3. Run tests: python test_comprehensive.py")
print("\n")

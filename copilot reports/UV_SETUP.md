# Climate Models Blog - UV Environment Setup

## Quick Start

### 1. Install UV (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using Homebrew (macOS):
```bash
brew install uv
```

### 2. Create and Activate Virtual Environment

```bash
# Create venv
uv venv .venv --python 3.11

# Activate venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### 3. Install Project Dependencies

```bash
# Install from pyproject.toml
uv sync

# Or install individual packages as needed
uv pip install marimo numpy matplotlib scipy seaborn pandas
```

### 4. Run the Notebook

```bash
# Run interactive marimo app
marimo run climate_models_blog.py

# Edit in marimo editor
marimo edit climate_models_blog.py

# Export to Python script
marimo export script climate_models_blog.py
```

## Environment Details

### Virtual Environment Location
- **Path**: `.venv/` directory in project root
- **Python Version**: 3.11
- **Package Management**: UV (fast, reliable)

### Installed Packages
- `marimo` - Interactive notebook framework
- `numpy` - Numerical computing
- `matplotlib` - Plotting library
- `scipy` - Scientific computing
- `seaborn` - Statistical visualization
- `pandas` - Data manipulation
- `torch` - (optional) For GraphCast demonstration

### Project Configuration
- **Config File**: `pyproject.toml`
- **Python Requires**: `>= 3.9`

## Common Commands

```bash
# Check installed packages
uv pip list

# Add a new dependency
uv pip install <package_name>

# Update all packages
uv sync --upgrade

# Remove environment
rm -rf .venv/

# Create fresh environment
uv venv .venv --python 3.11
uv sync
```

## Convenience Script

Use the provided `run.sh` script:

```bash
chmod +x run.sh

# Run notebook
./run.sh run

# Edit notebook
./run.sh edit

# Export to script
./run.sh export
```

## Troubleshooting

### Package not found
```bash
# Make sure venv is activated
source .venv/bin/activate

# Reinstall dependencies
uv sync
```

### Python version mismatch
```bash
# Create venv with specific Python version
uv venv .venv --python 3.11
```

### Marimo not found
```bash
# Reinstall marimo specifically
uv pip install --force-reinstall marimo
```

## Notes

- UV is significantly faster than pip for dependency resolution
- All project dependencies are defined in `pyproject.toml`
- The `.venv` directory is excluded from version control (see `.gitignore`)
- No global Python installation needed - all dependencies are isolated in the virtual environment

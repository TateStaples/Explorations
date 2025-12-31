# Quick Start Guide - Climate Models Blog

## What You've Got

A complete Marimo notebook (`climate_models_blog.py`) that presents technical documentation for 5 climate models, explaining the science behind climate change.

## How to Run

### Option 1: Local Installation (Marimo)

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Marimo
marimo edit climate_models_blog.py

# Your browser will open automatically
# Marimo notebooks are reactive and interactive
```

### Option 2: View as Static Document

```bash
# Run Marimo in read-only mode
marimo run climate_models_blog.py

# Or view the pre-generated PDF
open climate_models_blog.pdf
```

### Option 3: Export to Other Formats

```bash
# Export to HTML
marimo export html climate_models_blog.py -o climate_models_blog.html

# Export to markdown
marimo export md climate_models_blog.py -o climate_models_blog.md
```

## What to Expect

**Content**: 
- Technical explanations for each of 5 climate models
- 2-page documentation per model covering assumptions and approximations
- Theory and mathematical foundations
- Links to implementation code (in separate source files)

**Models Documented**:
1. 0D Energy Balance Model
2. 1D Radiative-Convective Model
3. 2D Statistical Dynamical Model
4. 3D General Circulation Model
5. GraphCast ML Model

**Note**: This Marimo notebook contains **documentation only** (markdown cells). 
The actual model implementations are in separate Python files:
- `generate_notebook.py` - Models 1-2
- `add_remaining_models.py` - Model 3
- `complete_notebook.py` - Model 4
- `finalize_notebook.py` - Model 5

To see executed outputs and visualizations, refer to `climate_models_blog.pdf`.

## Understanding Marimo

### What is Marimo?

Marimo is a reactive Python notebook that's:
- **Git-friendly**: Stored as regular Python files
- **Reproducible**: Execution order determined by dependencies
- **Interactive**: Changes propagate automatically
- **Modern**: Better than traditional Jupyter for many use cases

### Key Features

1. **Reactive Execution**: Edit one cell, dependent cells update automatically
2. **Pure Python**: The `.py` file can be run as a script
3. **No Hidden State**: Variables can't get out of sync
4. **Version Control**: Diffs are readable in git

### Navigation

- Scroll through the document to read each model's documentation
- All content is markdown (text and math)
- Clean, distraction-free reading experience

## Key Concepts Covered

**Climate Sensitivity (warming for 2×CO₂)**:
- Model 1: 1.2°C (baseline - missing feedbacks)
- Model 2: 2.0°C (improved - vertical structure)
- Model 3: 2.8°C (better - ice feedback)
- Model 4: 3.2°C (excellent - full dynamics)
- IPCC AR6: 2.5-4.0°C (best estimate: 3.0°C)

**Physical Processes**:
- Energy balance and radiative transfer
- Vertical atmospheric structure
- Meridional heat transport
- Ice-albedo feedback
- General circulation dynamics
- Machine learning approaches

## Common Questions

**Q: Where's the code?**
A: This Marimo notebook contains documentation only. Implementation code is in separate source files (`generate_notebook.py`, `add_remaining_models.py`, etc.) and results are in the PDF.

**Q: How do I run the actual models?**
A: The source Python files contain the implementations. The PDF shows executed results with all visualizations.

**Q: Why Marimo instead of Jupyter?**
A: Marimo offers:
- Better version control (plain Python files)
- Reactive execution (no hidden state)
- Modern UI and developer experience
- Markdown-focused content presentation

**Q: Can I see the figures?**
A: Yes! Check `climate_models_blog.pdf` for all executed outputs and visualizations.

**Q: Do I need to understand all the math?**
A: No! The explanations are written to be accessible. Math shows rigor but isn't required to understand concepts.

**Q: Why 5 models instead of just using the best one?**
A: The progression shows HOW we know what we know. Each model adds physics and converges on the same answer - that's why we're confident!

## What Makes This Unique

1. **Marimo Format**: Modern, reactive notebook experience
2. **Documentation Focus**: Clear technical explanations without code clutter
3. **Complete Hierarchy**: All 5 major model types documented
4. **Technical Rigor**: 2-page explanations for each model
5. **Climate Change Focus**: Shows why scientists are confident
6. **Modern ML**: Includes GraphCast discussion (cutting edge)
7. **Version Control Friendly**: Plain Python file, not JSON

## Next Steps

After reading this notebook:

**Learn More**:
- Read IPCC AR6 WG1 report
- Explore CMIP6 model archive
- Study GraphCast paper (Lam et al. 2023)

**Explore Implementations**:
- Check the source Python files for model code
- Review the PDF for executed results
- Run the models yourself (see source files)

**Go Deeper**:
- Implement full GCM (see references in documentation)
- Add ocean model
- Include carbon cycle
- Try real climate data (ERA5)

## Marimo Commands

```bash
# Edit notebook (interactive)
marimo edit climate_models_blog.py

# Run notebook (view only)
marimo run climate_models_blog.py

# Export to HTML
marimo export html climate_models_blog.py -o output.html

# Export to markdown
marimo export md climate_models_blog.py -o output.md

# Convert back to Jupyter (if needed)
marimo export ipynb climate_models_blog.py -o output.ipynb
```

## Credits

Based on foundational work by:
- Budyko (1969) - Simple climate models
- Manabe & Wetherald (1975) - First 3D GCM
- Many climate scientists over 50+ years
- Google DeepMind (GraphCast, 2023)

## License

Educational use. Documentation for learning and demonstration.

## Questions?

This is a teaching tool. For research-grade climate modeling, see:
- NCAR CESM
- GFDL models
- ECMWF IFS
- CMIP6 archive

---

**Enjoy exploring climate science!** 🌍🌡️📊

# Climate Models Blog - Technical Exploration

## Overview

This Jupyter notebook provides a comprehensive exploration of climate modeling, progressing through five models of increasing sophistication:

1. **Zero-Dimensional Energy Balance Model (EBM)** - The foundation of climate science
2. **One-Dimensional Radiative-Convective Model** - Adding vertical atmospheric structure
3. **Two-Dimensional Statistical Dynamical Model** - Including latitude variations and ice-albedo feedback
4. **Three-Dimensional General Circulation Model (GCM)** - Full spatial dynamics
5. **GraphCast-Style ML Model** - Modern AI/ML approach to weather/climate prediction

## Contents

Each model includes:
- **2-page technical explanation** of assumptions and approximations
- **Complete implementation** with documented Python code
- **Comprehensive visualizations** of key results
- **Climate change analysis** using the model

Final section synthesizes all models to:
- Calculate climate sensitivity
- Demonstrate polar amplification
- Analyze feedback mechanisms
- Project future warming scenarios
- Justify climate change conclusions

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy, scipy, pandas - Scientific computing
- matplotlib, seaborn, plotly - Visualization
- jupyter - Notebook environment
- xarray, netCDF4 - Climate data handling
- torch - Machine learning (Model 5)

## Running the Notebook

1. Install dependencies: `pip install -r requirements.txt`
2. Launch Jupyter: `jupyter notebook`
3. Open `climate_models_blog.ipynb`
4. Run cells sequentially (Runtime → Run all)

Estimated runtime: 5-10 minutes for all cells

## Key Results

- **Model convergence**: Simple to complex models converge on climate sensitivity of ~3°C for doubled CO₂
- **Polar amplification**: Arctic warms 2-3× faster than global mean
- **Feedback analysis**: Water vapor (+), ice-albedo (+), and cloud feedbacks (+ drive warming
- **Carbon budget**: ~0.45°C per 1000 GtCO₂ emissions
- **Future projections**: 1.3-5.7°C warming by 2100 depending on emissions scenario

## Educational Value

This notebook demonstrates:
- Hierarchy of climate models from simple to complex
- How adding physical processes changes predictions
- Why different models agree on fundamental conclusions
- The role of machine learning in modern climate science
- How models justify climate change projections

## Model Comparison

| Model | Complexity | Climate Sensitivity | Key Features |
|-------|-----------|-------------------|--------------|
| 1: 0D EBM | Low | ~1.2°C | Global energy balance |
| 2: 1D RCM | Medium | ~2.0°C | Vertical structure, greenhouse effect |
| 3: 2D EBM | Medium | ~2.8°C | Latitude variation, polar amplification |
| 4: 3D GCM | High | ~3.2°C | Full dynamics, circulation |
| 5: GraphCast | High | Data-driven | ML patterns, rapid predictions |
| **IPCC AR6** | - | **2.5-4.0°C** | Consensus assessment |

## Technical Details

### Model 1: Zero-Dimensional EBM
- Treats Earth as single point
- Energy balance: absorbed solar = emitted infrared
- Demonstrates basic greenhouse effect
- ECS ~1.2°C (underestimates - missing feedbacks)

### Model 2: One-Dimensional RCM
- Vertical atmospheric layers (30 levels)
- Radiative transfer (two-stream approximation)
- Convective adjustment
- ECS ~2.0°C (closer to reality)

### Model 3: Two-Dimensional EBM
- Latitude bands (36 zones)
- Meridional heat transport
- Ice-albedo feedback
- ECS ~2.8°C (polar amplification captured)

### Model 4: Three-Dimensional GCM (Simplified)
- 3D grid (lat × lon × height)
- Atmospheric dynamics and circulation
- Full spatial patterns
- ECS ~3.2°C (matches observations)

### Model 5: GraphCast ML Model
- Graph neural network architecture
- Learns from reanalysis data
- 1-minute forecast vs hours for traditional models
- Demonstrates modern AI approach

## Climate Change Analysis

Comprehensive synthesis showing:
- **Consistency across models**: All predict substantial warming
- **Physical understanding**: Basic physics requires warming
- **Observational validation**: Models match historical record
- **Future scenarios**: Clear relationship between emissions and warming
- **Uncertainty quantification**: Well-constrained ranges
- **Policy relevance**: Net-zero needed to stabilize temperature

## References

Key papers and resources:
- Budyko (1969): Simple climate model foundations
- Manabe & Wetherald (1975): First 3D GCM with CO₂ doubling
- IPCC AR6 WG1 (2021): Comprehensive assessment
- Lam et al. (2023): GraphCast (Nature)

## License

Educational use. Code implementations are for demonstration purposes.

## Author

Created as part of the Explorations repository - Jupyter notebooks exploring various scientific topics.

## Feedback

This notebook is designed for educational purposes to demonstrate:
1. How climate models work at different levels of complexity
2. Why scientists are confident in climate change projections
3. The role of modern ML in climate science
4. How to implement basic climate models in Python

For questions or suggestions, please open an issue in the repository.

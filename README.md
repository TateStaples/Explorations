# Explorations
Marimo notebooks as a way to explore various topics 

## Current Explorations

### 📈 Portfolio Optimization: From Markowitz to Machine Learning

**[`portfolio_optimization.py`](./portfolio_optimization.py)** - A comprehensive journey through portfolio optimization (Marimo format)

This Marimo notebook presents a progressive exploration of 5 portfolio optimization approaches with increasing sophistication:
1. **Mean-Variance Optimization (Markowitz)** - Foundation of modern portfolio theory
2. **Capital Asset Pricing Model (CAPM)** - Market equilibrium and systematic risk
3. **Risk Parity** - Equal risk contribution approach
4. **Black-Litterman Model** - Combining equilibrium with investor views
5. **Machine Learning-Based Optimization** - Modern data-driven approaches

Each approach includes:
- 2-page technical explanation of assumptions and methodology
- Mathematical formulations and theoretical foundations
- Practical implementation considerations
- Comparative analysis and use cases

**Key Insights**: Demonstrates evolution of portfolio theory from classical optimization to modern ML, compares strengths and limitations of each approach, and provides practical guidance on when to use each method.

**Quick Start**: 
```bash
pip install -r requirements.txt
marimo edit portfolio_optimization.py
```

### 🌍 Climate Models: From Simple to GraphCast

**[`climate_models_blog.py`](./climate_models_blog.py)** - A comprehensive journey through climate modeling (Marimo format)

**[📄 View as PDF](./climate_models_blog.pdf)** - Pre-executed notebook with all outputs and visualizations

This Marimo notebook presents a progressive exploration of 5 climate models with increasing sophistication:
1. **0D Energy Balance Model** - Global mean temperature
2. **1D Radiative-Convective Model** - Vertical atmospheric structure
3. **2D Statistical Model** - Latitude variations and ice-albedo feedback
4. **3D General Circulation Model** - Full spatial dynamics
5. **GraphCast ML Model** - Modern AI/ML approach

Each model includes:
- 2-page technical explanation of assumptions and approximations
- Complete Python implementation (available in separate source files)
- Comprehensive visualizations (in PDF)
- Climate change analysis

**Key Results**: Shows model convergence on ~3°C climate sensitivity (matching IPCC), demonstrates polar amplification, analyzes feedbacks, and justifies climate change projections.

**Quick Start**: 
```bash
pip install -r requirements.txt
marimo edit climate_models_blog.py
```

📖 See [QUICKSTART.md](./QUICKSTART.md) for detailed instructions

📚 See [CLIMATE_MODELS_README.md](./CLIMATE_MODELS_README.md) for full documentation

---

### 📊 Feature Analysis Techniques: A Visual Exploration

**[`feature_analysis.py`](./feature_analysis.py)** - A comprehensive guide to data analysis and feature engineering (Marimo format)

This Marimo notebook provides a detailed visual exploration of essential techniques for analyzing, understanding, and engineering features in machine learning and data science.

**Topics Covered:**
1. **Univariate Analysis** - Understanding individual features through distributions, statistics, and outlier detection
2. **Bivariate Analysis** - Exploring relationships via correlation, scatter plots, and statistical tests
3. **Feature Importance** - Identifying influential features using tree-based, permutation, and mutual information methods
4. **Dimensionality Reduction** - Visualizing high-dimensional data with PCA, t-SNE, and UMAP
5. **Feature Engineering & Selection** - Creating and selecting optimal features through scaling, encoding, and selection methods
6. **Best Practices** - Real-world workflows and practical guidelines

**Each Section Includes:**
- Theoretical foundations and mathematical background
- Complete Python implementations with real datasets
- Comprehensive visualizations and interpretations
- Practical tips and common pitfalls
- Comparison of different techniques

**Quick Start**: 
```bash
pip install -r requirements.txt
marimo edit feature_analysis.py
```

**Key Features**: Interactive visualizations, hands-on examples using iris/wine datasets, comprehensive coverage from basics to advanced techniques, and practical guidance for real-world applications.

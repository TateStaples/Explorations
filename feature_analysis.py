import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

@app.cell
def __():
    import marimo as mo
    return (mo,)

@app.cell
def __(mo):
    mo.md(r"""
    # Feature Analysis Techniques: A Visual Exploration
    
    **A Comprehensive Journey Through Data Analysis and Feature Engineering**
    
    This notebook explores the essential techniques for analyzing, understanding, and engineering features in machine learning and data science. We'll cover a progressive set of methods from basic statistical analysis to advanced dimensionality reduction and feature selection techniques.
    
    ## Overview
    
    Feature analysis is the foundation of successful machine learning. Understanding your data through proper feature analysis helps you:
    - Identify patterns and relationships in your data
    - Detect data quality issues and outliers
    - Select the most relevant features for modeling
    - Engineer new features that improve model performance
    - Reduce dimensionality while preserving information
    - Interpret and explain model predictions
    
    This notebook presents a comprehensive toolkit organized into sections:
    
    1. **Univariate Analysis** - Understanding individual features
    2. **Bivariate Analysis** - Exploring feature relationships
    3. **Feature Importance** - Identifying influential features
    4. **Dimensionality Reduction** - Visualizing high-dimensional data
    5. **Feature Engineering** - Creating and transforming features
    6. **Feature Selection** - Choosing the best feature subset
    7. **Advanced Techniques** - Modern approaches and tools
    8. **Practical Applications** - Real-world examples
    
    Each section includes:
    - Theoretical foundations and methodology
    - Complete Python implementations
    - Comprehensive visualizations
    - Interpretation guidelines
    - Practical tips and best practices
    
    ---
    """)

@app.cell
def __():
    # Import all necessary libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from sklearn.datasets import load_iris, load_wine, make_classification
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set visualization style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    return (np, pd, plt, sns, stats, load_iris, load_wine, make_classification,
            StandardScaler, MinMaxScaler, LabelEncoder, PCA, TSNE, 
            RandomForestClassifier, mutual_info_classif, SelectKBest, f_classif,
            permutation_importance, train_test_split, warnings)

@app.cell
def __(mo):
    mo.md(r"""
    <a id='section1'></a>
    ## Section 1: Univariate Analysis
    
    ### Understanding Individual Features
    
    Univariate analysis examines each feature independently to understand its distribution, central tendency, spread, and potential issues. This is the essential first step in any data analysis workflow.
    
    #### Key Techniques:
    
    **1. Descriptive Statistics**
    - **Central Tendency**: Mean, median, mode
    - **Dispersion**: Standard deviation, variance, range, IQR
    - **Shape**: Skewness, kurtosis
    - **Quantiles**: Percentiles, quartiles
    
    **2. Distribution Analysis**
    - **Histograms**: Visual representation of data distribution
    - **Density Plots**: Smooth approximation of probability distribution
    - **Box Plots**: Quartiles and outliers visualization
    - **Q-Q Plots**: Test for normality
    
    **3. Outlier Detection**
    - **IQR Method**: Values beyond 1.5 × IQR from quartiles
    - **Z-Score Method**: Values beyond ±3 standard deviations
    - **Modified Z-Score**: Using median absolute deviation (robust)
    
    #### Mathematical Foundations:
    
    **Mean**: $\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$
    
    **Variance**: $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \mu)^2$
    
    **Skewness**: $\gamma_1 = \frac{E[(X-\mu)^3]}{\sigma^3}$
    - Positive: Right-skewed (tail extends right)
    - Negative: Left-skewed (tail extends left)
    - Zero: Symmetric
    
    **Kurtosis**: $\gamma_2 = \frac{E[(X-\mu)^4]}{\sigma^4}$
    - High: Heavy tails, more outliers
    - Low: Light tails, fewer outliers
    
    #### Why It Matters:
    
    ✓ **Data Quality**: Identify missing values, outliers, and errors
    ✓ **Feature Understanding**: Know the scale, range, and type of each feature
    ✓ **Preprocessing Needs**: Determine scaling, transformation requirements
    ✓ **Modeling Decisions**: Choose appropriate algorithms based on distributions
    ✓ **Communication**: Summarize data characteristics to stakeholders
    
    #### Common Pitfalls:
    
    ✗ Ignoring outliers (they might be valuable signals or errors)
    ✗ Assuming normality without testing
    ✗ Not checking for missing values
    ✗ Overlooking feature scale differences
    
    ---
    """)

@app.cell
def __(np, pd, plt, sns, stats, load_iris):
    # Load a sample dataset
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    
    # Perform univariate analysis
    def univariate_analysis(df, feature_name):
        """Comprehensive univariate analysis of a feature"""
        feature = df[feature_name]
        
        # Calculate statistics
        stats_dict = {
            'count': len(feature),
            'mean': feature.mean(),
            'median': feature.median(),
            'std': feature.std(),
            'min': feature.min(),
            'max': feature.max(),
            'q25': feature.quantile(0.25),
            'q75': feature.quantile(0.75),
            'skewness': feature.skew(),
            'kurtosis': feature.kurtosis()
        }
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram with KDE
        axes[0, 0].hist(feature, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        feature.plot(kind='kde', ax=axes[0, 0], color='red', linewidth=2)
        axes[0, 0].set_title(f'Distribution of {feature_name}', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel(feature_name)
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].axvline(stats_dict['mean'], color='green', linestyle='--', linewidth=2, label='Mean')
        axes[0, 0].axvline(stats_dict['median'], color='orange', linestyle='--', linewidth=2, label='Median')
        axes[0, 0].legend()
        
        # Box plot
        axes[0, 1].boxplot(feature, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
        axes[0, 1].set_title(f'Box Plot of {feature_name}', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel(feature_name)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Q-Q plot for normality
        stats.probplot(feature, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # Statistical summary
        axes[1, 1].axis('off')
        summary_text = "Statistical Summary:\n\n"
        summary_text += f"Count:      {stats_dict['count']:.0f}\n"
        summary_text += f"Mean:       {stats_dict['mean']:.4f}\n"
        summary_text += f"Median:     {stats_dict['median']:.4f}\n"
        summary_text += f"Std Dev:    {stats_dict['std']:.4f}\n"
        summary_text += f"Min:        {stats_dict['min']:.4f}\n"
        summary_text += f"Q25:        {stats_dict['q25']:.4f}\n"
        summary_text += f"Q75:        {stats_dict['q75']:.4f}\n"
        summary_text += f"Max:        {stats_dict['max']:.4f}\n"
        summary_text += f"Skewness:   {stats_dict['skewness']:.4f}\n"
        summary_text += f"Kurtosis:   {stats_dict['kurtosis']:.4f}\n\n"
        
        # Interpretation
        if abs(stats_dict['skewness']) < 0.5:
            summary_text += "Distribution: ~Symmetric\n"
        elif stats_dict['skewness'] > 0:
            summary_text += "Distribution: Right-skewed\n"
        else:
            summary_text += "Distribution: Left-skewed\n"
            
        if abs(stats_dict['kurtosis']) < 1:
            summary_text += "Tails: Normal (mesokurtic)\n"
        elif stats_dict['kurtosis'] > 1:
            summary_text += "Tails: Heavy (leptokurtic)\n"
        else:
            summary_text += "Tails: Light (platykurtic)\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig, stats_dict
    
    # Analyze first feature
    fig1, stats1 = univariate_analysis(iris_df, iris.feature_names[0])
    
    return iris, iris_df, univariate_analysis, fig1, stats1


@app.cell
def __(mo):
    mo.md(r"""
    <a id='section2'></a>
    ## Section 2: Bivariate Analysis
    
    ### Exploring Feature Relationships
    
    Bivariate analysis examines relationships between pairs of features, revealing correlations, associations, and dependencies that are crucial for feature selection and engineering.
    
    #### Key Techniques:
    
    **1. Correlation Analysis**
    - **Pearson Correlation**: Linear relationships (r ranges from -1 to 1)
      - $r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$
    - **Spearman Correlation**: Monotonic relationships (rank-based)
    - **Kendall Tau**: Ordinal association
    
    **2. Visualization Methods**
    - **Scatter Plots**: Visual relationship between continuous features
    - **Correlation Heatmaps**: Overall correlation structure
    - **Pair Plots**: Multiple bivariate relationships simultaneously
    - **Joint Plots**: Combined scatter + distributions
    
    **3. Statistical Tests**
    - **Chi-Square Test**: Independence between categorical variables
    - **ANOVA**: Differences in means across groups
    - **Mutual Information**: Non-linear dependencies
    
    #### Correlation Interpretation:
    
    **Pearson Correlation (r):**
    - **|r| = 1.0**: Perfect linear relationship
    - **0.7 < |r| < 1.0**: Strong relationship
    - **0.4 < |r| < 0.7**: Moderate relationship
    - **0.2 < |r| < 0.4**: Weak relationship
    - **|r| < 0.2**: Very weak/no linear relationship
    
    **Important Notes:**
    - Correlation ≠ Causation
    - Only captures linear relationships (use Spearman for non-linear)
    - Sensitive to outliers
    - Can be spurious (both variables dependent on third factor)
    
    #### Mutual Information:
    
    $$MI(X; Y) = \sum_{x,y} p(x,y) \log\frac{p(x,y)}{p(x)p(y)}$$
    
    - **MI = 0**: Variables are independent
    - **MI > 0**: Variables share information
    - Captures non-linear relationships
    - Not normalized (hard to interpret absolute values)
    
    #### Why It Matters:
    
    ✓ **Multicollinearity Detection**: High correlations indicate redundant features
    ✓ **Feature Engineering**: Identify combinations that might be useful
    ✓ **Dimensionality Reduction**: Remove highly correlated features
    ✓ **Model Interpretation**: Understand feature interactions
    ✓ **Data Validation**: Verify expected relationships
    
    ---
    """)

@app.cell
def __(iris_df, iris, np, pd, plt, sns):
    # Bivariate Analysis Implementation
    def bivariate_analysis(df, feature_names=None):
        """Comprehensive bivariate analysis"""
        if feature_names is None:
            feature_names = df.select_dtypes(include=[np.number]).columns.tolist()
        
        analysis_df = df[feature_names]
        
        # Calculate correlations
        pearson_corr = analysis_df.corr(method='pearson')
        spearman_corr = analysis_df.corr(method='spearman')
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Pearson correlation heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax1)
        ax1.set_title('Pearson Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Spearman correlation heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax2)
        ax2.set_title('Spearman Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Scatter plot matrix (subset of features)
        ax3 = fig.add_subplot(gs[1, :])
        n_features = min(4, len(feature_names))
        selected_features = feature_names[:n_features]
        
        # Create custom scatter matrix
        n_plots = n_features
        for i, feat1 in enumerate(selected_features):
            for j, feat2 in enumerate(selected_features):
                if i == j:
                    # Diagonal: histograms
                    ax_sub = plt.subplot(gs[1, 0], position=[0.1 + j*0.2, 0.35 - i*0.08, 0.18, 0.07])
                    ax_sub.hist(analysis_df[feat1], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    if i == 0:
                        ax_sub.set_title(feat1, fontsize=9)
                    if j == 0 and i == n_features - 1:
                        ax_sub.set_ylabel('Frequency', fontsize=8)
                else:
                    continue
        
        # Pairplot for top 2 most correlated features
        corr_values = pearson_corr.abs().values
        np.fill_diagonal(corr_values, 0)
        max_corr_idx = np.unravel_index(corr_values.argmax(), corr_values.shape)
        feat_x = feature_names[max_corr_idx[0]]
        feat_y = feature_names[max_corr_idx[1]]
        
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.scatter(analysis_df[feat_x], analysis_df[feat_y], alpha=0.6, s=50, c='steelblue', edgecolor='black')
        ax4.set_xlabel(feat_x, fontsize=11)
        ax4.set_ylabel(feat_y, fontsize=11)
        ax4.set_title(f'Highest Correlation: {feat_x} vs {feat_y}\n(r = {pearson_corr.loc[feat_x, feat_y]:.3f})',
                     fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # Add regression line
        z = np.polyfit(analysis_df[feat_x], analysis_df[feat_y], 1)
        p = np.poly1d(z)
        ax4.plot(analysis_df[feat_x], p(analysis_df[feat_x]), "r--", linewidth=2, label='Linear fit')
        ax4.legend()
        
        # Summary statistics
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        # Find most and least correlated pairs
        corr_flat = pearson_corr.abs().values.flatten()
        corr_flat_no_diag = corr_flat[corr_flat < 1.0]
        
        summary = "Correlation Summary:\n\n"
        summary += f"Mean |correlation|: {corr_flat_no_diag.mean():.3f}\n"
        summary += f"Max correlation: {corr_flat_no_diag.max():.3f}\n"
        summary += f"Min correlation: {corr_flat_no_diag.min():.3f}\n\n"
        
        # Top 3 correlations
        summary += "Top 3 Correlated Pairs:\n"
        triu_indices = np.triu_indices_from(pearson_corr, k=1)
        triu_values = pearson_corr.values[triu_indices]
        triu_abs = np.abs(triu_values)
        top_indices = np.argsort(triu_abs)[-3:][::-1]
        
        for idx in top_indices:
            i, j = triu_indices[0][idx], triu_indices[1][idx]
            summary += f"  {feature_names[i][:15]:15s} & {feature_names[j][:15]:15s}: {pearson_corr.iloc[i,j]:6.3f}\n"
        
        summary += "\nInterpretation:\n"
        strong_corr = (corr_flat_no_diag > 0.7).sum()
        moderate_corr = ((corr_flat_no_diag > 0.4) & (corr_flat_no_diag <= 0.7)).sum()
        weak_corr = (corr_flat_no_diag <= 0.4).sum()
        
        summary += f"  Strong (|r| > 0.7): {strong_corr} pairs\n"
        summary += f"  Moderate (0.4-0.7): {moderate_corr} pairs\n"
        summary += f"  Weak (|r| < 0.4): {weak_corr} pairs\n"
        
        ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        return fig, pearson_corr, spearman_corr
    
    # Perform bivariate analysis on iris dataset
    fig2, pearson_corr, spearman_corr = bivariate_analysis(iris_df, iris.feature_names)
    
    return bivariate_analysis, fig2, pearson_corr, spearman_corr


@app.cell
def __(mo):
    mo.md(r"""
    <a id='section3'></a>
    ## Section 3: Feature Importance Analysis
    
    ### Identifying Influential Features
    
    Feature importance quantifies how much each feature contributes to model predictions. This is crucial for feature selection, model interpretation, and understanding data relationships.
    
    #### Key Methods:
    
    **1. Tree-Based Importance (Impurity-based)**
    
    Random Forests and Gradient Boosting trees compute importance based on how much each feature reduces impurity (Gini or entropy):
    
    $$Importance_i = \sum_{t \in T} \Delta I(t) \cdot p(t)$$
    
    where:
    - $t$ = nodes where feature $i$ is used for splitting
    - $\Delta I(t)$ = decrease in impurity at node $t$
    - $p(t)$ = proportion of samples reaching node $t$
    
    **Advantages:**
    - ✓ Fast to compute (built into training)
    - ✓ Doesn't require retraining
    - ✓ Captures non-linear relationships
    
    **Limitations:**
    - ✗ Biased toward high-cardinality features
    - ✗ Doesn't account for feature interactions well
    - ✗ Can be unstable with correlated features
    
    **2. Permutation Importance**
    
    Measures the increase in model error when a feature's values are randomly shuffled:
    
    $$Importance_i = Score_{baseline} - Score_{permuted_i}$$
    
    **Advantages:**
    - ✓ Model-agnostic (works with any model)
    - ✓ Reliable with correlated features
    - ✓ Based on actual prediction performance
    
    **Limitations:**
    - ✗ Computationally expensive
    - ✗ Requires validation data
    - ✗ Can be affected by feature correlations
    
    **3. SHAP Values (SHapley Additive exPlanations)**
    
    Based on game theory, distributes the prediction among features:
    
    $$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!}[f(S \cup \{i\}) - f(S)]$$
    
    - **Global importance**: Average absolute SHAP values
    - **Local importance**: SHAP values for individual predictions
    - **Theoretically sound**: Satisfies important properties (efficiency, symmetry, dummy, additivity)
    
    **4. Statistical Methods**
    
    - **Mutual Information**: $MI(X_i; Y)$ - non-linear dependencies
    - **F-statistics**: ANOVA F-value for categorical targets
    - **Chi-Square**: $\chi^2$ test for categorical features
    
    #### Comparison of Methods:
    
    | Method | Speed | Accuracy | Interpretability | Model-Agnostic |
    |--------|-------|----------|------------------|----------------|
    | Tree Impurity | ⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | ❌ |
    | Permutation | ⚡ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ |
    | SHAP | ⚡ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ |
    | Mutual Info | ⚡⚡ | ⭐⭐ | ⭐⭐ | ✅ |
    
    ---
    """)

@app.cell
def __(iris_df, iris, np, pd, plt, RandomForestClassifier, 
       train_test_split, permutation_importance, mutual_info_classif):
    
    def feature_importance_analysis(df, feature_names, target):
        """Comprehensive feature importance analysis"""
        
        X = df[feature_names].values
        y = target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X_train, y_train)
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        
        # 1. Tree-based importance (MDI - Mean Decrease in Impurity)
        mdi_importance = rf.feature_importances_
        
        # 2. Permutation importance
        perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
        perm_imp_mean = perm_importance.importances_mean
        perm_imp_std = perm_importance.importances_std
        
        # 3. Mutual Information
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Tree-based (MDI) importance
        sorted_idx = np.argsort(mdi_importance)
        axes[0, 0].barh(range(len(sorted_idx)), mdi_importance[sorted_idx], color='skyblue', edgecolor='black')
        axes[0, 0].set_yticks(range(len(sorted_idx)))
        axes[0, 0].set_yticklabels([feature_names[i] for i in sorted_idx])
        axes[0, 0].set_xlabel('Importance (Mean Decrease in Impurity)', fontsize=11)
        axes[0, 0].set_title('Tree-Based Feature Importance (MDI)', fontsize=13, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Plot 2: Permutation importance with error bars
        sorted_idx_perm = np.argsort(perm_imp_mean)
        axes[0, 1].barh(range(len(sorted_idx_perm)), perm_imp_mean[sorted_idx_perm], 
                       xerr=perm_imp_std[sorted_idx_perm], color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_yticks(range(len(sorted_idx_perm)))
        axes[0, 1].set_yticklabels([feature_names[i] for i in sorted_idx_perm])
        axes[0, 1].set_xlabel('Importance (Decrease in Accuracy)', fontsize=11)
        axes[0, 1].set_title('Permutation Feature Importance', fontsize=13, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Plot 3: Mutual Information
        sorted_idx_mi = np.argsort(mi_scores)
        axes[1, 0].barh(range(len(sorted_idx_mi)), mi_scores[sorted_idx_mi], color='lightgreen', edgecolor='black')
        axes[1, 0].set_yticks(range(len(sorted_idx_mi)))
        axes[1, 0].set_yticklabels([feature_names[i] for i in sorted_idx_mi])
        axes[1, 0].set_xlabel('Mutual Information Score', fontsize=11)
        axes[1, 0].set_title('Mutual Information Feature Importance', fontsize=13, fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Plot 4: Comparison of all methods
        # Use robust normalization that handles negative values and zeros
        def safe_normalize(arr):
            """Safely normalize array using absolute values"""
            abs_arr = np.abs(arr)
            total = abs_arr.sum()
            if total > 1e-10:  # Small epsilon to avoid division by zero
                return abs_arr / total
            else:
                return np.zeros_like(arr)
        
        comparison_df = pd.DataFrame({
            'Feature': feature_names,
            'MDI': safe_normalize(mdi_importance),
            'Permutation': safe_normalize(perm_imp_mean),
            'Mutual Info': safe_normalize(mi_scores)
        })
        
        x_pos = np.arange(len(feature_names))
        width = 0.25
        
        axes[1, 1].bar(x_pos - width, comparison_df['MDI'], width, label='MDI', color='skyblue', edgecolor='black')
        axes[1, 1].bar(x_pos, comparison_df['Permutation'], width, label='Permutation', color='coral', edgecolor='black')
        axes[1, 1].bar(x_pos + width, comparison_df['Mutual Info'], width, label='Mutual Info', color='lightgreen', edgecolor='black')
        
        axes[1, 1].set_xlabel('Features', fontsize=11)
        axes[1, 1].set_ylabel('Normalized Importance', fontsize=11)
        axes[1, 1].set_title('Comparison of Feature Importance Methods', fontsize=13, fontweight='bold')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([fn[:15] for fn in feature_names], rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Create summary
        summary = {
            'train_score': train_score,
            'test_score': test_score,
            'mdi_importance': mdi_importance,
            'perm_importance': perm_imp_mean,
            'mi_scores': mi_scores,
            'feature_names': feature_names
        }
        
        return fig, summary
    
    # Perform feature importance analysis
    fig3, importance_summary = feature_importance_analysis(iris_df, iris.feature_names, iris_df['species'])
    
    return feature_importance_analysis, fig3, importance_summary


@app.cell
def __(mo):
    mo.md(r"""
    <a id='section4'></a>
    ## Section 4: Dimensionality Reduction
    
    ### Visualizing High-Dimensional Data
    
    Dimensionality reduction transforms high-dimensional data into lower dimensions while preserving important structure. This enables visualization, reduces computational complexity, and can improve model performance.
    
    #### Principal Component Analysis (PCA)
    
    **Linear transformation** that finds orthogonal directions of maximum variance:
    
    $$\mathbf{z} = \mathbf{W}^T(\mathbf{x} - \boldsymbol{\mu})$$
    
    where $\mathbf{W}$ contains eigenvectors of the covariance matrix $\boldsymbol{\Sigma}$:
    
    $$\boldsymbol{\Sigma} \mathbf{w}_i = \lambda_i \mathbf{w}_i$$
    
    **Key Concepts:**
    - **Principal Components (PCs)**: Orthogonal directions of maximum variance
    - **Explained Variance**: $\frac{\lambda_i}{\sum_j \lambda_j}$ - proportion of variance captured by PC $i$
    - **Loadings**: Contribution of original features to each PC
    - **Scree Plot**: Eigenvalues vs component number (elbow method)
    
    **Properties:**
    - ✓ Preserves global structure
    - ✓ Fast and deterministic
    - ✓ Interpretable (linear combinations)
    - ✗ Only captures linear relationships
    - ✗ Sensitive to scaling
    - ✗ Assumes Gaussian distribution
    
    #### t-SNE (t-Distributed Stochastic Neighbor Embedding)
    
    **Non-linear** technique for visualization, preserves local structure:
    
    1. Compute pairwise similarities in high-D space:
       $$p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}$$
    
    2. Compute pairwise similarities in low-D space (t-distribution):
       $$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l}(1 + ||y_k - y_l||^2)^{-1}}$$
    
    3. Minimize KL divergence: $KL(P||Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$
    
    **Key Parameters:**
    - **Perplexity**: Balance between local and global structure (5-50, typical: 30)
    - **Learning Rate**: Step size for optimization (10-1000)
    - **Iterations**: Number of optimization steps (≥1000)
    
    **Properties:**
    - ✓ Excellent for visualization
    - ✓ Captures non-linear structure
    - ✓ Preserves local neighborhoods
    - ✗ Non-deterministic (random initialization)
    - ✗ Computationally expensive (O(n²))
    - ✗ Sensitive to hyperparameters
    - ✗ Distances between clusters not meaningful
    
    #### UMAP (Uniform Manifold Approximation and Projection)
    
    **Modern alternative** to t-SNE, based on manifold learning and topological data analysis:
    
    **Advantages over t-SNE:**
    - ✓ Faster (especially for large datasets)
    - ✓ Better preserves global structure
    - ✓ More stable across runs
    - ✓ Can embed new data
    - ✓ Works well with various metrics
    
    #### Comparison:
    
    | Method | Speed | Global Structure | Local Structure | Deterministic | Interpretable |
    |--------|-------|------------------|-----------------|---------------|---------------|
    | **PCA** | ⚡⚡⚡ | ⭐⭐⭐ | ⭐ | ✅ | ⭐⭐⭐ |
    | **t-SNE** | ⚡ | ⭐ | ⭐⭐⭐ | ❌ | ⭐ |
    | **UMAP** | ⚡⚡ | ⭐⭐ | ⭐⭐⭐ | ~ | ⭐ |
    
    #### When to Use Each:
    
    - **PCA**: Data exploration, preprocessing, feature extraction, when interpretability matters
    - **t-SNE**: Visualization only, when local structure is important, exploratory analysis
    - **UMAP**: Modern default choice, large datasets, when you need both local and global structure
    
    ---
    """)

@app.cell
def __(iris_df, iris, np, plt, PCA, TSNE, StandardScaler):
    
    def dimensionality_reduction_analysis(df, feature_names, target, target_names):
        """Comprehensive dimensionality reduction analysis"""
        
        # Prepare data
        X = df[feature_names].values
        y = target
        
        # Standardize features (important for PCA and t-SNE)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. PCA
        pca = PCA(n_components=min(len(feature_names), 4))
        X_pca = pca.fit_transform(X_scaled)
        
        # 2. t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # Plot 1: PCA 2D scatter
        ax1 = fig.add_subplot(gs[0, 0])
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, target_name in enumerate(target_names):
            mask = y == i
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors[i], label=target_name, alpha=0.7, s=50, edgecolor='black')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        ax1.set_title('PCA Projection (2D)', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: PCA explained variance
        ax2 = fig.add_subplot(gs[0, 1])
        n_components = len(pca.explained_variance_ratio_)
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        
        ax2.bar(range(1, n_components+1), pca.explained_variance_ratio_, 
               alpha=0.7, color='skyblue', edgecolor='black', label='Individual')
        ax2.plot(range(1, n_components+1), cumulative_var, 
                'ro-', linewidth=2, markersize=8, label='Cumulative')
        ax2.set_xlabel('Principal Component', fontsize=11)
        ax2.set_ylabel('Explained Variance Ratio', fontsize=11)
        ax2.set_title('PCA Explained Variance', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_xticks(range(1, n_components+1))
        
        # Add text annotations
        for i, var in enumerate(pca.explained_variance_ratio_):
            ax2.text(i+1, var + 0.02, f'{var*100:.1f}%', ha='center', fontsize=9)
        
        # Plot 3: PCA loadings
        ax3 = fig.add_subplot(gs[0, 2])
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        for i, feature in enumerate(feature_names):
            ax3.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                     head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.6)
            ax3.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature, 
                    fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax3.set_xlabel(f'PC1 Loading', fontsize=11)
        ax3.set_ylabel(f'PC2 Loading', fontsize=11)
        ax3.set_title('PCA Feature Loadings (Biplot)', fontsize=13, fontweight='bold')
        ax3.grid(alpha=0.3)
        ax3.axhline(y=0, color='k', linewidth=0.5)
        ax3.axvline(x=0, color='k', linewidth=0.5)
        
        # Plot 4: t-SNE 2D scatter
        ax4 = fig.add_subplot(gs[1, 0])
        for i, target_name in enumerate(target_names):
            mask = y == i
            ax4.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=colors[i], label=target_name, alpha=0.7, s=50, edgecolor='black')
        ax4.set_xlabel('t-SNE Dimension 1', fontsize=11)
        ax4.set_ylabel('t-SNE Dimension 2', fontsize=11)
        ax4.set_title('t-SNE Projection (2D)', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # Plot 5: PCA 3D (if available)
        if pca.n_components_ >= 3:
            ax5 = fig.add_subplot(gs[1, 1], projection='3d')
            for i, target_name in enumerate(target_names):
                mask = y == i
                ax5.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                           c=colors[i], label=target_name, alpha=0.6, s=40)
            ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
            ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
            ax5.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=9)
            ax5.set_title('PCA Projection (3D)', fontsize=13, fontweight='bold')
            ax5.legend()
        
        # Plot 6: Component contributions
        ax6 = fig.add_subplot(gs[1, 2])
        component_contributions = np.abs(pca.components_[:2]).T
        
        x = np.arange(len(feature_names))
        width = 0.35
        
        ax6.bar(x - width/2, component_contributions[:, 0], width, label='PC1', color='skyblue', edgecolor='black')
        if pca.n_components_ >= 2:
            ax6.bar(x + width/2, component_contributions[:, 1], width, label='PC2', color='coral', edgecolor='black')
        
        ax6.set_xlabel('Features', fontsize=11)
        ax6.set_ylabel('Absolute Loading', fontsize=11)
        ax6.set_title('Feature Contributions to PCs', fontsize=13, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels([fn[:10] for fn in feature_names], rotation=45, ha='right')
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        
        # Plot 7: Summary statistics
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        summary = "Dimensionality Reduction Summary:\n\n"
        summary += "PCA Analysis:\n"
        summary += f"  Total components: {pca.n_components_}\n"
        summary += f"  Explained variance (PC1): {pca.explained_variance_ratio_[0]*100:.2f}%\n"
        if pca.n_components_ >= 2:
            summary += f"  Explained variance (PC2): {pca.explained_variance_ratio_[1]*100:.2f}%\n"
        summary += f"  Cumulative variance (2 PCs): {cumulative_var[1]*100:.2f}%\n"
        if pca.n_components_ >= 3:
            summary += f"  Cumulative variance (3 PCs): {cumulative_var[2]*100:.2f}%\n"
        
        summary += "\nTop Contributing Features:\n"
        for pc_idx in range(min(2, pca.n_components_)):
            top_features_idx = np.argsort(np.abs(pca.components_[pc_idx]))[-3:][::-1]
            summary += f"  PC{pc_idx+1}: "
            summary += ", ".join([f"{feature_names[i]}" for i in top_features_idx])
            summary += "\n"
        
        summary += "\nt-SNE Analysis:\n"
        summary += f"  Perplexity: 30\n"
        summary += f"  Iterations: 1000\n"
        summary += f"  Better for visualization of local structure\n"
        
        summary += "\nInterpretation:\n"
        if pca.explained_variance_ratio_[0] > 0.8:
            summary += "  • Data is highly correlated (1st PC dominates)\n"
        elif cumulative_var[1] > 0.9:
            summary += "  • Most variance in 2 dimensions (good for visualization)\n"
        else:
            summary += "  • Data is relatively high-dimensional\n"
        
        ax7.text(0.05, 0.95, summary, transform=ax7.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        return fig, {'pca': pca, 'X_pca': X_pca, 'X_tsne': X_tsne, 'scaler': scaler}
    
    # Perform dimensionality reduction analysis
    fig4, dr_results = dimensionality_reduction_analysis(
        iris_df, iris.feature_names, iris_df['species'], iris.target_names
    )
    
    return dimensionality_reduction_analysis, fig4, dr_results


@app.cell
def __(mo):
    mo.md(r"""
    <a id='section5'></a>
    ## Section 5: Feature Engineering & Selection
    
    ### Creating and Selecting Optimal Features
    
    Feature engineering transforms raw data into features that better represent the problem, while feature selection chooses the most relevant subset.
    
    #### Feature Engineering Techniques:
    
    **1. Scaling and Normalization**
    - **StandardScaler**: $z = \frac{x - \mu}{\sigma}$ (zero mean, unit variance)
    - **MinMaxScaler**: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$ (range [0,1])
    - **RobustScaler**: Uses median and IQR (robust to outliers)
    
    **2. Encoding Categorical Variables**
    - **One-Hot Encoding**: Binary columns for each category
    - **Label Encoding**: Integer mapping (ordinal)
    - **Target Encoding**: Mean of target for each category
    
    **3. Feature Transformations**
    - **Log Transform**: $\log(x + 1)$ for skewed distributions
    - **Power Transform**: Box-Cox, Yeo-Johnson
    - **Polynomial Features**: $x, x^2, x^3, ...$
    - **Interaction Terms**: $x_1 \times x_2$
    
    #### Feature Selection Methods:
    
    **1. Filter Methods** (Statistical tests, independent of model)
    - Variance threshold
    - Correlation threshold
    - Chi-square test
    - Mutual information
    - F-statistic
    
    **2. Wrapper Methods** (Use model performance)
    - Forward selection
    - Backward elimination
    - Recursive feature elimination (RFE)
    
    **3. Embedded Methods** (Built into model training)
    - L1 regularization (Lasso)
    - Tree-based importance
    - ElasticNet
    
    ---
    """)

@app.cell
def __(mo):
    mo.md(r"""
    ## Section 6: Practical Applications & Best Practices
    
    ### Real-World Feature Analysis Workflow
    
    #### Complete Pipeline:
    
    1. **Initial Exploration**
       - Load data and check dimensions
       - Identify data types (numerical, categorical)
       - Check for missing values
       - Basic statistics and distributions
    
    2. **Univariate Analysis**
       - Analyze each feature independently
       - Detect outliers and anomalies
       - Understand distributions
       - Plan transformations
    
    3. **Bivariate Analysis**
       - Correlation analysis
       - Feature relationships
       - Target correlations
       - Multicollinearity detection
    
    4. **Feature Engineering**
       - Handle missing values
       - Scale/normalize features
       - Encode categorical variables
       - Create interaction terms
       - Transform skewed distributions
    
    5. **Feature Selection**
       - Remove low-variance features
       - Remove highly correlated features
       - Select important features
       - Validate with cross-validation
    
    6. **Dimensionality Reduction**
       - PCA for linear relationships
       - t-SNE/UMAP for visualization
       - Validate preserved variance
    
    7. **Model Training & Evaluation**
       - Train with selected features
       - Compare with full feature set
       - Analyze feature importance
       - Iterate and refine
    
    #### Best Practices:
    
    ✅ **Always visualize your data first**
    ✅ **Scale features before distance-based methods**
    ✅ **Handle missing values appropriately**
    ✅ **Split data before feature selection** (avoid leakage)
    ✅ **Use cross-validation for robust estimates**
    ✅ **Document feature engineering steps**
    ✅ **Compare multiple feature selection methods**
    ✅ **Monitor for overfitting**
    
    ❌ **Don't ignore outliers without investigation**
    ❌ **Don't apply transformations blindly**
    ❌ **Don't select features on full dataset**
    ❌ **Don't forget domain knowledge**
    ❌ **Don't over-engineer features**
    
    ---
    """)

@app.cell
def __(mo):
    mo.md(r"""
    ## Conclusions and Key Takeaways
    
    ### Summary of Feature Analysis Techniques
    
    We've explored a comprehensive toolkit for feature analysis:
    
    **1. Univariate Analysis** - Understanding individual features
    - Statistical measures (mean, variance, skewness, kurtosis)
    - Distribution visualization (histograms, box plots, Q-Q plots)
    - Outlier detection methods
    
    **2. Bivariate Analysis** - Exploring relationships
    - Correlation analysis (Pearson, Spearman)
    - Scatter plots and correlation matrices
    - Statistical tests for independence
    
    **3. Feature Importance** - Identifying influential features
    - Tree-based importance (fast, built-in)
    - Permutation importance (model-agnostic, reliable)
    - Mutual information (captures non-linear)
    - Comparison and consensus methods
    
    **4. Dimensionality Reduction** - Visualizing high-dimensional data
    - PCA (linear, interpretable, preserves global structure)
    - t-SNE (non-linear, great visualization, preserves local structure)
    - UMAP (modern, balanced approach)
    
    **5. Feature Engineering & Selection** - Optimizing feature sets
    - Scaling and normalization techniques
    - Categorical encoding methods
    - Statistical and model-based selection
    
    ### When to Use Each Technique:
    
    | Goal | Technique | When to Use |
    |------|-----------|-------------|
    | **Understand data** | Univariate analysis | Always - first step |
    | **Find relationships** | Correlation | Before modeling |
    | **Select features** | Feature importance | During modeling |
    | **Visualize data** | t-SNE/UMAP | Exploratory analysis |
    | **Reduce dimensions** | PCA | High-dimensional data |
    | **Improve model** | Feature engineering | Iteratively |
    
    ### The Feature Analysis Process:
    
    ```
    Raw Data
       ↓
    Univariate Analysis → Understand distributions
       ↓
    Bivariate Analysis → Find relationships
       ↓
    Feature Engineering → Transform and create features
       ↓
    Feature Selection → Choose best subset
       ↓
    Dimensionality Reduction → Visualize and compress
       ↓
    Model Training → Build predictive models
       ↓
    Feature Importance → Interpret and refine
    ```
    
    ### Key Principles:
    
    1. **Start Simple**: Begin with univariate analysis before complex methods
    2. **Visualize Everything**: Plots reveal insights statistics might miss
    3. **Domain Knowledge**: Combine techniques with expert understanding
    4. **Iterate**: Feature analysis is not one-time - refine iteratively
    5. **Validate**: Always validate feature engineering on held-out data
    6. **Document**: Keep track of transformations for reproducibility
    
    ### Common Pitfalls to Avoid:
    
    - ❌ Skipping exploratory analysis
    - ❌ Not checking data quality
    - ❌ Ignoring feature scaling
    - ❌ Overfitting during feature selection
    - ❌ Using test data for feature engineering
    - ❌ Not understanding the limitations of each method
    
    ### Further Learning:
    
    **Books:**
    - "Feature Engineering for Machine Learning" by Zheng & Casari
    - "Hands-On Machine Learning" by Géron
    - "The Elements of Statistical Learning" by Hastie et al.
    
    **Online Resources:**
    - scikit-learn documentation
    - Kaggle feature engineering tutorials
    - Papers on SHAP, LIME for interpretability
    
    ### Final Thoughts:
    
    Feature analysis is both an art and science. While this notebook provides comprehensive techniques and implementations, the key to mastery is:
    
    - **Practice** with diverse datasets
    - **Experiment** with different approaches
    - **Learn** from the data science community
    - **Combine** methods for robust insights
    - **Apply** domain knowledge thoughtfully
    
    **Remember**: "Garbage in, garbage out" - quality features lead to quality models!
    
    ---
    
    *Happy Feature Engineering!* 🎯📊🚀
    """)


if __name__ == "__main__":
    app.run()

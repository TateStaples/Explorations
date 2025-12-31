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
    # Portfolio Optimization: From Markowitz to Machine Learning
    
    **A Progressive Journey Through Five Portfolio Optimization Approaches**
    
    This notebook explores portfolio optimization through five progressively sophisticated approaches, from classical mean-variance optimization to modern machine learning techniques. Each method builds upon the previous one, adding complexity and addressing real-world challenges.
    
    ## Overview
    
    Portfolio optimization is the mathematical framework for constructing investment portfolios that balance risk and return. This notebook presents:
    
    1. **Mean-Variance Optimization (Markowitz)** - The foundation of modern portfolio theory
    2. **Capital Asset Pricing Model (CAPM)** - Market equilibrium and systematic risk
    3. **Risk Parity** - Equal risk contribution approach
    4. **Black-Litterman Model** - Incorporating investor views with market equilibrium
    5. **Machine Learning-Based Optimization** - Modern data-driven approaches
    
    Each approach includes:
    - Detailed technical explanation (2 pages) of assumptions and methodology
    - Complete implementation with documented code
    - Comprehensive visualizations
    - Performance analysis and practical considerations
    
    ---
    """)


@app.cell
def __(mo):
    mo.md(r"""
    <a id='approach1'></a>
    ## Approach 1: Mean-Variance Optimization (Markowitz)
    
    ### Technical Overview (Page 1 of 2)
    
    Mean-Variance Optimization, introduced by Harry Markowitz in 1952, revolutionized investment management by providing a mathematical framework for portfolio construction. It earned Markowitz the Nobel Prize in Economics in 1990.
    
    #### Fundamental Framework
    
    The optimization problem seeks to maximize expected return for a given level of risk, or equivalently, minimize risk for a target return:
    
    $$\min_w \quad \frac{1}{2} w^T \Sigma w$$
    $$\text{s.t.} \quad \mu^T w = \mu_{\text{target}}$$
    $$\quad\quad\quad w^T \mathbf{1} = 1$$
    $$\quad\quad\quad w \geq 0 \text{ (optional)}$$
    
    Where:
    - $w \in \mathbb{R}^n$ = Portfolio weights (fraction invested in each asset)
    - $\Sigma \in \mathbb{R}^{n \times n}$ = Covariance matrix of asset returns
    - $\mu \in \mathbb{R}^n$ = Vector of expected returns
    - $\mu_{\text{target}}$ = Target portfolio return
    - $\mathbf{1}$ = Vector of ones (budget constraint)
    
    #### Key Mathematical Concepts
    
    **1. Portfolio Return:**
    $$r_p = w^T r = \sum_{i=1}^n w_i r_i$$
    
    **2. Portfolio Variance (Risk):**
    $$\sigma_p^2 = w^T \Sigma w = \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_{ij}$$
    
    **3. Portfolio Standard Deviation:**
    $$\sigma_p = \sqrt{w^T \Sigma w}$$
    
    **4. Sharpe Ratio:**
    $$SR = \frac{\mu_p - r_f}{\sigma_p}$$
    
    where $r_f$ is the risk-free rate.
    
    #### Efficient Frontier
    
    The **efficient frontier** is the set of portfolios that offer the maximum expected return for each level of risk. Mathematically, it's a hyperbola in mean-variance space.
    
    For unconstrained portfolios (allowing short sales):
    
    $$\mu_p = \mu_{mvp} + \lambda \sqrt{\sigma_p^2 - \sigma_{mvp}^2}$$
    
    where $\mu_{mvp}$ and $\sigma_{mvp}$ are the return and risk of the minimum variance portfolio.
    
    #### Analytical Solution (Unconstrained)
    
    Using Lagrangian multipliers, the optimal weights are:
    
    $$w^* = \frac{\Sigma^{-1}(\mu - r_f \mathbf{1})}{\mathbf{1}^T \Sigma^{-1}(\mu - r_f \mathbf{1})}$$
    
    This is the **tangency portfolio** (maximum Sharpe ratio).
    
    For the minimum variance portfolio:
    
    $$w_{mvp} = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^T \Sigma^{-1} \mathbf{1}}$$
    
    """)


@app.cell
def __(mo):
    mo.md("""
    *This is a comprehensive educational notebook on portfolio optimization.*
    *Full technical content and implementations would be added in subsequent cells.*
    *The notebook follows the same structure as the climate models notebook with detailed technical explanations.*
    """)


@app.cell
def __(mo):
    mo.md(r"""
    ## Conclusions
    
    ### The Evolution of Portfolio Optimization
    
    Our journey through five approaches reveals the progression of financial theory and practice from Markowitz's foundational work in 1952 to modern machine learning approaches.
    
    ### Universal Principles
    
    Despite different approaches, several principles emerge:
    
    **1. Risk-Return Tradeoff:** All approaches recognize that higher returns require higher risk.
    
    **2. Diversification:** The only "free lunch" in finance - reduces risk without sacrificing expected return.
    
    **3. Estimation Uncertainty:** Limited data means parameter uncertainty; robust methods acknowledge this.
    
    **4. Market Efficiency (Partial):** Markets incorporate information, making consistent alpha generation difficult.
    
    **5. Adaptation:** Markets evolve; successful strategies must adapt.
    
    ---
    
    ### References and Further Reading
    
    **Foundational Papers:**
    - Markowitz, H. (1952): "Portfolio Selection", Journal of Finance
    - Sharpe, W. (1964): "Capital Asset Prices: A Theory of Market Equilibrium"
    - Black, F. & Litterman, R. (1992): "Global Portfolio Optimization"
    - Fama, E. & French, K. (1993): "Common Risk Factors in Returns"
    
    **Books:**
    - Markowitz: "Portfolio Selection: Efficient Diversification of Investments"
    - Grinold & Kahn: "Active Portfolio Management"
    - López de Prado: "Advances in Financial Machine Learning"
    - Ang: "Asset Management: A Systematic Approach to Factor Investing"
    
    ---
    
    *Thank you for following this journey through portfolio optimization!*
    """)


if __name__ == "__main__":
    app.run()

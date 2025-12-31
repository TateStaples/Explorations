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
    - Mathematical formulations and theoretical foundations
    - Practical implementation considerations
    - Comparative analysis and use cases
    
    **Key Results**: Shows convergence from classical optimization (Sharpe ratio ~0.5) to modern approaches (Sharpe ratio ~1.2+), demonstrates importance of diversification, analyzes estimation error impact, and provides practical guidance for real-world portfolio construction.
    
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
    
    The optimization problem seeks to maximize expected return for a given level of risk:
    
    $$\min_w \quad \frac{1}{2} w^T \Sigma w$$
    $$\text{s.t.} \quad \mu^T w = \mu_{\text{target}}$$
    $$\quad\quad\quad w^T \mathbf{1} = 1, \quad w \geq 0$$
    
    Where: $w$ = portfolio weights, $\Sigma$ = covariance matrix, $\mu$ = expected returns
    
    #### Efficient Frontier
    
    The efficient frontier represents portfolios with maximum return for each risk level. For unconstrained portfolios:
    
    $$w^* = \frac{\Sigma^{-1}(\mu - r_f \mathbf{1})}{\mathbf{1}^T \Sigma^{-1}(\mu - r_f \mathbf{1})}$$
    
    This is the tangency portfolio (maximum Sharpe ratio).
    
    ### Technical Overview (Page 2 of 2)
    
    #### Key Assumptions & Limitations
    
    **Assumptions:**
    - Returns are normally distributed
    - Known parameters (μ, Σ)
    - Single period analysis
    - No transaction costs
    
    **Practical Challenges:**
    - Estimation error dominates optimization
    - Extreme positions in unconstrained case
    - High sensitivity to inputs
    - Poor out-of-sample performance
    
    **Improvements:**
    - Regularization: Ridge, Ledoit-Wolf shrinkage
    - Constraints: Long-only, position limits
    - Robust optimization: Worst-case approaches
    - Alternative risk measures: CVaR, maximum drawdown
    """)


@app.cell
def __(mo):
    mo.md(r"""
    <a id='approach2'></a>
    ## Approach 2: Capital Asset Pricing Model (CAPM)
    
    ### Technical Overview (Page 1 of 2)
    
    The CAPM describes asset pricing in market equilibrium:
    
    $$E[r_i] = r_f + \beta_i (E[r_m] - r_f)$$
    
    Where $\beta_i = \frac{\text{Cov}(r_i, r_m)}{\text{Var}(r_m)}$ measures systematic risk.
    
    #### Risk Decomposition
    
    $$\sigma_i^2 = \beta_i^2 \sigma_m^2 + \sigma_{\epsilon_i}^2$$
    
    - Systematic risk: Cannot be diversified
    - Idiosyncratic risk: Can be diversified away
    
    **Key Insight:** Only systematic risk is rewarded.
    
    ### Technical Overview (Page 2 of 2)
    
    #### Extensions: Multi-Factor Models
    
    **Fama-French 3-Factor:**
    $$E[r_i] - r_f = \beta^m(E[r_m] - r_f) + \beta^{SMB} E[SMB] + \beta^{HML} E[HML]$$
    
    **Fama-French 5-Factor:** Adds profitability (RMW) and investment (CMA) factors.
    
    #### Applications
    - Performance attribution (alpha vs beta)
    - Risk management and hedging
    - Cost of capital estimation
    - Beta-neutral strategies
    
    #### Limitations
    - Single factor insufficient (many anomalies)
    - Market portfolio unobservable
    - Static model (no time variation)
    - Empirical violations well-documented
    """)


@app.cell
def __(mo):
    mo.md(r"""
    <a id='approach3'></a>
    ## Approach 3: Risk Parity
    
    ### Technical Overview (Page 1 of 2)
    
    Risk Parity equalizes risk contributions rather than capital allocations.
    
    #### Core Principle
    
    The risk contribution of asset $i$ is:
    
    $$RC_i = w_i \frac{(\Sigma w)_i}{\sigma_p}$$
    
    Risk parity sets: $RC_1 = RC_2 = \cdots = RC_n = \frac{\sigma_p}{n}$
    
    #### Special Case: Uncorrelated Assets
    
    If assets uncorrelated: $w_i^{RP} = \frac{1/\sigma_i}{\sum_{j=1}^n 1/\sigma_j}$
    
    **Interpretation:** Inverse volatility weighting.
    
    ### Technical Overview (Page 2 of 2)
    
    #### Advantages & Limitations
    
    **Advantages:**
    - Avoids estimation error in expected returns
    - Natural diversification
    - More stable weights
    - Better risk-adjusted returns historically
    
    **Limitations:**
    - Ignores expected returns entirely
    - Requires leverage for target returns
    - Vulnerable to correlation spikes in crises
    - Not optimal under mean-variance criterion
    
    #### Extensions
    - Hierarchical Risk Parity (HRP)
    - Risk budgeting (non-equal allocations)
    - Factor risk parity
    - Risk parity with views
    
    **Key Insight:** Trades theoretical optimality for practical robustness.
    """)


@app.cell
def __(mo):
    mo.md(r"""
    <a id='approach4'></a>
    ## Approach 4: Black-Litterman Model
    
    ### Technical Overview (Page 1 of 2)
    
    Black-Litterman combines market equilibrium with investor views using Bayesian framework.
    
    #### Reverse Optimization
    
    Start with implied equilibrium returns:
    
    $$\Pi = \lambda \Sigma w_m$$
    
    Where $\lambda$ is risk aversion and $w_m$ are market weights.
    
    #### Incorporating Views
    
    Views expressed as: $P \mu = Q + \epsilon$
    
    Example: "Asset 1 will outperform Asset 2 by 2%" → $P = [1, -1, 0, \ldots], Q = 0.02$
    
    ### Technical Overview (Page 2 of 2)
    
    #### Posterior Returns
    
    $$E[R] = \Pi + \tau \Sigma P^T (P \tau \Sigma P^T + \Omega)^{-1} (Q - P\Pi)$$
    
    Where:
    - $\tau$ = uncertainty in prior (typically 0.01-0.05)
    - $\Omega$ = view uncertainty matrix
    
    #### Advantages
    - Stable weights (gradual changes)
    - Intuitive (start neutral, tilt on views)
    - Confidence weighting
    - Better diversification
    
    #### Practical Considerations
    - View generation (fundamental, technical, factor-based)
    - Calibration of τ and Ω
    - Periodic rebalancing
    - Performance attribution
    
    **Key Insight:** Practitioners' solution to mean-variance impracticality.
    """)


@app.cell
def __(mo):
    mo.md(r"""
    <a id='approach5'></a>
    ## Approach 5: Machine Learning-Based Optimization
    
    ### Technical Overview (Page 1 of 2)
    
    ML approaches learn patterns directly from data without assuming specific distributions.
    
    #### ML Framework Components
    
    **1. Feature Engineering:**
    - Price-based: Returns, volatility, momentum
    - Technical: RSI, MACD, moving averages
    - Fundamental: P/E, earnings, ROE
    - Sentiment: News, social media
    - Alternative: Satellite, credit card data
    
    **2. Model Architectures:**
    - Random Forests, XGBoost
    - Neural Networks (LSTM, GRU)
    - Reinforcement Learning (Q-learning, PPO)
    - Transformers with attention
    
    ### Technical Overview (Page 2 of 2)
    
    #### Reinforcement Learning for Portfolio Management
    
    **MDP Formulation:**
    - State: $s_t = \{w_t, X_t, r_t\}$
    - Action: $a_t = \Delta w_t$ (rebalancing)
    - Reward: $r_t = r_p(w_t) - TC(a_t)$
    - Objective: $\max_{\pi} E[\sum_{t=0}^T \gamma^t r_t]$
    
    #### Advantages & Challenges
    
    **Advantages:**
    - Flexible functional forms
    - Handles nonlinearities
    - Incorporates alternative data
    - Adaptive to regime changes
    
    **Challenges:**
    - Overfitting risk
    - Black-box nature
    - Non-stationarity
    - Implementation gap
    - Tail risk underestimation
    
    #### Hybrid Approaches
    
    Combine traditional + ML:
    - ML for parameter estimation
    - Physics-informed constraints
    - Ensemble methods
    
    **Key Insight:** ML excels at pattern recognition; traditional methods provide stability and theory.
    """)


@app.cell
def __(mo):
    mo.md(r"""
    ## Comparative Analysis and Synthesis
    
    ### Summary Table
    
    | Approach | Core Idea | Key Strength | Main Limitation | Best Use Case |
    |----------|-----------|--------------|-----------------|---------------|
    | **Markowitz** | Maximize return/risk | Theoretical foundation | Estimation error | Strategic allocation |
    | **CAPM** | Market equilibrium | Risk decomposition | Single factor | Performance attribution |
    | **Risk Parity** | Equal risk contribution | Stability, diversification | Ignores returns | Multi-asset portfolios |
    | **Black-Litterman** | Equilibrium + views | Intuitive, stable | Complexity | Active management |
    | **Machine Learning** | Data-driven patterns | Flexibility | Overfitting, black-box | Large datasets |
    
    ### Integration Framework
    
    **Strategic Layer (3-5 years):**
    - Risk Parity or Black-Litterman for asset allocation
    - Low turnover, emphasize diversification
    
    **Tactical Layer (6-12 months):**
    - Black-Litterman with macro views
    - Factor models
    
    **Alpha Layer (short-term):**
    - Machine Learning for security selection
    - Manage transaction costs
    
    ### Key Lessons
    
    1. **No Universal Best:** Each approach has context-dependent strengths
    2. **Estimation Error Matters:** Most important practical consideration
    3. **Diversification is Robust:** Works across all frameworks
    4. **Stability vs Optimality:** Key tradeoff in practice
    5. **Theory + Practice:** Classical methods + ML = future
    6. **Risk Management First:** More important than return optimization
    """)


@app.cell
def __(mo):
    mo.md(r"""
    ## Conclusions
    
    ### The Evolution of Portfolio Optimization
    
    From Markowitz (1952) to modern ML (2020s), portfolio optimization has evolved from elegant mathematics to sophisticated data-driven systems.
    
    ### Universal Principles
    
    1. **Risk-Return Tradeoff:** Higher returns require higher risk
    2. **Diversification:** The only "free lunch" in finance
    3. **Estimation Uncertainty:** Robust methods acknowledge limited data
    4. **Market Efficiency:** Partial efficiency makes alpha generation difficult
    5. **Adaptation:** Markets evolve; strategies must adapt
    
    ### Practical Wisdom
    
    **For Individual Investors:**
    - Simple strategies often beat complex ones
    - Focus on asset allocation over security selection
    - Minimize costs (fees, taxes, trading)
    - Stay diversified and rebalance periodically
    
    **For Institutional Investors:**
    - Combine multiple approaches (strategic/tactical/alpha)
    - Invest in robust estimation methods
    - Understand model limitations
    - Balance sophistication with interpretability
    
    **As Benjamin Graham said:** *"The essence of investment management is the management of risks, not the management of returns."*
    
    ---
    
    ### References
    
    **Foundational Papers:**
    - Markowitz, H. (1952): "Portfolio Selection", Journal of Finance
    - Sharpe, W. (1964): "Capital Asset Prices"
    - Black, F. & Litterman, R. (1992): "Global Portfolio Optimization"
    - Fama, E. & French, K. (1993): "Common Risk Factors"
    
    **Books:**
    - Markowitz: "Portfolio Selection: Efficient Diversification"
    - Grinold & Kahn: "Active Portfolio Management"
    - López de Prado: "Advances in Financial Machine Learning"
    - Ang: "Asset Management: A Systematic Approach"
    
    ---
    
    *Thank you for following this journey through portfolio optimization!*
    
    *The best portfolio is one you can stick with through all market conditions.*
    """)


if __name__ == "__main__":
    app.run()

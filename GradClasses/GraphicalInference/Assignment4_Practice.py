# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.19.0",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Homework 4
    Due Friday February 27, 2026

    Reading:
    * Required: Lecture 6 Notes
    * Optional: PGMCT Ch. 6 and GMEFVI Ch. 6
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. (Learning with a single variable and factor).
    Let $X \in [m] := \{1, 2, \dots , m\}$ and consider the one-factor model

    $$
    p_\theta (x) = \frac{\exp(\theta_x)}{Z(\theta)}, \quad Z(\theta) = \sum_{u=1}^m \exp(\theta_u)
    $$

    where $\theta = (\theta_1 , \dots , \theta_m ) \in \mathbb{R}^m$. Given i.i.d. data $\mathcal{D} = \{x^{(1)} , \dots , x^{(N)} \}$, define counts

    $$
    N_x := \sum_{\ell=1}^N 1\{x^{(\ell)} = x\}, \quad \hat{p}_{\mathcal{D}} (x) := \frac{N_x}{N}.
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(a)** Write the normalized log-likelihood $\frac{1}{N} \mathcal{L}(\theta)$ explicitly in terms of $\{N_x\}$ and $\theta$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(b)** Compute $\frac{\partial}{\partial \theta_x} \frac{1}{N} \mathcal{L}(\theta)$ and simplify it to the difference between data and model terms.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(c)** Show that $\frac{1}{N} \mathcal{L}(\theta)$ has a maximizer that is not unique due to an extra degree of freedom.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(d)** Impose the constraint $\sum_{x=1}^m \theta_x = 0$. Under this constraint, solve for the MLE $\hat{\theta}$ in closed form in terms of $\hat{p}_{\mathcal{D}}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. (Identifiability and gauge).
    Consider a factor graph with indicator/table factors

    $$
    f_a(x_{\partial a}; \theta_a) = \exp(\theta_{a, x_{\partial a}}), \quad
    \theta_a \in \mathbb{R}^{|\mathcal{X}_{\partial a}|}, \quad \theta_{a, x_{\partial a}} = [\theta_a]_{x_{\partial a}}
    $$

    Assume the graph is connected and all variables are discrete.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(a)** Show that if you add a constant $c_a$ to each entry of $\theta_a$ (possibly different across factors), the distribution $p_\theta(x)$ does not change. Thus, the constants $c_a$ cannot be identified from data.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(b)** Show that there is at least one additional redundancy: a "reparameterization" that preserves $p_\theta$ in a pairwise MRF. Specifically, consider a pairwise MRF

    $$
    p_\theta(x) \propto \exp \left( \sum_{i \in V} \theta_i(x_i) + \sum_{(i,j) \in E} \theta_{ij}(x_i, x_j) \right)
    $$

    Show that we can use arbitrary functions $\phi_i: \mathcal{X}_i \to \mathbb{R}$ to define new parameters $\theta'_i$, $\theta'_{ij}$ (depending on $\phi_i$) so that

    $$
    \sum_i \theta'_i(x_i) + \sum_{(i,j)} \theta'_{ij}(x_i, x_j) = \sum_i \theta_i(x_i) + \sum_{(i,j)} \theta_{ij}(x_i, x_j) \quad \text{for all } x.
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(c)** Propose a set of constraints that removes these redundancies (not necessarily uniquely), and briefly argue why it fixes identifiability.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. (EM soft clustering).
    Let $Y \in \{1, \dots, m\}$ be observed and $H \in \{1, \dots, K\}$ hidden. Consider

    $$
    p_\theta(y, h) \propto \exp(\theta_{yh} + \alpha_y + \beta_h)
    $$

    One can interpret $H$ as a latent class (or cluster) index and $q^{(\ell)}(h)$ as a soft cluster assignment for datum $y^{(\ell)}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(a)** Show that $p_\theta(h | y)$ is a softmax in the scores $\theta_{y*} + \beta$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(b)** For data $\{y^{(\ell)}\}_{\ell=1}^N$, write the E-step responsibilities and explain their clustering meaning.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(c)** Derive the M-step objective and first-order condition in terms of soft counts.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(d)** Describe explicit constraints for $(\theta, \alpha, \beta)$ that remove symmetries to make the parameters more identifiable.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. (Small-step viewpoint: EM direction as likelihood gradient.)
    Let $L_{\text{obs}}(\theta) = \sum_{\ell=1}^N \ln p^\theta(y^{(\ell)})$ be the observed-data log-likelihood in a latent-variable model with latent $H$ and complete-data log-density $\ln p^\theta(y, h)$.

    Fix a current parameter $\theta$ and define the Q-function

    $$
    Q(\theta' \mid \theta) := \sum_{\ell=1}^N \mathbb{E}_{p^\theta(H \mid y^{(\ell)})} \left[ \ln p^{\theta'}(y^{(\ell)}, H) \right].
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(a)** Show that $\nabla_\theta L_{\text{obs}}(\theta) = \nabla_{\theta'} Q(\theta' \mid \theta)\big|_{\theta'=\theta}$.

    (Assume you can interchange derivative and summation/expectation.)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(b)** Conclude that the update $\theta_{\text{new}} = \theta + \eta\,\nabla_{\theta'} Q(\theta' \mid \theta)\big|_{\theta'=\theta}$ with $\eta > 0$ small is a first-order ascent step for $L_{\text{obs}}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(c)** (Interpretation) Explain in one paragraph why this supports the intuition that "the E-step is exact in the small-change limit."
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. (A Linear-Programming Solver for Sudoku.)
    Combinatorial problems that optimize over permutations can be quite challenging computationally. In this problem, we explore low-complexity linear-programming (LP) relaxations.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(a)** Assume $n$ tasks must be assigned to $n$ agents and let $x_{i,j}$ denote the indicator variable for "task $i$ assigned to agent $j$". If the cost of assigning task $i$ to agent $j$ is $c_{i,j}$, then the best assignment problem is equivalent to the integer linear program

    $$
    \min_{\{x_{i,j}\}} \sum_{i,j} c_{i,j} x_{i,j} \quad \text{subject to} \quad x_{i,j} \in \{0,1\}, \quad \sum_i x_{i,j} = 1, \quad \sum_j x_{i,j} = 1.
    $$

    Solve this problem by hand for $n = 4$ and

    $$
    C = \begin{pmatrix} 10 & 7 & 24 & 6 \\ 19 & 8 & 7 & 8 \\ 16 & 22 & 12 & 15 \\ 4 & 8 & 10 & 14 \end{pmatrix},
    $$

    Now, solve the linear program where the constraint $x_{i,j} \in \{0,1\}$ is relaxed to $x_{i,j} \in [0,1]$. This minimizes average cost when $x_{i,j}$ is the probability of assigning task $i$ to agent $j$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(np):
    from itertools import permutations as _permutations
    from scipy.optimize import linprog as _linprog

    _C = np.array([[10, 7, 24, 6], [19, 8, 7, 8], [16, 22, 12, 15], [4, 8, 10, 14]])
    _n = 4

    # TODO: Brute-force integer program over all permutations

    # TODO: LP relaxation using linprog
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(b)** Using a similar approach, an $N \times N$ Sudoku game (where $N$ is a perfect square) can be cast as an integer linear program with $N^3$ variables. Let $x_{i,j,k}$ denote the indicator variable for "the square in row $i$ and column $j$ is assigned the value $k$". Based on the previous part, write the permutation constraints as linear equalities for $N = 4$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(c)** The Sudoku puzzle is solved by computing

    $$
    \min_{\{x_{i,j,k}\}} \sum_{i,j,k} c_{i,j,k} x_{i,j,k},
    $$

    where $c_{i,j,k}$ is the cost of assigning the integer $k$ to the $(i,j)$-th entry of the puzzle. Known values in the puzzle can be handled either by **adding equality constraints** or by **adjusting the cost function**. For example, "the square in row $i$ and column $j$ is observed to be $k$" implies the equality constraint $x_{i,j,k} = 1$. Known values can also be included by initializing $c_{i,j,k}$ to all zeros and then setting $c_{i,j,k} = -1$ for each $(i,j,k)$ observation. Choose one of these approaches and use linear programming to solve the $N = 4$ Sudoku:

    $$
    \begin{pmatrix} \cdot & 2 & \cdot & \cdot \\ 1 & \cdot & \cdot & \cdot \\ \cdot & \cdot & 2 & \cdot \\ \cdot & \cdot & 3 & \cdot \end{pmatrix}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


@app.cell
def _():
    from scipy.optimize import linprog as _linprog2

    _N = 4
    _sqN = 2
    _nv = _N**3

    def _idx(_i, _j, _k):
        return _i * _N**2 + _j * _N + _k

    # TODO: Set up cost vector c with -1 for known cells
    # Known values (0-indexed): (row, col) -> value
    _known = {(0, 1): 2, (1, 0): 1, (2, 2): 2, (3, 2): 3}

    # TODO: Build equality constraint matrices for row, column, cell, and box constraints

    # TODO: Call linprog and display the solved grid
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(d)** Let $\mathcal{X} = \{1, 2, \ldots, N\}$ and $f: \mathcal{X}^N \to \{0,1\}$ be a factor defined by $f(z_1^N) = 1$ if $z_1^N$ defines a permutation of $\mathcal{X}$ and $0$ otherwise. From this, one can construct the Sudoku factor graph by introducing $N^2$ variables and $3N^2$ permutation factor nodes. Local 0/1 factors can be used to account for known values. In this case, marginalization gives

    $$
    g_i(x_i) = \sum_{\bar{x} \setminus x_i} f(\bar{x}) = \#\{\text{valid patterns matching all observations with fixed } x_i\}.
    $$

    Unfortunately, naïve implementation of the sum-product algorithm for the permutation factor requires the summation of $N!$ terms. Describe how one might implement such a factor-graph permutation constraint with much lower complexity than $N!$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Solution:**
    """)
    return


if __name__ == "__main__":
    app.run()

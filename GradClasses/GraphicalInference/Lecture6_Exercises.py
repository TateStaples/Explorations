# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.19.0",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
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

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Lecture 6 Exercises — Learning Factor Graph Parameters
    ECE 590.17: Graphical Models and Inference for Machine Learning
    Duke University, Spring 2026

    Reference: Section 6.8 of Lecture 6 Notes
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise 1 (EM soft statistics).
    Derive the expression for $\widehat{T}_{a,\text{soft}}$ in the E-step and explain why it can be computed from the factor marginals $p^{\theta^{\text{old}}}_{X_{\partial a}|Y}(x_{\partial a} \mid y^{(\ell)})$ returned by BP on a tree.
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
    ## Exercise 2 (Gauge freedom for table factors).
    Let $\mathcal{X}_{\partial a}$ be finite and define $f_a(u) = \exp(\theta_{a,u})$. Show that adding a constant $c$ to all $\theta_{a,u}$ multiplies $f_a$ by $e^c$ and does not change $p^\theta_X$ after renormalization. Give one gauge constraint that removes this redundancy.
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
    ## Exercise 3 (MLE for a single CPT via direct optimization).
    Let $X \in \{1, \ldots, r\}$ and $U \in \{1, \ldots, s\}$, and suppose the conditional distribution is a CPT $p_\theta(x \mid u) = \theta_{x|u}$ with constraints $\sum_{x=1}^r \theta_{x|u} = 1$ for each $u$. Given an i.i.d. dataset $\mathcal{D} = \{(x^{(\ell)}, u^{(\ell)})\}_{\ell=1}^N$, define counts $N(x, u) = \sum_{\ell=1}^N \mathbb{1}\{x^{(\ell)} = x,\, u^{(\ell)} = u\}$ and $N(u) = \sum_x N(x, u)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **1.** Write the (normalized) log-likelihood $\frac{1}{N}\mathcal{L}(\theta)$ in terms of $N(x, u)$ and $\theta_{x|u}$.
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
    **2.** Using Lagrange multipliers (one constraint per $u$), derive the MLE for $\theta_{x|u}$.
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
    **3.** Explain what happens if $N(u) = 0$ and give a simple practical fix.
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
    ## Exercise 4 (Convexity/concavity of the undirected log-likelihood).
    Consider the exponential-family factor graph model

    $$
    p^\theta(x) = \frac{1}{Z(\theta)} \exp\!\left(\sum_{a \in F} \theta_a^\top T_a(x_{\partial a})\right), \quad \theta = (\theta_a)_{a \in F}.
    $$

    Let $\frac{1}{N}\mathcal{L}(\theta)$ be the normalized log-likelihood for fully observed data.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **1.** Show that $\nabla_\theta \ln Z(\theta) = \mathbb{E}_{p^\theta}[T(X)]$, where $T(X)$ is the concatenation of all $T_a(X_{\partial a})$.
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
    **2.** Show that the Hessian satisfies $\nabla^2_\theta \ln Z(\theta) = \text{Cov}_{p^\theta}(T(X))$.
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
    **3.** Conclude that $\ln Z(\theta)$ is convex and that $\ell(\theta)$ is concave.
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
    ## Exercise 5 (One-step contrastive divergence update (CD-1) for a log-linear model).
    Consider a discrete energy-based model

    $$
    p^\theta(x) = \frac{1}{Z(\theta)} \exp(\theta^\top T(x)).
    $$

    Let $\widehat{\mathbb{E}}_\mathcal{D}[T(X)]$ be the empirical feature average. In CD-1, one approximates the model expectation $\mathbb{E}_{p^\theta}[T(X)]$ by sampling a data point $X^0 \sim \widehat{p}^\mathcal{D}_X$, performing a Gibbs step to get $X^1$, and using $T(X^1)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **1.** Write the exact gradient of the normalized log-likelihood $\frac{1}{N}\mathcal{L}(\theta)$.
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
    **2.** Write the CD-1 approximate gradient $\widehat{g}_{\text{CD1}}(\theta)$ in terms of $T(X^0)$ and $T(X^1)$.
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
    **3.** Give the resulting CD-1 update rule with step size $\eta$.
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
    ## Exercise 6 (EM monotonicity via a KL decomposition).
    Recall the EM functional

    $$
    \mathcal{F}(\theta, \{q^{(\ell)}\}) := \sum_{\ell=1}^N \mathbb{E}_{q^{(\ell)}}\!\left[\ln p^\theta(y^{(\ell)}, H)\right] + \sum_{\ell=1}^N H(q^{(\ell)}).
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **1.** Show that for each $\ell$,

    $$
    \ln p^\theta(y^{(\ell)}) = \mathcal{F}_\ell(\theta, q^{(\ell)}) + D(q^{(\ell)}(\cdot) \| p^\theta(H \mid y^{(\ell)})),
    $$

    where $\mathcal{F}_\ell$ is the $\ell$th summand of $\mathcal{F}$.
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
    **2.** Use this identity to argue that choosing $q^{(\ell)} = p^{\theta^{\text{old}}}(H \mid y^{(\ell)})$ maximizes $\mathcal{F}$ over $q$ for fixed $\theta^{\text{old}}$.
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
    **3.** Conclude (informally) why the E-step followed by an exact M-step does not decrease the observed-data log-likelihood.
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
    ## Exercise 7 (HMM forward–backward messages and $\gamma$, $\xi$).
    Consider an HMM with hidden states $S_t \in \{1, \ldots, K\}$ and observations $Y_t \in \mathcal{Y}$ with parameters: initial $\pi(i) = p(S_1 = i)$, transition $A_{ij} = p(S_t = j \mid S_{t-1} = i)$, and emission $B_j(y) = p(Y_t = y \mid S_t = j)$. Given a single observation sequence $y_{1:T}$:
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **1.** Define forward messages $\alpha_t(j)$ and give their recursion (up to normalization).
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
    **2.** Define backward messages $\beta_t(i)$ and give their recursion (up to normalization).
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
    **3.** Express $\gamma_t(j) = p(S_t = j \mid y_{1:T})$ in terms of $\alpha_t$, $\beta_t$.
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
    **4.** Express $\xi_t(i, j) = p(S_{t-1} = i, S_t = j \mid y_{1:T})$ in terms of $\alpha_{t-1}$, $A$, $B$, $\beta_t$.
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
    ## Exercise 8 (EM M-step for an HMM with multiple sequences).
    Suppose we observe $N$ independent sequences $\{y^{(\ell)}_{1:T_\ell}\}_{\ell=1}^N$ from the same time-homogeneous HMM parameters $(\pi, A, B)$. Let $\gamma^{(\ell)}_t(j)$ and $\xi^{(\ell)}_t(i, j)$ be the posteriors computed in the E-step for sequence $\ell$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **1.** Derive the M-step update for the initial distribution $\pi(j)$.
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
    **2.** Derive the M-step update for the transition probabilities $A_{ij}$.
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
    **3.** Assume emissions are categorical over $\mathcal{Y}$ (finite alphabet). Derive the M-step update for $B_j(y)$.
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

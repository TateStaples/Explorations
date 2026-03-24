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
    # Homework 3
    Due Wednesday February 18, 2026

    Reading:
    * Required: Lecture 4-5 Notes
    * Optional: Ch. 2 of IPC
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. (IPC: 2.01) Exercise 2.1
    As a particular realization of the above example, consider an $8 \times 8$ chessboard and a special piece sitting on it. At any time step, the piece will stay still (with probability $1/2$) or move randomly to one of the neighboring positions (with probability $1/2$). Does this process satisfy the detailed balance condition? Which positions on the chess board have lower and higher "energy"? Compute the partition function.
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
    ## 2. (IPC: 2.02) Exercise 2.2
    A two-level system. This is the simplest non-trivial example: $\mathcal{X} = \{1, 2\}$, $E(1) = \epsilon_1$, $E(2) = \epsilon_2$. Without loss of generality, we assume $\epsilon_1 < \epsilon_2$. This example can be used as a mathematical model for many physical systems, such as the spin-1/2 particle discussed above.

    Derive the following for the thermodynamic potentials (where $\Delta = \epsilon_2 - \epsilon_1$ is the energy gap):

    $$
    F(\beta) = \epsilon_1 - \frac{1}{\beta} \log\left(1 + e^{-\beta\Delta}\right)
    $$

    $$
    U(\beta) = \epsilon_1 + \frac{e^{-\beta\Delta}}{1 + e^{-\beta\Delta}} \Delta
    $$

    $$
    S(\beta) = \frac{e^{-\beta\Delta}}{1 + e^{-\beta\Delta}} \beta\Delta + \log\left(1 + e^{-\beta\Delta}\right)
    $$

    Additionally, work out the asymptotics and verify the general high- and low-temperature behavior.
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
    ## 3. (FGML: 4.01) Ising Effective field and single-spin conditional.
    Consider an Ising model on graph $G$ with energy

    $$
    E(\sigma) = -\sum_{(i,j) \in E} J_{ij} \sigma_i \sigma_j - \sum_i h_i^{\text{ext}} \sigma_i, \quad \sigma \in \{\pm 1\}^n
    $$

    Define $h_i^{\text{eff}} = h_i^{\text{ext}} + \sum_{j \in \partial i} J_{ij} \sigma_j$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(a)** Derive the conditional distribution $P(\sigma_i = +1 \mid \sigma_{-i})$ in logistic form.
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
    **(b)** Show that $\mathbb{E}[\sigma_i \mid \sigma_{\neq i}] = \tanh(\beta h_i^{\text{eff}})$.
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
    ## 4. (FGML: 4.02) Mean-field self-consistency.
    The following fully-connected Ising model on $n$ spins, known as the Curie–Weiss model, is defined by

    $$
    E(\sigma) = -\frac{J}{n} \sum_{1 \leq i < j \leq n} \sigma_i \sigma_j - h \sum_{i=1}^n \sigma_i, \quad \sigma_i \in \{\pm 1\}, \quad J > 0
    $$

    Let $m = \frac{1}{n} \sum_i \sigma_i$ be the magnetization.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(a)** In this model, one assumes that the magnetization concentrates around its average $m \approx \frac{1}{n} \sum_i \mathbb{E}[\sigma_i]$. Then, this is used to compute new means $\mathbb{E}[\sigma_i \mid \sigma_{-i}]$ based on the other spins. Derive the self-consistency equation for $m$ implied by equilibrium.
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
    **(b)** For $h = 0$, determine the range of $\beta$ for which the $m = 0$ fixed point is stable.
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
    ## 5. (IPC: 2.06) Exercise 2.6
    Consider the one-dimensional Ising model in zero field, $B = 0$. Show that when $\delta N < i < j < (1-\delta)N$, the correlation function $\langle \sigma_i \sigma_j \rangle$ is, in the large-$N$ limit,

    $$
    \langle \sigma_i \sigma_j \rangle = e^{-|i-j|/\xi(\beta)}, \quad \text{where} \quad \xi(\beta) = \frac{-1}{\log(\tanh(\beta))}
    $$

    [Hint: You can either use the general transfer matrix formalism or, more simply, use the identity $e^{\beta \sigma_i \sigma_{i+1}} = \cosh[\beta(1 + \sigma_i \sigma_{i+1} \tanh(\beta))]$.]
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

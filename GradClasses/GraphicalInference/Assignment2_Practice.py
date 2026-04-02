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
    # Homework 2
    Due Monday February 9, 2026

    Reading:
    * Required: Course Notes
    * Suggested: Ch 2-3 [EIT]
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. (EIT: 2.04) Entropy of functions of a random variable.
    Let $X$ be a discrete random variable. Show that the entropy of a function of $X$ is less than or equal to the entropy of $X$ by justifying the following steps:

    $$
    \begin{align*}
    H(X, g(X)) &\overset{(a)}{=} H(X) + H(g(X) \mid X) \\
               &\overset{(b)}{=} H(X); \\
    H(X, g(X)) &\overset{(c)}{=} H(g(X)) + H(X \mid g(X)) \\
               &\overset{(d)}{\geq} H(g(X)).
    \end{align*}
    $$

    Thus $H(g(X)) \leq H(X)$.
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
    ## 2. (EIT: 2.12) Example of joint entropy.
    Let $p(x, y)$ be given by

    | $X \backslash Y$ | 0 | 1 |
    |:-:|:-:|:-:|
    | 0 | 1/3 | 1/3 |
    | 1 | 0 | 1/3 |

    Find:
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(a)** $H(X)$, $H(Y)$.
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
    **(b)** $H(X \mid Y)$, $H(Y \mid X)$.
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
    **(c)** $H(X, Y)$.
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
    **(d)** $H(Y) - H(Y \mid X)$.
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
    **(e)** $I(X; Y)$.
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
    **(f)** Draw a Venn diagram for the quantities in parts (a) through (e).
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
    ## 3. (GRIT: 1.06) Entropy of time to first success.
    A fair coin is flipped repeatedly. Let $X_i$ denote the number of flips required for the $i$th occurrence of a head. (i.e., if the coin is a head on the first flip then $X_1 = 1$). For example, the flip sequence "TTT HT HT T H . . ." generates the outcome $(X_1, X_2, X_3, \ldots) = (4, 2, 3, \ldots)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(a)** Find the entropy $H(X_1)$ in bits.
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
    **(b)** Find the conditional entropy $H(X_2 \mid X_1)$ in bits.
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
    **(c)** Show that $H(X_1) \leq H(X_2) \leq 2H(X_1)$. [Hint: try expressing $H(X_1, X_2)$ two different ways.]
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
    ## 4. (GRIT: 2.01) Cross-Entropy Loss.
    Consider a problem where we are given $n$ iid samples $(x_i, y_i) \in \mathcal{X} \times \mathcal{Y}$ drawn from $p(x, y)$ and the goal is to learn a predictor $f$ that maps each $x \in \mathcal{X}$ to a prediction of $y \in \mathcal{Y}$. Without loss of generality, we will assume the prediction is an estimate of the posterior distribution of $y$ given $x$. A common approach in machine learning is to define a loss function and then use the predictor that minimizes the average loss.

    Let $\mathcal{P}(\mathcal{Y})$ denote the set of pmfs over $\mathcal{Y}$. Then, the cross-entropy loss $\ell : \mathcal{Y} \times \mathcal{P}(\mathcal{Y}) \to \mathbb{R}_{\geq 0}$ maps the true $y \in \mathcal{Y}$ and the predicted posterior $q(y)$ to the non-negative real loss

    $$
    \ell(y, q) = \log\frac{1}{q(y)}.
    $$

    For each $\theta \in \mathbb{R}^d$, let $f_\theta(x) : \mathcal{X} \to \mathcal{P}(\mathcal{Y})$ be a candidate predictor that maps each $x \in \mathcal{X}$ to a pmf over $\mathcal{Y}$. Then, the average cross-entropy loss is given by

    $$
    L_n(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(y_i, f_\theta(x_i)).
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(a)** To what non-random value will $L_n(\theta)$ converge to as $n \to \infty$?
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
    **(b)** For fixed $\theta$, write the quantity in (a) as $H(Y \mid X)$ plus the average (over $\mathcal{X}$) divergence between $p(y \mid x)$ and $q(y \mid x) = f_\theta(x)(y)$.
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
    **(c)** Use the non-negativity of divergence to identify the minimum possible cross-entropy loss and choice of $f_\theta$ that achieves it.
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
    ## 5. (EIT: 3.04) AEP.
    Let $X_i$ be iid $\sim p(x)$, $x \in \{1, 2, \ldots, m\}$. Let $\mu = \mathbb{E}[X]$ and $H = H(X)$. Let

    $$
    A^n = \left\{ x^n \in \mathcal{X}^n : \left| -\frac{1}{n} \log p(x^n) - H \right| \leq \epsilon \right\}
    $$

    $$
    B^n = \left\{ x^n \in \mathcal{X}^n : \left| \frac{1}{n} \sum_{i=1}^n X_i - \mu \right| \leq \epsilon \right\}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(a)** Does $P[X^n \in A^n] \to 1$?
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
    **(b)** Does $P[X^n \in A^n \cap B^n] \to 1$?
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
    **(c)** Show that $|A^n \cap B^n| \leq 2^{n(H+\epsilon)}$.
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
    **(d)** Show that $|A^n \cap B^n| \geq \frac{1}{2} 2^{n(H-\epsilon)}$ for $n$ large enough.
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
    ## 6. (EIT: 2.29) Inequalities.
    Let $X$, $Y$ and $Z$ be joint random variables. Prove the following inequalities and find conditions for equality.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(a)** $H(X, Y \mid Z) \geq H(X \mid Z)$.
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
    **(b)** $I(X, Y; Z) \geq I(X; Z)$.
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
    **(c)** $H(X, Y, Z) - H(X, Y) \leq H(X, Z) - H(X)$.
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
    **(d)** $I(X; Z \mid Y) \geq I(Z; Y \mid X) - I(Z; Y) + I(X; Z)$.
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

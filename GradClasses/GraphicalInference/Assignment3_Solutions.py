# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.19.0",
#     "matplotlib==3.10.8",
#     "nbconvert==7.17.0",
#     "nbformat==5.10.4",
#     "numpy==2.4.2",
#     "playwright==1.58.0",
#     "pydantic-ai==1.61.0",
#     "pyzmq",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


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
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    As a particular realization of the above example, consider an $8 \times 8$ chessboard and a special piece sitting on it. At any time step, the piece will stay still (with probability $1/2$) or move randomly to one of the neighboring positions (with probability $1/2$). Does this process satisfy the detailed balance condition (2.15)? Which positions on the chess board have lower and higher ‘energy’? Compute the partition function.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - $x \in \mathcal{X}$ be a position on the board
    - $N(x) \subset \mathcal{X}$ be the set of neighboring positions (reachable in one step).
    - $d(x) = |N(x)|$ be the degree of position $x$.
    - Stationary distribution $\pi(x)$ over positions $x$.
    - Transition probabilities:

    $$
    P(y|x) = \begin{cases}
    1/2 & \text{if } y = x \\
    \frac{1}{2 d(x)} & \text{if } y \in N(x) \\
    0 & \text{otherwise}
    \end{cases}
    $$

    - Balance condition: $\pi(x) P(y|x) = \pi(y) P(x|y)$.


    $$
    \begin{align*}
    \pi(x) \frac{1}{2 d(x)} = \pi(y) \frac{1}{2 d(y)} &\implies \frac{\pi(x)}{d(x)} = \frac{\pi(y)}{d(y)}
    \end{align*}
    $$

    This distribution does exist when $\pi(x) \propto d(x)$ and therefore satisfies the detailed balance equations

    Energy is inversely proportional to probability and therefore inverse with $d(x)$. This means corners are highest energy, edges are middle energy, and center squares are the lowest.


    * Corners (4 squares): 3 neighbors.
    * Edges not corners ($6 \times 4 = 24$ squares): 5 neighbors.
    * Interior ($6 \times 6 = 36$ squares): 8 neighbors.

    $Z = 4(3) + 24(5) + 36(8) = 12 + 120 + 288 = 420$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. (IPC: 2.02) Exercise 2.2 A two-level system.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This is the simplest non-trivial example: $X = \{1, 2\}$, $E(1) = \epsilon_1$, $E(2) = \epsilon_2$. Without loss of generality, we assume $\epsilon_1 < \epsilon_2$. This example can be used as a mathematical model for many physical systems, such as the spin-1/2 particle discussed above.

    Derive the following for the thermodynamic potentials (where $\Delta = \epsilon_2 - \epsilon_1$ is the energy gap):

    $$
    F(\beta) = \epsilon_1 - \frac{1}{\beta} \log(1 + e^{-\beta \Delta})
    $$

    $$
    U(\beta) = \epsilon_1 + \frac{\Delta e^{-\beta \Delta}}{1 + e^{-\beta \Delta}}
    $$

    $$
    S(\beta) = \frac{\beta \Delta e^{-\beta \Delta}}{1 + e^{-\beta \Delta}} + \log(1 + e^{-\beta \Delta})
    $$

    For $\epsilon_1 = -1$ and $\epsilon_2 = 1$, these functions are plotted below (see also Fig. 2.1).
    """)
    return


@app.cell
def _(np, plt):
    def F(beta, epsilon1=-1, epsilon2=1):
        """Free energy per spin."""
        delta = epsilon2 - epsilon1
        return -np.log(np.exp(-beta * epsilon1) + np.exp(-beta * epsilon2)) / beta

    def U(beta, epsilon1=-1, epsilon2=1):
        """Energy per spin."""
        delta = epsilon2 - epsilon1
        return (
            epsilon1 * np.exp(-beta * epsilon1) + epsilon2 * np.exp(-beta * epsilon2)
        ) / (np.exp(-beta * epsilon1) + np.exp(-beta * epsilon2))

    def S(beta, epsilon1=-1, epsilon2=1):
        """Entropy per spin."""
        delta = epsilon2 - epsilon1
        return beta * U(beta, epsilon1, epsilon2) - beta * F(beta, epsilon1, epsilon2)

    temperature_range = np.linspace(0.01, 3, 100)
    plt.figure(figsize=(12, 4))
    plt.plot(temperature_range, F(1 / temperature_range), label="Free Energy")
    plt.plot(
        temperature_range, U(1 / temperature_range), label="Energy", linestyle="--"
    )
    plt.plot(
        temperature_range, S(1 / temperature_range), label="Entropy", linestyle="-."
    )
    plt.xlabel("Temperature")
    plt.ylabel("value")
    plt.title("Thermodynamic Quantities for a Two-Level System")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    The reader should work out the asymptotics, and verify the general high- and low-temperature behavior shown above.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    \begin{align*}
    Z(\beta) &= \sum_{x \in \set{1, 2}} e^{-\beta E(x)} \\
    &= e^{-\beta \epsilon_1} + e^{-\beta \epsilon_2} \\
    &= e^{-\beta \epsilon_1}(1 + e^{-\beta(\epsilon_2 - \epsilon_1)}) \\
    &= e^{-\beta \epsilon_1}(1 + e^{-\beta \Delta})
    \\
    F(\beta) &= -\frac{1}{\beta} \ln Z(\beta) \\
    &= -\frac{1}{\beta} \left( -\beta \epsilon_1 + \ln(1 + e^{-\beta \Delta}) \right) \\
    &= \epsilon_1 - \frac{1}{\beta} \ln(1 + e^{-\beta \Delta})
    \\
    U(\beta)&= -\frac{\partial}{\partial \beta} \ln Z(\beta) \\
    &= \frac{\partial}{\partial \beta} [\beta F(\beta)] \\
    &= \frac{\partial}{\partial \beta} [\beta \epsilon_1 - \ln(1 + e^{-\beta \Delta})]\\
    &= \epsilon_1 - \frac{1}{1 + e^{-\beta \Delta}} \cdot (e^{-\beta \Delta}) \cdot (-\Delta) \\
    &= \epsilon_1 + \frac{\Delta e^{-\beta \Delta}}{1 + e^{-\beta \Delta}}\\
    S(\beta) &= \beta(U - F)\\
    &= \beta \left[ \left(\epsilon_1 + \frac{\Delta e^{-\beta \Delta}}{1 + e^{-\beta \Delta}}\right) - \left(\epsilon_1 - \frac{1}{\beta} \ln(1 + e^{-\beta \Delta})\right) \right] \\
    &= \beta \left[ \frac{\Delta e^{-\beta \Delta}}{1 + e^{-\beta \Delta}} + \frac{1}{\beta} \ln(1 + e^{-\beta \Delta}) \right] \\
    &= \frac{\beta \Delta e^{-\beta \Delta}}{1 + e^{-\beta \Delta}} + \ln(1 + e^{-\beta \Delta})
    \end{align*}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. (FGML: 4.01) Ising Effective field and single-spin conditional.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Consider an Ising model on graph $G$ with energy

    $$
    E(\sigma) = - \sum_{(i,j) \in E} J_{ij} \sigma_i \sigma_j - \sum_i h_i^{\text{ext}} \sigma_i, \quad \sigma \in \{\pm 1\}^n
    $$

    Define $h_i^{\text{eff}} = h_i^{\text{ext}} + \sum_{j \in \partial i} J_{ij} \sigma_j$.

    (a) Derive the conditional distribution $P(\sigma_i = +1 | \sigma_{-i})$ in logistic form.

    (b) Show that $\mathbb{E}[\sigma_i | \sigma_{-i}] = \tanh(\beta h_i^{\text{eff}})$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(a)** The conditional distribution is given by:

    $$
    \begin{align*}
    P(\sigma_i | \sigma_{-i}) &= \frac{e^{-\beta E(\sigma)}}{Z(\sigma_{-i})} \propto e^{-\beta E(\sigma)}\\
    E(\sigma) &= - \sigma_i \left( h_i^{\text{ext}} + \sum_{j \in \partial i} J_{ij} \sigma_j \right)\\
    &=  - \sigma_i h_i^{\text{eff}}\\
    P(\sigma_{i} =  +1 | \sigma_{-i}) &=  \frac{e^{\beta h_i^{\text{eff}}}}{e^{\beta h_i^{\text{eff}}} + e^{-\beta h_i^{\text{eff}}}} \\
    &=  \frac{1}{1 + e^{-2\beta h_i^{\text{eff}}}}\\
    &= \sigma(2\beta h_i^{\text{eff}})
    \end{align*}
    $$

    This is the logistic function (sigmoid) with argument $2\beta h_i^{\text{eff}}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(b)** The conditional expectation is:

    $$
    \begin{align}
    \mathbb{E}[\sigma_i | \sigma_{-i}] &= (+1) P(\sigma_i = +1 | \sigma_{-i}) + (-1) P(\sigma_i = -1 | \sigma_{-i}) \\
    &= \frac{e^{\beta h_i^{\text{eff}}} - e^{-\beta h_i^{\text{eff}}}}{e^{\beta h_i^{\text{eff}}} + e^{-\beta h_i^{\text{eff}}}} \\
    &= \tanh(\beta h_i^{\text{eff}})
    \end{align}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. (FGML: 4.02) Mean-field self-consistency.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The following fully-connected Ising model on $n$ spins, known as the Curie–Weiss model, is defined by

    $$
    E(\sigma) = - \frac{J}{n} \sum_{1 \le i < j \le n} \sigma_i \sigma_j - h \sum_{i=1}^n \sigma_i, \quad \sigma_i \in \{\pm 1\}, J > 0.
    $$

    Let $m = \frac{1}{n} \sum_i \sigma_i$ be the magnetization.

    (a) In this model, one assumes that the magnetization concentrates around its average $m \approx \frac{1}{n} \sum_i \mathbb{E}[\sigma_i]$. Then, this is used to compute new means $\mathbb{E}[\sigma_i | \sigma_{-i}]$ based on the other spins. Derive the self-consistency equation for $m$ implied by equilibrium.

    (b) For $h = 0$, determine the range of $\beta$ for which the $m = 0$ fixed point is stable.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(a)** The effective field on spin $i$ is:

    $$
    h_i^{\text{eff}} = h + \sum_{j \neq i} \frac{J}{n} \sigma_j
    $$

    Approximating the sum using the mean magnetization $m = \frac{1}{n} \sum_k \sigma_k$:

    $$
    \sum_{j \neq i} \frac{J}{n} \sigma_j \approx \frac{J}{n} (n m) = J m
    $$

    (ignoring the $O(1/n)$ correction for excluding $i$).

    So $h_i^{\text{eff}} \approx h + J m$.

    From Problem 3, the equilibrium expectation for a single spin in a fixed field is $\tanh(\beta h^{\text{eff}})$. Self-consistency requires the average magnetization to match this expectation:

    $$
    m = \tanh(\beta(h + Jm))
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(b)** For $h=0$, the equation is $m = \tanh(\beta J m)$.
    The fixed point $m=0$ is always a solution ($\tanh(0) = 0$).

    For $m=0$ to be a stable fixed point, we need the slope to be less than 1 (so the contraction maps $m$ back to 0). The slope of the RHS at $m=0$ is:

    $$
    \frac{d}{dm} \tanh(\beta J m) \Big|_{m=0} = \text{sech}^2(0) \cdot (\beta J) = \beta J
    $$

    $$
    \beta J < 1 \implies \beta < 1/J \quad \text{or} \quad T > J
    $$

    If $\beta J > 1$, the slope is $>1$, and $m=0$ becomes unstable, with new stable solutions appearing at $m \neq 0$ (ferromagnetic phase transition).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. (IPC: 2.06) Exercise 2.6
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Consider the one-dimensional Ising model in zero field, $B = 0$. Show that when $\delta N < i < j < (1 - \delta)N$, the correlation function $\langle \sigma_i \sigma_j \rangle$ is, in the large-$N$ limit,

    $$
    \langle \sigma_i \sigma_j \rangle = e^{-|i-j|/\xi(\beta)}, \text{ where } \xi(\beta) = - \frac{1}{\log(\tanh(\beta))}
    $$

    [Hint: You can either use the general transfer matrix formalism or, more simply, use the identity $e^{\beta \sigma_i \sigma_{i+1}} = \cosh(\beta) [1 + \sigma_i \sigma_{i+1} \tanh(\beta)]$.]
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    \begin{align*}
    \mathbb{P}(\sigma) \propto\prod_{k} e^{\beta \sigma_k \sigma_{k+1}} &= \prod_k \cosh(\beta) (1 + \tanh(\beta) \sigma_k \sigma_{k+1})\\
    Z &= 2^N (\cosh \beta)^{N-1}
    \end{align*}
    $$

    $$
    \begin{align*}
    \langle \sigma_i \sigma_{j} \rangle&= \frac{1}{Z} \sum_\sigma \sigma_i \sigma_j \prod_k \cosh(\beta) (1 + \tanh(\beta) \sigma_k \sigma_{k+1})\\
    &= \frac{2^{N}\cosh(\beta)^{N-1}}{Z}\sum_\sigma \sigma_{i} \sigma_{j} \prod_{k}(1 + \tanh(\beta) \sigma_k \sigma_{k+1})\\
    &= \tanh(\beta)^{j-i} \to \text{all } \sigma \text{ square to }1 \text{ \& appear twice}\\
    &= \tanh(\beta)^{|i-j|}\\
    \end{align*}
    $$

    $$
    \begin{align*}
    \tanh(\beta)^{|i-j|} &= e^{|i-j| \ln(\tanh \beta)}\\
    &= e^{-|i-j| [-\ln(\tanh \beta)]}\\
    \frac{1}{\xi(\beta)} &= -\ln(\tanh \beta)\\
    \xi(\beta) &= -\frac{1}{\ln(\tanh \beta)}\\
    \tanh(\beta)^{|i-j|} &= e^{|i-j|/\xi(\beta)}
    \end{align*}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Practice Problems (do not hand in):
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1. (IPC: 2.03) Exercise 2.3
    We return to the example of the previous section: one water molecule, modeled as a point, in a bottle. We consider the case of a cylindrical bottle of base $B \subset \mathbb{R}^2$ (surface area $|B|$) and height $d$.

    Using the energy function $E(x) = wh(x)$ (see (2.13)), where $h(x)$ is the height of $x$ and $w$ is a constant, derive the following explicit expressions for the thermodynamic potentials:

    $$
    F(\beta) = -\frac{1}{\beta} \log(|B|) - \frac{1}{\beta} \log \left( \frac{1 - e^{-\beta wd}}{\beta w} \right)
    $$

    $$
    U(\beta) = \frac{1}{\beta} - \frac{wd}{e^{\beta wd} - 1}
    $$

    $$
    S(\beta) = \log(|B|d) + 1 - \frac{\beta wd}{e^{\beta wd} - 1} - \log \left( \frac{\beta wd}{1 - e^{-\beta wd}} \right)
    $$

    Note that the internal-energy formula can be used to compute the average height of the molecule $\langle h(x) \rangle = U(\beta)/w$. This is a consequence of the definition of the energy (see eqn (2.13) and eqn (2.21)). If we plug in the correct constant $w$, we can find that the average height falls below $49.99\%$ of the height of the bottle $d = 20\text{cm}$ only when the temperature is below $3.2$ K.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2. (IPC: 2.04) Exercise 2.4
    Using eqns (2.34)-(2.36), derive the low-temperature expansions:

    $$
    F(\beta) = -\frac{1}{\beta} \log \frac{|B|}{\beta w} + \Theta(e^{-\beta wd})
    $$

    $$
    U(\beta) = \frac{1}{\beta} + \Theta(e^{-\beta wd})
    $$

    $$
    S(\beta) = \log \frac{|B|e}{\beta w} + \Theta(e^{-\beta wd})
    $$

    In this case $X$ is continuous, and the energy has no gap. Nevertheless, these results can be understood as follows: at low temperature, the molecule is confined to a layer of height of order $1/(\beta w)$ above the bottom of the bottle. It therefore occupies a volume of size $|B|/(\beta w)$. Its entropy is approximately given by the logarithm of such a volume.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3. (IPC: 2.05) Exercise 2.5
    Let us reconsider the above example and assume the bottle to have a different shape, for instance a sphere of radius $R$. In this case it is difficult to compute explicit expressions for the thermodynamic potentials, but one can easily compute the low-temperature expansions. For the entropy, one gets at large $\beta$:

    $$
    S(\beta) = \log \left( \frac{2 \pi e^2 R}{\beta^2 w} \right) + \Theta\left(\frac{1}{\beta}\right)
    $$

    The reader should try to understand the difference between this result and eqn (2.39) and provide an intuitive explanation, as in the previous example. Physicists say that the low-temperature thermodynamic potentials reveal the ‘low-energy structure’ of the system.
    """)
    return


if __name__ == "__main__":
    app.run()

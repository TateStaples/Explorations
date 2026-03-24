# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.19.0",
#     "nbconvert==7.17.0",
#     "nbformat==5.10.4",
#     "numpy==2.4.2",
#     "playwright==1.58.0",
#     "pyzmq",
# ]
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Assignment 2 Solutions
    Due Monday 02/09/26
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Problem 1 (EIT 2.04) - Entropy of functions of a random variable

    Let $X$ be a discrete random variable. Show that the entropy of a function of $X$ is less than or equal to the entropy of $X$ by justifying the following steps:

    $$
    \begin{align}
    H(X, g(X)) &=^{(a)} H(X) + H(g(X) | X)\\
    &=^{(b)} H(X) \\
    H(X) + H(g(X) | X) &=^{(c)} H(g(X)) + H(X | g(X)) \\
    &\geq^{(d)} H(g(X))
    \end{align}
    $$

    Thus $H(g(X)) \leq H(X)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Step a)

    Any multivariate distribution can be decomposed into a product of a marginal distribution and a conditional distribution. In this case, we can write the joint distribution of $X$ and $g(X)$ as:
    $$
    P(X, g(X)) = P(X) P(g(X) | X)
    $$
    Taking the logarithm and then the expectation, we get:
    $$
    \begin{align}
    H(X, g(X)) &= -\sum_{x, g(x)} P(X, g(X)) \log P(X, g(X))\\
    &= -\sum_{x, g(x)} P(X) P(g(X) | X) \log [P(X) P(g(X) | X)]\\
    &= -\sum_{x, g(x)} P(X) P(g(X) | X) [\log P(X) + \log P(g(X) | X)]\\
    &= -\sum_{x} P(X) \log P(X) - \sum_{x, g(x)} P(X) P(g(X) | X) \log P(g(X) | X)\\
    &= H(X) + H(g(X) | X)
    \end{align}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Step b)
    Since $g(X)$ is a function of $X$, it is completely determined by $X$. Therefore, the conditional entropy $H(g(X) | X)$ is zero, because $g(X)$ is a completely known value (and thus has no uncertainty) given $X$. Therefore $H(X) + 0 = H(X)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Step c)
    $H(X) + H(g(X) | X) =^{(a)} H(X, g(X)) = H(g(X)) + H(X | g(X))$. This can be seen by the same logic as in step (a), but now we decompose the joint distribution $P(X, g(X))$ as $P(g(X)) P(X | g(X))$ instead of $P(X) P(g(X) | X)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Step d)

    $H(X | g(X)) \geq 0$ because functions can have multiple inputs that map to the same output, but they cannot have multiple outputs for the same input. This induces uncertainty in $X$ given $g(X)$, meaning that $H(X | g(X))$ is $0$ when $g(X)$ is a one-to-one function of $X$ and is positive otherwise. Therefore, $H(g(X)) + H(X | g(X)) \geq H(g(X))$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Final Conclusion

    Using the equality $H(X, g(X)) = H(X) + H(g(X) | X) = H(g(X)) + H(X | g(X))$ and our results:
    $$
    \begin{align}
    H(X, g(X)) &= H(X) \\
    H(X, g(X)) &\geq H(g(X))
    \end{align}
    $$
    we can conclude that $H(g(X)) \leq H(X)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Problem 2 (EIT 2.12) - Example of joint entropy

    Let $p(x, y)$ be given by:

    | X/Y | 0   | 1   |
    |-----|-----|-----|
    | 0   | 1/3 | 1/3 |
    | 1   | 0   | 1/3 |

    Find:

    * **(a)** $H(X)$, $H(Y)$
    * **(b)** $H(X|Y)$, $H(Y|X)$
    * **(c)** $H(X, Y)$
    * **(d)** $H(Y) - H(Y|X)$
    * **(e)** $I(X; Y)$
    * **(f)** Draw a Venn diagram for the quantities in parts (a) through (e)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part a)

    To find $H(X)$ and $H(Y)$, we first need to compute the marginal distributions of $X$ and $Y$.

    $$
    \begin{align}
    P(X=0) &= P(X=0, Y=0) + P(X=0, Y=1) = \frac{1}{3} + \frac{1}{3} = \frac{2}{3} \\
    P(X=1) &= P(X=1, Y=0) + P(X=1, Y=1) = 0 + \frac{1}{3} = \frac{1}{3} \\
    P(Y=0) &= P(X=0, Y=0) + P(X=1, Y=0) = \frac{1}{3} + 0 = \frac{1}{3} \\
    P(Y=1) &= P(X=0, Y=1) + P(X=1, Y=1) = \frac{1}{3} + \frac{1}{3} = \frac{2}{3}
    \end{align}
    $$

    Now we can compute the entropies using the formula $H(X) = -\sum_x P(X=x) \log P(X=x)$ and similarly for $H(Y)$.

    $$
    \begin{align}
    H(X) &= -\left( \frac{2}{3} \log \frac{2}{3} + \frac{1}{3} \log \frac{1}{3} \right) \\
    H(Y) &= -\left( \frac{1}{3} \log \frac{1}{3} + \frac{2}{3} \log \frac{2}{3} \right)
    \end{align}
    $$
    """)
    return


@app.cell
def _(np):
    # Python numerics for Part (a)
    H_x = - (2/3 * np.log2(2/3) + 1/3 * np.log2(1/3))
    print(f"H(X) = {H_x:.4f} bits")
    H_y = - (1/3 * np.log2(1/3) + 2/3 * np.log2(2/3))
    print(f"H(Y) = {H_y:.4f} bits")
    return H_x, H_y


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part b)

    To find $H(X|Y)$ and $H(Y|X)$, we need to compute the conditional probabilities first.
    $$
    \begin{align}
    P(X=0|Y=0) &= \frac{P(X=0, Y=0)}{P(Y=0)} = \frac{\frac{1}{3}}{\frac{1}{3}} = 1 \\
    P(X=1|Y=0) &= \frac{P(X=1, Y=0)}{P(Y=0)} = \frac{0}{\frac{1}{3}} = 0 \\
    P(X=0|Y=1) &= \frac{P(X=0, Y=1)}{P(Y=1)} = \frac{\frac{1}{3}}{\frac{2}{3}} = \frac{1}{2} \\
    P(X=1|Y=1) &= \frac{P(X=1, Y=1)}{P(Y=1)} = \frac{\frac{1}{3}}{\frac{2}{3}} = \frac{1}{2} \\
    P(Y=0|X=0) &= \frac{P(X=0, Y=0)}{P(X=0)} = \frac{\frac{1}{3}}{\frac{2}{3}} = \frac{1}{2} \\
    P(Y=1|X=0) &= \frac{P(X=0, Y=1)}{P(X=0)} = \frac{\frac{1}{3}}{\frac{2}{3}} = \frac{1}{2} \\
    P(Y=0|X=1) &= \frac{P(X=1, Y=0)}{P(X=1)} = \frac{0}{\frac{1}{3}} = 0 \\
    P(Y=1|X=1) &= \frac{P(X=1, Y=1)}{P(X=1)} = \frac{\frac{1}{3}}{\frac{1}{3}} = 1
    \end{align}
    $$

    Now we can compute the conditional entropies:
    $$
    \begin{align}
    H(X|Y) &= P(Y=0) H(X|Y=0) + P(Y=1) H(X|Y=1) \\
    &= \frac{1}{3} \cdot 0 + \frac{2}{3} \cdot \left( -\left( \frac{1}{2} \log \frac{1}{2} + \frac{1}{2} \log \frac{1}{2} \right) \right) \\
    &= \frac{2}{3} \cdot 1 = \frac{2}{3} \\
    H(Y|X) &= P(X=0) H(Y|X=0) + P(X=1) H(Y|X=1) \\
    &= \frac{2}{3} \cdot \left( -\left( \frac{1}{2} \log \frac{1}{2} + \frac{1}{2} \log \frac{1}{2} \right) \right) + \frac{1}{3} \cdot 0 \\
    &= \frac{2}{3} \cdot 1 = \frac{2}{3}
    \end{align}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part c)

    $$
    \begin{align}
    H(X, Y) &= H(X) + H(Y|X) \\
    &= -\left( \frac{2}{3} \log \frac{2}{3} + \frac{1}{3} \log \frac{1}{3} \right) + \frac{2}{3} \\
    \end{align}
    $$
    """)
    return


@app.cell
def _(H_x):
    # Python numerics for Part (c)
    H_xy = H_x + (2/3)
    print(f"H(X, Y) = {H_xy:.4f} bits")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part d)
    $$
    H(Y) - H(Y|X) = -\left( \frac{1}{3} \log \frac{1}{3} + \frac{2}{3} \log \frac{2}{3} \right) - \frac{2}{3} \\
    $$
    """)
    return


@app.cell
def _(H_y):
    # Python numerics for Part (d)
    H_mutual = H_y - (2/3)
    print(f"I(X; Y) = H(Y) - H(Y|X) = {H_mutual:.4f} bits")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part e)
    This is the mutual information, which is the same value as part (d) because $I(X; Y) = H(Y) - H(Y|X)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part f)
    > Venn diagram sourced from https://en.wikipedia.org/wiki/Mutual_information#/media/File:Entropy-mutual-information-relative-entropy-relation-diagram.svg
    """)
    return


@app.cell
def _(mo):
    path = '/Users/tatestaples/Code/GradClasses/GraphicalInference/files/Entropy-mutual-information-relative-entropy-relation-diagram.svg'
    mo.image(src=path, width="180px", height="180px", rounded=True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Problem 3 (GRIT 1.06) - Entropy of time to first success

    A fair coin is flipped repeatedly. Let $X_i$ denote the number of flips required for the $i$-th occurrence of a head. (i.e., if the coin is a head on the first flip then $X_1 = 1$). For example, the flip sequence "TTTHTHTTH..." generates the outcome $(X_1, X_2, X_3, \ldots) = (4, 2, 3, \ldots)$.

    * **(a)** Find the entropy $H(X_1)$ in bits.
    * **(b)** Find the conditional entropy $H(X_2|X_1)$ in bits.
    * **(c)** Show that $H(X_1) \leq H(X_2) \leq 2H(X_1)$. [Hint: try expressing $H(X_1, X_2)$ two different ways.]
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part a)

    This is a geometric distribution with success probability $p = \frac{1}{2}$. The entropy of a geometric distribution is given by:
    $$
    H(X) = -\sum_{k=1}^{\infty} p(1-p)^{k-1} \log_2(p(1-p)^{k-1})
    $$
    For $p = \frac{1}{2}$, this becomes:
    $$
    H(X_1) = -\sum_{k=1}^{\infty} \frac{1}{2} \left(\frac{1}{2}\right)^{k-1} \log_2\left(\frac{1}{2} \left(\frac{1}{2}\right)^{k-1}\right)
    $$
    Calculating this sum gives:
    $$
    H(X_1) = 2 \text{ bits}
    $$

    You can understand this as a simple weighted average on number of bits. 50% of the time you get 1 bit (H), 25% of the time you get 2 bits (TH), 12.5% of the time you get 3 bits (TTH), etc. This converges to 2 bits.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part b)
    > I am assuming $X_2$ is the number of flips total til the second head (description), not the number of flips til the second head after the first head (example).

    $X_2 = X_1 + Z$, where $Z$ is the number of flips after the first head til the second head. Note that $Z$ is independent of $X_1$ and has the same distribution as $X_1$ (both are geometric with $p = \frac{1}{2}$).


    $P(X_2|X_1) = P(Z)$ because $X_2$ is completely determined by $X_1$ and $Z$. Since $Z$ has the same distribution as $X_1$ and $X_1$ is completely given, the entire uncertainty in $X_2$ given $X_1$ is just the uncertainty in $Z$, which is the same as the uncertainty in $X_1$. Therefore, $H(X_2|X_1) = H(Z) = H(X_1) = 2$ bits.
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
    ### (Part d)

    Upper Bound:
    $$
    \begin{align}
    H(X_1, X_2) &= H(X_1) + H(X_2 | X_1) \\
    &= H(X_1) + H(Z) \
    &= H(X_1) + H(X_1) \\
    &= 2H(X_1)
    \end{align}
    $$
    Since entropy is non-negative, we have $H(X_2) \leq H(X_1, X_2) \leq 2H(X_1)$.

    Lower Bound:
    $$
    H(X_2|X_1) = H(X_1)
    $$

    Since providing $X_1$ must give non-negative information about $X_2$, we have $H(X_2) \geq H(X_1)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Problem 4 (GRIT 2.01) - Cross-Entropy Loss

    Consider a problem where we are given $n$ iid samples $(x_i, y_i) \in \mathcal{X} \times \mathcal{Y}$ drawn from $p(x, y)$ and the goal is to learn a predictor $f$ that maps each $x \in \mathcal{X}$ to a prediction of $y \in \mathcal{Y}$. Without loss of generality, we will assume the prediction is an estimate of the posterior distribution of $y$ given $x$. A common approach in machine learning is to define a loss function and then use the predictor that minimizes the average loss.

    Let $P(\mathcal{Y})$ denote the set of pmfs over $\mathcal{Y}$. Then, the cross-entropy loss $\ell : \mathcal{Y} \times P(\mathcal{Y}) \to \mathbb{R}_{\geq 0}$ maps the true $y \in \mathcal{Y}$ and the predicted posterior $q(y)$ to the non-negative real loss

    $$\ell(y, q) = \log \frac{1}{q(y)}$$

    For each $\theta \in \mathbb{R}^d$, let $f_\theta(x) : \mathcal{X} \to P(\mathcal{Y})$ be a candidate predictor that maps each $x \in \mathcal{X}$ to a pmf over $\mathcal{Y}$. Then, the average cross-entropy loss is given by

    $$L_n(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(y_i, f_\theta(x_i))$$

    The following should help illuminate an important connection between machine learning and information theory. Note: One can denote the pmf $f_\theta(x)$ evaluated at $y \in \mathcal{Y}$ by $f_\theta(x)(y)$.

    * **(a)** To what non-random value will $L_n(\theta)$ converge to as $n \to \infty$?
    * **(b)** For fixed $\theta$, write the quantity in (a) as $H(Y|X)$ plus the average (over $X$) divergence between $p(y|x)$ and $q(y|x) = f_\theta(x)(y)$.
    * **(c)** Use the non-negativity of divergence to identify the minimum possible cross-entropy loss and choice of $f_\theta$ that achieves it.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part a)
    > From the notes
    **KL and model fitting.** If the true distribution is p, then
    $$
    \mathbb{E}_p[− \log q_\theta(X)] = H(p, q_\theta) = H(p) + D(p∥q_\theta).
    $$
    where $p = p(y |x)$ and $q_\theta = f_\theta(x)(y)$. Therefore, as $n \to \infty$, the average cross-entropy loss $L_n(\theta)$ converges to $\mathbb{E}_{p(x, y)}[\ell(y, f_\theta(x))] = H(Y|X) + D(p(y|x)∥f_\theta(x)(y))$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part b)
    $$
    L(\theta) = H(Y|X) + \mathbb{E}_{p(x)}[D(p(y|x)∥f_\theta(x)(y))]
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part c)

    The minimum possible cross-entropy loss is $H(Y|X)$, which is achieved when the average divergence between $p(y|x)$ and $q(y|x) = f_\theta(x)(y)$ is zero. This occurs when $f_\theta(x)(y) = p(y|x)$ for all $x \in \mathcal{X}$, meaning that the predictor perfectly matches the true conditional distribution of $Y$ given $X$.

    This value can still be positive because $y$ can be uncertain given $x$. For example, if $Y$ is a noisy label of $X$, then even the best predictor will have some uncertainty in its predictions, leading to a positive cross-entropy loss.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Problem 5 (EIT 3.04) - AEP

    Let $X_i$ be iid $\sim p(x)$, $x \in \{1, 2, \ldots, m\}$. Let $\mu = E[X]$ and $H = H(X)$. Let

    $$A^n = \left\{x^n \in \mathcal{X}^n : \left| -\frac{1}{n} \log p(x^n) - H \right| \leq \epsilon \right\}$$

    $$B^n = \left\{x^n \in \mathcal{X}^n : \left| \frac{1}{n} \sum_{i=1}^n X_i - \mu \right| \leq \epsilon \right\}$$

    * **(a)** Does $P[X^n \in A^n] \to 1$?
    * **(b)** Does $P[X^n \in A^n \cap B^n] \to 1$?
    * **(c)** Show that $|A^n \cap B^n| \leq 2^{n(H+\epsilon)}$
    * **(d)** Show that $|A^n \cap B^n| \geq \left(\frac{1}{2}\right) 2^{n(H-\epsilon)}$ for $n$ large enough
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part a)
    Yes, for any $\epsilon > 0$, the AEP states that as $n \to \infty$, the probability that $X^n$ is the typical set $\mathcal{T}_\epsilon^{(n)}$ goes to 1. Since $A^n$ is defined as the set of sequences that are within $\epsilon$ of the entropy $H$, it is essentially the typical set.

    Given $x^n$ in $\mathcal{T}_\epsilon^{(n)}$, we have:
    $$
    \begin{align}
    2^{-n(H+\epsilon)} &\leq p(x^n) \leq 2^{-n(H-\epsilon)} \\
    |-\frac{1}{n} \log p(x^n) - H| &\leq \epsilon
    \end{align}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part b)

    We know by Law of Large Numbers that sample means converge to the true mean, so $P[X^n \in B^n] \to 1$. Since $P[X^n \in A^n] \to 1$ and $P[X^n \in B^n] \to 1$, we have $P[X^n \in A^n \cap B^n] \to 1$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part c)
    Since $A^n \cap B^n \subseteq A^n$, all $x^n$ in this set satisfy the upper bound on $p(x^n)$ given by part (a):
    $$
    \begin{align}
    -\frac{1}{n} \log p(x^n) &\leq H + \epsilon \\
    p(x^n) &\geq 2^{-n(H+\epsilon)} \\
    1 &\geq \sum_{x^n \in A^n \cap B^n} p(x^n) \geq |A^n| 2^{-n(H+\epsilon)} \\
    |A^n| &\leq 2^{n(H+\epsilon)}
    \end{align}
    $$
    Since $|A^n \cap B^n| \leq |A^n|$, we have $|A^n \cap B^n| \leq 2^{n(H+\epsilon)}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part d)
    Since $A^n \cap B^n \subseteq A^n$, all $x^n$ in this set satisfy the lower bound on $p(x^n)$ given by part (a):

    $$
    \begin{align}
    -\frac{1}{n} \log p(x^n) &\geq H - \epsilon \\
    p(x^n) &\leq 2^{-n(H-\epsilon)} \\
    \mathbb{P}[X^n \in A^n \cap B^n] &= \sum_{x^n \in A^n \cap B^n} p(x^n) \\
    &\leq |A^n \cap B^n| 2^{-n(H-\epsilon)} \\
    \mathbb{P}[X^n \in A^n \cap B^n] &> \frac{1}{2} \text{ for $n$ large enough} \\
    |A^n \cap B^n| &> \left(\frac{1}{2}\right) 2^{n(H-\epsilon)}
    \end{align}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Problem 6 (EIT 2.29) - Inequalities

    Let $X$, $Y$, and $Z$ be joint random variables. Prove the following inequalities and find conditions for equality.

    * **(a)** $H(X, Y | Z) \geq H(X | Z)$
    * **(b)** $I(X, Y; Z) \geq I(X; Z)$
    * **(c)** $H(X, Y, Z) - H(X, Y) \leq H(X, Z) - H(X)$
    * **(d)** $I(X; Z | Y) \geq I(Z; Y | X) - I(Z; Y) + I(X; Z)$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part a)

    $$
    \begin{align}
    H(X, Y | Z) &= H(X | Z) + H(Y | X, Z) \\
    &\geq H(X | Z)
    \end{align}
    $$

    This is given by the decomposition of the joint entropy $H(X, Y | Z)$ into the sum of the marginal entropy $H(X | Z)$ and the conditional entropy $H(Y | X, Z)$. Since $H(Y | X, Z)$ is non-negative, we have $H(X, Y | Z) \geq H(X | Z)$.

    Conditions for equality: $H(Y | X, Z) = 0$, which occurs when $Y$ is a deterministic function of $X$ and $Z$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part b)

    $$
    \begin{align}
    I(X, Y; Z) &= H(X, Y) - H(X, Y | Z) \\
    &= [H(X) + H(Y | X)] - [H(X | Z) + H(Y | X, Z)] \\
    I(X; Z) &= H(X) - H(X | Z) \\
    I(X, Y; Z) - I(X; Z) &= H(Y | X) - H(Y | X, Z) \\
    &\geq 0 \implies I(X, Y; Z) \geq I(X; Z)
    \end{align}
    $$

    Condition for equality: $I(X, Y; Z) = I(X; Z)$ if and only if $H(Y | X) = H(Y | X, Z)$, which means that $Y$ is conditionally independent of $Z$ given $X$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part c)
    $$
    \begin{align}
    H(X, Y, Z) - H(X, Y) &= [H(X, Z) + H(Y | X, Z)] - [H(X) + H(Y | X)] \\
    &= [H(X, Z) - H(X)] + [H(Y | X, Z) - H(Y | X)] \\
    &= [H(X, Z) - H(X)] - I(Y; Z | X) \\
    &\implies H(X, Y, Z) - H(X, Y) \leq H(X, Z) - H(X) \\
    &\to \text{non-negativity of mutual information}
    \end{align}
    $$

    Condtions for equality: $I(Y; Z | X) = 0$, meaning that $Y$ and $Z$ are conditionally independent given $X$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (Part d)

    $$
    \begin{align}
    I(X; Z | Y) &= H(X | Y) - H(X | Y, Z) \\
    I(Z; Y | X) &= H(Z | X) - H(Z | X, Y) \\
    I(Z; Y) &= H(Z) - H(Z | Y) \\
    I(X; Z) &= H(Z) - H(Z | X) \\
    I(X; Z | Y) - I(Z; Y | X) + I(Z; Y) - I(X; Z) &= H(Z | Y) - H(Z | X, Y) \\
    &= I(Z; X | Y) \\
    &= I(X; Z | Y) \\
    &\implies I(X; Z | Y) \geq I(Z; Y | X) - I(Z; Y) + I(X; Z)
    \end{align}
    $$

    Conditions for equality: always
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Practice Problems (do not hand in)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Practice Problem 1 (GRIT 1.01) - Union Bounds

    Let $A_1, A_2, \ldots$ be events defined on a sample space $\Omega$. Prove the following two inequalities.

    * **(a)** $P\left[\bigcup_{i=1}^\infty A_i\right] \leq \sum_{i=1}^\infty P[A_i]$
    * **(b)** $P\left[\bigcap_{i=1}^\infty A_i\right] \geq 1 - \sum_{i=1}^\infty P[A_i^C]$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Practice Problem 2 (GRIT 1.02) - Independence

    Construct examples that meet the following conditions.

    * **(a)** Three discrete random variables $X$, $Y$, $Z$ that are not independent but are pairwise independent.
    * **(b)** Three discrete random variables $X$, $Y$, and $Z$ such that $X$ and $Y$ are dependent but $X$ and $Y$ are conditionally independent given $Z$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Practice Problem 3 (EIT 2.01) - Coin flips

    A fair coin is flipped until the first head occurs. Let $X$ denote the number of flips required.

    * **(a)** Find the entropy $H(X)$ in bits. The following formulas may be useful:

    $$\sum_{n=0}^\infty r^n = \frac{1}{1-r}, \quad \sum_{n=0}^\infty n r^n = \frac{r}{(1-r)^2}$$

    * **(b)** A random variable $X$ is drawn according to this distribution. Find an "efficient" sequence of yes-no questions of the form: "Is $X$ contained in the set $S$". Compare $H(X)$ to the expected number of questions required to determine $X$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Practice Problem 4 (EIT 3.01) - Markov's inequality and Chebyshev's inequality

    * **(a)** (Markov's inequality) For any nonnegative random variable $X$ and any $t > 0$, show that

    $$\Pr\{X \geq t\} \leq \frac{EX}{t}$$

    Exhibit a random variable that achieves this inequality with equality.

    * **(b)** (Chebyshev's inequality) Let $Y$ be a random variable with mean $\mu$ and variance $\sigma^2$. By letting $X = (Y - \mu)^2$, show that for any $\epsilon > 0$

    $$\Pr\{|Y - \mu| > \epsilon\} \leq \frac{\sigma^2}{\epsilon^2}$$

    * **(c)** (Weak law of large numbers) Let $Z_1, Z_2, \ldots, Z_n$ be a sequence of i.i.d. random variables with mean $\mu$ and variance $\sigma^2$. Let $\bar{Z}_n = \frac{1}{n} \sum_{i=1}^n Z_i$ be the sample mean. Show that

    $$\Pr\left\{|\bar{Z}_n - \mu| > \epsilon\right\} \leq \frac{\sigma^2}{n\epsilon^2}$$

    Thus, $\Pr\{|\bar{Z}_n - \mu| > \epsilon\} \to 0$ as $n \to \infty$. This is known as the weak law of large numbers.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Practice Problem 5 (EIT 2.32) - Fano

    We are given the following joint distribution on $(X, Y)$:

    | X/Y | a    | b    | c    |
    |-----|------|------|------|
    | 1   | 1/6  | 1/12 | 1/12 |
    | 2   | 1/12 | 1/6  | 1/12 |
    | 3   | 1/12 | 1/12 | 1/6  |

    Let $\hat{X}(Y)$ be an estimator for $X$ (based on $Y$) and let $P_e = \Pr\{\hat{X}(Y) \neq X\}$.

    * **(a)** Find the minimum probability of error estimator $\hat{X}(Y)$ and the associated $P_e$.
    * **(b)** Evaluate Fano's inequality for this problem and compare.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Practice Problem 6 (EIT 2.28) - Mixing increases entropy

    Show that the entropy of the probability distribution $(p_1, \ldots, p_i, \ldots, p_j, \ldots, p_m)$ is less than the entropy of the distribution $(p_1, \ldots, \frac{p_i + p_j}{2}, \ldots, \frac{p_i + p_j}{2}, \ldots, p_m)$. Show that in general any transfer of probability that makes the distribution more uniform increases the entropy.
    """)
    return


if __name__ == "__main__":
    app.run()

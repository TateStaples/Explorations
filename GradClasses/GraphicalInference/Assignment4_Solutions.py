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

    **Solution:**

    The log-likelihood of the data is:

    $$
    \begin{align*}
    \frac{1}{N} \mathcal{L}(\theta) &= \frac{1}{N} \sum_{\ell=1}^N \log p_\theta(x^{(\ell)}) \\
    &= \frac{1}{N} \sum_{\ell=1}^N \left( \theta_{x^{(\ell)}} - \log Z(\theta) \right) \\
    &= \frac{1}{N} \sum_{x=1}^m N_x \theta_x - \log \sum_{u=1}^m \exp(\theta_u)
    \end{align*}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(b)** Compute $\frac{\partial}{\partial \theta_x} \frac{1}{N} \mathcal{L}(\theta)$ and simplify it to the difference between data and model terms.

    **Solution:**

    Taking the derivative with respect to $\theta_x$:

    $$
    \begin{align*}
    \frac{\partial}{\partial \theta_x} \left( \frac{1}{N} \mathcal{L}(\theta) \right) &= \frac{\partial}{\partial \theta_x} \left( \sum_{u=1}^m \hat{p}_{\mathcal{D}}(u) \theta_u - \log Z(\theta) \right) \\
    &= \hat{p}_{\mathcal{D}}(x) - \frac{1}{Z(\theta)} \frac{\partial}{\partial \theta_x} Z(\theta) \\
    &= \hat{p}_{\mathcal{D}}(x) - \frac{\exp(\theta_x)}{Z(\theta)} \\
    &= \hat{p}_{\mathcal{D}}(x) - p_\theta(x)
    \end{align*}
    $$

    This is exactly the difference between the empirical data distribution and the model distribution. That we saw in class
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(c)** Show that $\frac{1}{N} \mathcal{L}(\theta)$ has a maximizer that is not unique due to an extra degree of freedom.

    **Solution:**

    Consider shifting all parameters by a constant $c$, let $\theta_{x}' = \theta_{x} + c$.

    $$
    p_{\theta'}(x) = \frac{\exp(\theta_x + c)}{\sum_u \exp(\theta_u + c)} = \frac{e^c \exp(\theta_x)}{e^c \sum_u \exp(\theta_u)} = p_\theta(x)
    $$

    The distribution is invariant to a uniform scalar shift of all $\theta$. As a result, the log-likelihood $\mathcal{L}(\theta)$ takes the exact same value for any $\theta' = \theta + c\mathbf{1}$. If $\theta^*$ maximizes the log-likelihood, then any $\theta^* + c\mathbf{1}$ is equivalently optimal. Hence, the maximizer is not unique.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(d)** Impose the constraint $\sum_{x=1}^m \theta_x = 0$. Under this constraint, solve for the MLE $\hat{\theta}$ in closed form in terms of $\hat{p}_{\mathcal{D}}$.

    **Solution:**

    Setting the gradient to zero to find the maximum yields:

    $$
    \nabla_\theta \mathcal{L}(\theta) = \hat{p}_{\mathcal{D}}(x) - p_{\hat{\theta}}(x) = 0 \implies \hat{p}_{\mathcal{D}}(x) = p_{\hat{\theta}}(x) =\frac{\exp(\hat{\theta}_x)}{Z(\hat{\theta})}
    $$

    Taking the log of both sides gives:

    $$
    \log \hat{p}_{\mathcal{D}}(x) = \hat{\theta}_x - \log Z(\hat{\theta}) \implies \hat{\theta}_x = \log \hat{p}_{\mathcal{D}}(x) + C
    $$

    where $C = \log Z(\hat{\theta})$ is a constant independent of $x$. We find $C$ by applying our constraint $\sum_{x=1}^m \hat{\theta}_x = 0$:

    $$
    0 = \sum_{x=1}^m \hat{\theta}_x = \sum_{x=1}^m \left( \log \hat{p}_{\mathcal{D}}(x) + C \right) = \sum_{x=1}^m \log \hat{p}_{\mathcal{D}}(x) + mC
    $$

    Thus, $C = -\frac{1}{m} \sum_{u=1}^m \log \hat{p}_{\mathcal{D}}(u)$. The unique constrained MLE is:

    $$
    \hat{\theta}_x = \log \hat{p}_{\mathcal{D}}(x) - \frac{1}{m} \sum_{u=1}^m \log \hat{p}_{\mathcal{D}}(u)
    $$
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
    The joint distribution is proportional to the product of factors:

    $$
    p_\theta(x) = \frac{1}{Z(\theta)} \prod_a \exp(\theta_{a, x_{\partial a}})
    $$

    If we shift the parameters $\theta_a$ by constants $c_a$, the new factors correspond to:

    $$
    \exp(\theta_{a, x_{\partial a}} + c_a) = e^{c_a} \exp(\theta_{a, x_{\partial a}})
    $$

    The new unnormalized probability multiplies by an overall constant $\prod_a e^{c_a} = \exp(\sum_a c_a)$. When applying the new normalization factor $Z' = \sum_x \exp(\sum_a c_a) \prod_a f_a(x_{\partial a};\theta_a) = \exp(\sum_a c_a) Z(\theta)$, the extra scaling cancels out entirely in the numerator and denominator. Hence, $p_{\theta'}(x) = p_\theta(x)$.
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

    **Solution:**

    We can shift parameter values between node terms and edge terms without changing the sum. Let $d_i$ be the degree of node $i$. Given arbitrary functions $\phi_i(x_i)$, we can add $\phi_i(x_i)$ to $\theta_i(x_i)$ and perfectly compensate by subtracting off fractional amounts from each incident edge.

    Define the reparameterized potentials:

    $$
    \begin{align*}
    \theta'_i(x_i) &= \theta_i(x_i) + \phi_i(x_i) \\
    \theta'_{ij}(x_i, x_j) &= \theta_{ij}(x_i, x_j) - \frac{\phi_i(x_i)}{d_i} - \frac{\phi_j(x_j)}{d_j}
    \end{align*}
    $$

    Notice that the term $\phi_i(x_i) / d_i$ occurs exactly $d_i$ times (once for every edge incident to $i$). Thus:

    $$
    \sum_{(i,j) \in E} \frac{\phi_i(x_i)}{d_i} + \sum_{(i,j) \in E} \frac{\phi_j(x_j)}{d_j} = \sum_i d_i \frac{\phi_i(x_i)}{d_i} = \sum_i \phi_i(x_i)
    $$

    These terms exactly cancel out, and the total sum remains $\sum_i \theta_i(x_i) + \sum_{(i,j)} \theta_{ij}(x_i, x_j)$, preserving the joint probability perfectly.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(c)** Propose a set of constraints that removes these redundancies (not necessarily uniquely), and briefly argue why it fixes identifiability.

    **Solution:**

    To remove the redundancies, we can fix the degrees of freedom by explicitly anchoring parameters to zero at a reference state. A common choice is to pick an arbitrary reference state for each domain (e.g. state $0 \in \mathcal{X}_i$) and impose "zero-margins":

    1. For the scalar shift redundancy, constrain $\theta_i(0) = 0$ for all $i$.
    2. For the parameter transfer between nodes and edges, constrain $\theta_{ij}(x_i, 0) = 0$ for all $x_i$, and $\theta_{ij}(0, x_j) = 0$ for all $x_j$, for all edges $(i,j) \in E$.

    **Why it fixes identifiability**:
    If $\theta_{ij}(0, x_j) = 0$, then we cannot transfer any parameters from $\theta_j(x_j)$ into $\theta_{ij}(x_i, x_j)$ without making $\theta_{ij}(0, x_j) \neq 0$. This stops the $\phi$-reparameterization freedom. Similarly, fixing $\theta_i(0) = 0$ uniquely resolves the scalar shift (since shifting by $c_a \neq 0$ would immediately violate the $0$ evaluation). This restricts the parameter space completely, leaving a minimal, identifiable representation.
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

    **Solution:**

    Using conditional probability, we divide the joint distribution $p_\theta(y, h)$ by the marginal $p_\theta(y) = \sum_{h'} p_\theta(y, h')$:

    $$
    \begin{align*}
    p_\theta(h | y) &= \frac{p_\theta(y, h)}{\sum_{h'} p_\theta(y, h')} \\
    &= \frac{ \frac{1}{Z} \exp(\theta_{yh} + \alpha_y + \beta_h) }{ \sum_{h'} \frac{1}{Z} \exp(\theta_{yh'} + \alpha_y + \beta_{h'}) } \\
    &= \frac{ \exp(\alpha_y) \exp(\theta_{yh} + \beta_h) }{ \exp(\alpha_y) \sum_{h'} \exp(\theta_{yh'} + \beta_{h'}) } \\
    &= \frac{ \exp(\theta_{yh} + \beta_h) }{ \sum_{h'} \exp(\theta_{yh'} + \beta_{h'}) }
    \end{align*}
    $$

    This is precisely the definition of the softmax function applied to the scores given by $\theta_{yh} + \beta_h$ over the valid clusters $h=1, \dots, K$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(b)** For data $\{y^{(\ell)}\}_{\ell=1}^N$, write the E-step responsibilities and explain their clustering meaning.

    **Solution:**

    In the E-step, the responsibilities are exactly the conditional probabilities from part (a) evaluated on a given observation $y^{(\ell)}$ using current parameters:

    $$
    q^{(\ell)}(h) := p_\theta(h | y = y^{(\ell)}) = \frac{ \exp(\theta_{y^{(\ell)}h} + \beta_h) }{ \sum_{h'} \exp(\theta_{y^{(\ell)}h'} + \beta_{h'}) }
    $$

    **Meaning:** This $q^{(\ell)}(h)$ is the "soft" probability that data point $y^{(\ell)}$ is generated by or belongs to cluster $h$. Instead of hard assigning the point entirely to its most likely cluster, the model proportionally weights its membership in all clusters based on these probabilities, distributing the point probabilistically across latent states.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(c)** Derive the M-step objective and first-order condition in terms of soft counts.

    **Solution:**

    The M-step objective is the expected complete-data log-likelihood, bounding the log-likelihood:

    $$
    \mathcal{Q}(\theta) = \sum_{\ell=1}^N \sum_{h=1}^K q^{(\ell)}(h) \log p_\theta(y^{(\ell)}, h)
    $$

    Substitute the log joint probability $\log p_\theta(y, h) = \theta_{yh} + \alpha_y + \beta_h - \log Z(\theta, \alpha, \beta)$:

    $$
    \mathcal{Q}(\theta) = \sum_{\ell=1}^N \sum_{h=1}^K q^{(\ell)}(h) \left( \theta_{y^{(\ell)}h} + \alpha_{y^{(\ell)}} + \beta_h \right) - N \log Z
    $$

    Defining soft counts $N_{y, h} = \sum_{\ell=1}^N q^{(\ell)}(h) 1\{y^{(\ell)} = y\}$, we rewrite $\mathcal{Q}$:

    $$
    \mathcal{Q}(\theta) = \sum_{y=1}^m \sum_{h=1}^K N_{y, h} (\theta_{yh} + \alpha_y + \beta_h) - N \log Z(\theta, \alpha, \beta)
    $$

    To find the optimal parameter updates via the first-order condition, take the derivative:

    $$
    \frac{\partial \mathcal{Q}}{\partial \theta_{yh}} = N_{y, h} - N \frac{\partial \log Z}{\partial \theta_{yh}} = 0
    $$

    Note that $\frac{\partial \log Z}{\partial \theta_{yh}} = p_\theta(y, h)$. Therefore:

    $$
    N_{y, h} - N p_\theta(y, h) = 0 \implies p_\theta(y, h) = \frac{N_{y, h}}{N}
    $$

    The first-order condition requires that the structural parameters be updated such that the model's new marginal distribution exactly matches the empirical soft-weighted joint distribution given by the responsibilities from the E-step.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(d)** Describe explicit constraints for $(\theta, \alpha, \beta)$ that remove symmetries to make the parameters more identifiable.

    **Solution:**

    There are $m + K + 1$ continuous redundancy degrees of freedom:

    1. **Global constant shift** ($1$ DOF): Adding a constant $c$ to every $\theta_{yh}$ is absorbed by $Z(\theta)$, leaving $p_\theta(y,h)$ unchanged.

    2. **$\alpha$–$\theta$ transfer** ($m$ DOF): For any function $f: \{1,\dots,m\} \to \mathbb{R}$, replacing $\alpha'_y = \alpha_y + f(y)$ and $\theta'_{yh} = \theta_{yh} - f(y)$ preserves the exponent $\theta_{yh} + \alpha_y$.

    3. **$\beta$–$\theta$ transfer** ($K$ DOF): For any function $g: \{1,\dots,K\} \to \mathbb{R}$, replacing $\beta'_h = \beta_h + g(h)$ and $\theta'_{yh} = \theta_{yh} - g(h)$ preserves $\theta_{yh} + \beta_h$.

    **Constraints to fix identifiability** ($m + K + 1$ total):

    $$
    \alpha_y = 0 \;\;\forall\, y, \qquad \beta_h = 0 \;\;\forall\, h, \qquad \sum_{y,h} \theta_{yh} = 0.
    $$

    The first $m$ constraints absorb $\alpha$ into $\theta$, the next $K$ absorb $\beta$ into $\theta$, and the final constraint pins the global scale.

    Additionally, there is a **discrete label-permutation symmetry**: for any permutation $\sigma$ of $\{1, \dots, K\}$, relabeling $H \to \sigma(H)$ (i.e., $\theta'_{y,h} = \theta_{y,\sigma^{-1}(h)}$) yields an equivalent model since $H$ is unobserved. To break this, impose an ordering constraint such as $\theta_{1,1} \leq \theta_{1,2} \leq \cdots \leq \theta_{1,K}$.
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

    **Solution:**

    Starting from the observed-data log-likelihood for a single data point:

    $$
    \nabla_\theta \ln p^\theta(y) = \frac{\nabla_\theta p^\theta(y)}{p^\theta(y)} = \frac{\sum_h \nabla_\theta p^\theta(y, h)}{p^\theta(y)}
    $$

    Rewriting $\nabla_\theta p^\theta(y,h) = p^\theta(y,h)\,\nabla_\theta \ln p^\theta(y,h)$:

    $$
    \nabla_\theta \ln p^\theta(y) = \sum_h \frac{p^\theta(y, h)}{p^\theta(y)} \nabla_\theta \ln p^\theta(y, h) = \mathbb{E}_{p^\theta(H|y)}\!\left[ \nabla_\theta \ln p^\theta(y, H) \right]
    $$

    Summing over all data points:

    $$
    \nabla_\theta L_{\text{obs}}(\theta) = \sum_{\ell=1}^N \mathbb{E}_{p^\theta(H | y^{(\ell)})}\!\left[ \nabla_\theta \ln p^\theta(y^{(\ell)}, H) \right]
    $$

    Now consider the Q-function. Since the expectation is taken under $p^\theta(H|y^{(\ell)})$ (which does not depend on $\theta'$):

    $$
    \nabla_{\theta'} Q(\theta' \mid \theta)\big|_{\theta'=\theta} = \sum_{\ell=1}^N \mathbb{E}_{p^\theta(H | y^{(\ell)})}\!\left[ \nabla_{\theta'} \ln p^{\theta'}(y^{(\ell)}, H) \right]\bigg|_{\theta'=\theta} = \sum_{\ell=1}^N \mathbb{E}_{p^\theta(H | y^{(\ell)})}\!\left[ \nabla_\theta \ln p^\theta(y^{(\ell)}, H) \right]
    $$

    This is identical to $\nabla_\theta L_{\text{obs}}(\theta)$ derived above, completing the proof.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(b)** Conclude that the update $\theta_{\text{new}} = \theta + \eta\,\nabla_{\theta'} Q(\theta' \mid \theta)\big|_{\theta'=\theta}$ with $\eta > 0$ small is a first-order ascent step for $L_{\text{obs}}$.

    **Solution:**

    From part (a), the update direction satisfies:

    $$
    \theta_{\text{new}} = \theta + \eta\,\nabla_{\theta'} Q(\theta' \mid \theta)\big|_{\theta'=\theta} = \theta + \eta\,\nabla_\theta L_{\text{obs}}(\theta)
    $$

    This is exactly gradient ascent on $L_{\text{obs}}$ with step size $\eta$. By a first-order Taylor expansion:

    $$
    L_{\text{obs}}(\theta_{\text{new}}) \approx L_{\text{obs}}(\theta) + \eta\,\|\nabla_\theta L_{\text{obs}}(\theta)\|^2
    $$

    Since $\eta > 0$ and $\|\nabla_\theta L_{\text{obs}}(\theta)\|^2 \geq 0$ (with equality only at a stationary point), we have $L_{\text{obs}}(\theta_{\text{new}}) > L_{\text{obs}}(\theta)$ for sufficiently small $\eta > 0$, provided we are not already at a stationary point. Hence, the small M-step update is a first-order ascent step for the observed-data log-likelihood.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(c)** (Interpretation) Explain in one paragraph why this supports the intuition that "the E-step is exact in the small-change limit."

    **Solution:**

    The E-step computes the posterior $q^{(\ell)}(h) = p^\theta(H=h \mid y^{(\ell)})$ at the current parameter $\theta$, and the resulting Q-function $Q(\theta'|\theta)$ serves as a surrogate for $L_{\text{obs}}$. Part (a) shows that $Q$ and $L_{\text{obs}}$ share exactly the same gradient at $\theta' = \theta$. When the M-step takes only an infinitesimally small step ($\theta_{\text{new}} \approx \theta$), the posterior computed at $\theta$ is still essentially correct for $\theta_{\text{new}}$ because the posterior changes continuously. Re-running the E-step at $\theta_{\text{new}}$ would yield virtually the same responsibilities; the Q-function constructed from the "stale" posterior remains an accurate local surrogate for $L_{\text{obs}}$ near $\theta$. Thus, no approximation error is introduced in the small-change limit — the E-step is exact.
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

    Now, solve the linear program where the constraint $x_{i,j} \in \{0,1\}$ is relaxed to $x_{i,j} \in [0,1]$. This amounts to solving the LP relaxation of the integer problem.

    **Solution:**

    We examine each row's cheapest assignments and search for a conflict-free matching:

    | Task | Cheapest agents (cost) |
    |:----:|:----------------------:|
    | 1 | Agent 4 (6), Agent 2 (7) |
    | 2 | Agent 3 (7), Agent 2/4 (8) |
    | 3 | Agent 3 (12), Agent 4 (15) |
    | 4 | Agent 1 (4), Agent 2 (8) |

    Selecting the  assignment $\sigma = (4, 2, 3, 1)$: Task $1 \to$ Agent $4$, Task $2 \to$ Agent $2$, Task $3 \to$ Agent $3$, Task $4 \to$ Agent $1$:

    $$
    \text{Cost} = 6 + 8 + 12 + 4 = 30
    $$

    No other permutation achieves a lower cost (verified by brute-force enumeration below).

    **LP relaxation:** By the Birkhoff–von Neumann theorem, the set of doubly stochastic matrices equals the a linear combination of the permutation matrices. Since we have chosen the minimum cost permutation, linearly trading off its cost with any other permutation will only increase the cost. Thus, the LP relaxation will also yield the same optimal cost of $30$ with the same assignment.
    """)
    return


@app.cell
def _(mo, np):
    from itertools import permutations as _permutations
    from scipy.optimize import linprog as _linprog

    _C = np.array([[10, 7, 24, 6], [19, 8, 7, 8], [16, 22, 12, 15], [4, 8, 10, 14]])
    _n = 4

    # Brute-force integer program
    _best_cost, _best_perm = min(
        ((sum(_C[_i, _p[_i]] for _i in range(_n)), _p) for _p in _permutations(range(_n))),
        key=lambda x: x[0]
    )

    # LP relaxation
    _Aeq, _beq = [], []
    for _i in range(_n):
        _row = np.zeros(_n**2)
        _row[_i*_n:(_i+1)*_n] = 1
        _Aeq.append(_row); _beq.append(1)
    for _j in range(_n):
        _row = np.zeros(_n**2)
        for _i in range(_n):
            _row[_i*_n + _j] = 1
        _Aeq.append(_row); _beq.append(1)

    _res = _linprog(_C.flatten(), A_eq=np.array(_Aeq), b_eq=np.array(_beq),
                    bounds=[(0, 1)] * _n**2, method='highs')
    _Xlp = np.round(_res.x.reshape(_n, _n), 3)

    _assign = ", ".join(
        "Task " + str(_i+1) + " → Agent " + str(_best_perm[_i]+1)
        for _i in range(_n)
    )
    mo.md(
        "**Verification:**\n\n"
        + "Integer program (brute-force): " + _assign + ", cost = **" + str(_best_cost) + "**\n\n"
        + "LP relaxation: cost = **" + str(int(_res.fun)) + "**, integer-valued solution: " + str(np.allclose(_Xlp, np.round(_Xlp))) + "\n\n"
        + "LP solution matrix:\n```\n" + str(_Xlp) + "\n```"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(b)** Write the permutation constraints as linear equalities for $N = 4$.

    **Solution:**

    Let $x_{i,j,k} \in \{0,1\}$ indicate "row $i$, column $j$ contains value $k$" for $i, j, k \in \{1,2,3,4\}$.

    **Cell constraint** — each cell gets exactly one value (16 constraints):

    $$
    \sum_{k=1}^{4} x_{i,j,k} = 1, \quad \forall\; i, j \in \{1,\dots,4\}
    $$

    **Row constraint** — each value appears once per row (16 constraints):

    $$
    \sum_{j=1}^{4} x_{i,j,k} = 1, \quad \forall\; i, k \in \{1,\dots,4\}
    $$

    **Column constraint** — each value appears once per column (16 constraints):

    $$
    \sum_{i=1}^{4} x_{i,j,k} = 1, \quad \forall\; j, k \in \{1,\dots,4\}
    $$

    **Box constraint** — each value appears once per $2 \times 2$ box (16 constraints). The four boxes are $B_1 = \{1,2\}^2$, $B_2 = \{1,2\} \times \{3,4\}$, $B_3 = \{3,4\} \times \{1,2\}$, $B_4 = \{3,4\}^2$:

    $$
    \sum_{(i,j) \in B_b} x_{i,j,k} = 1, \quad \forall\; b \in \{1,\dots,4\},\; k \in \{1,\dots,4\}
    $$

    Together with bounds $x_{i,j,k} \in [0, 1]$ (LP relaxation) or $x_{i,j,k} \in \{0,1\}$ (ILP), these 64 equality constraints over $N^3 = 64$ variables define the feasible set.
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

    **Solution:**

    We use the cost-function approach: set $c_{i,j,k} = 0$ everywhere, then $c_{i,j,k} = -1$ for each known entry $(i,j,k)$. Minimizing over the LP relaxation with all row, column, cell, and box constraints yields the solution.
    """)
    return


@app.cell
def _(mo, np):
    from scipy.optimize import linprog as _linprog

    _N = 4
    _sqN = 2
    _nv = _N**3

    def _idx(_i, _j, _k):
        return _i * _N**2 + _j * _N + _k

    # Converted to 0-indexed: (row, col, value-1)
    # (row, col): value
    _known_dict = {
        (0, 1): 2,
        (1, 0): 1,
        (1, 2): 2,
        (2, 0): 2,
        (2, 2): 3,
        (3, 1): 3
    }
    _c = np.zeros(_nv)
    _knowns = [(_i, _j, _v-1) for (_i, _j), _v in _known_dict.items()]
    for (_ki, _kj, _kk) in _knowns:
        _c[_idx(_ki, _kj, _kk)] = -1

    _Aeq, _beq = [], []
    # Row constraints: sum_j x_{i,j,k} = 1
    for _i in range(_N):
        for _k in range(_N):
            _row = np.zeros(_nv)
            for _j in range(_N):
                _row[_idx(_i, _j, _k)] = 1
            _Aeq.append(_row); _beq.append(1)
    # Column constraints: sum_i x_{i,j,k} = 1
    for _j in range(_N):
        for _k in range(_N):
            _row = np.zeros(_nv)
            for _i in range(_N):
                _row[_idx(_i, _j, _k)] = 1
            _Aeq.append(_row); _beq.append(1)
    # Cell constraints: sum_k x_{i,j,k} = 1
    for _i in range(_N):
        for _j in range(_N):
            _row = np.zeros(_nv)
            for _k in range(_N):
                _row[_idx(_i, _j, _k)] = 1
            _Aeq.append(_row); _beq.append(1)
    # Box constraints
    for _bi in range(_sqN):
        for _bj in range(_sqN):
            for _k in range(_N):
                _row = np.zeros(_nv)
                for _di in range(_sqN):
                    for _dj in range(_sqN):
                        _row[_idx(_bi*_sqN+_di, _bj*_sqN+_dj, _k)] = 1
                _Aeq.append(_row); _beq.append(1)

    _res = _linprog(_c, A_eq=np.array(_Aeq), b_eq=np.array(_beq),
                    bounds=[(0, 1)] * _nv, method='highs')

    _grid = np.zeros((_N, _N), dtype=int)
    for _i in range(_N):
        for _j in range(_N):
            _vals = [_res.x[_idx(_i, _j, _k)] for _k in range(_N)]
            _grid[_i, _j] = np.argmax(_vals) + 1

    _is_int = np.allclose(_res.x, np.round(_res.x))
    _grid_str = "\n".join("  ".join(str(_v) for _v in _gr) for _gr in _grid)
    mo.md(
        "**LP Solver Result:**\n\n"
        + "Integer-valued solution: " + str(_is_int) + "\n\n"
        + "Solved 4x4 Sudoku:\n```\n" + _grid_str + "\n```"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **(d)** Let $\mathcal{X} = \{1,2,\ldots,N\}$ and $f: \mathcal{X}^N \to \{0,1\}$ be a factor defined by $f(z_1^N) = 1$ if $z_1^N$ defines a permutation of $\mathcal{X}$ and $0$ otherwise. From this, one can construct the Sudoku factor graph by introducing $N^2$ variables and $3N^2$ permutation factor nodes. Local 0/1 factors can be used to account for known values. In this case, marginalization gives

    $$
    g_i(x_i) = \sum_{\bar{x} \setminus x_i} f(\bar{x}) = \#\{\text{valid patterns matching all observations with fixed } x_i\}.
    $$

    Describe how one might implement such a factor-graph permutation constraint with much lower complexity than $N!$. Unfortunately, naïve implementation of the sum-product algorithm for the permutation factor requires the summation of $N!$ terms.

    **Solution:**

    The key idea is to decompose the global permutation factor $f(z_1, \dots, z_N)$ into a **chain of local factors** by introducing auxiliary "state" variables that track which values remain available.

    **Construction:** Introduce auxiliary variables $S_0, S_1, \dots, S_N$ where $S_k \subseteq \{1, \dots, N\}$ represents the set of values not yet assigned after positions $1, \dots, k$:

    - $S_0 = \{1, \dots, N\}$ (all values available).
    - For each $k = 1, \dots, N$, a ternary factor $g_k(S_{k-1}, z_k, S_k)$ enforces:
        1. $z_k \in S_{k-1}$ (assigned value is available), and
        2. $S_k = S_{k-1} \setminus \{z_k\}$ (remove used value).
    - $S_N = \emptyset$ (all values used exactly once).

    The resulting factor graph is a chain:

    $$
    S_0 \;-\; g_1 \;-\; (z_1, S_1) \;-\; g_2 \;-\; (z_2, S_2) \;-\; \cdots \;-\; g_N \;-\; (z_N, S_N)
    $$

    **Complexity:** Each auxiliary variable $S_k$ ranges over subsets of size $N - k$, so $|S_k| = \binom{N}{N-k}$. The sum-product messages along the chain have total complexity:

    $$
    \sum_{k=1}^{N} \binom{N}{k-1} \cdot (N-k+1) = N \cdot 2^{N-1}
    $$

    which is $O(N \cdot 2^N)$, dramatically better than $O(N!)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()

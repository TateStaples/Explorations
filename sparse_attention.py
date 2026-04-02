# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.17.0",
#     "torch>=2.0.0",
#     "plotly>=5.14.0",
#     "numpy>=1.24.0",
#     "entmax>=1.0",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import torch
    import numpy as np
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    from plotly.subplots import make_subplots
    from entmax import sparsemax, entmax15, entmax_bisect

    return entmax15, entmax_bisect, ff, go, make_subplots, np, sparsemax, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualizing Adaptive Sparse Attention Models

    *Sasha Rush ([@srush_nlp](https://twitter.com/srush_nlp)) — HuggingFace Reading Group*

    Based on:
    - **Sparse Sequence-to-Sequence Models** — Peters, Niculae, Martins
      ([arXiv:1905.05702](https://arxiv.org/pdf/1905.05702.pdf))
    - **Adaptively Sparse Transformers** — Correia, Niculae, Martins
      ([arXiv:1909.00015](https://arxiv.org/pdf/1909.00015.pdf))
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goals

    - Learn a transformer model where attention is **more sparse** in its token selection.
    - Do so **without** hard span constraints on distance.
    - Apply an interesting method to a hard task.
    - Hopefully discover concrete patterns in the data, and perhaps improve generalization.

    This notebook focuses on the technical aspects of how this is done. See the papers for full results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Background: Softmax and the Probability Simplex

    Key to understanding attention is to really understand **softmax**. The standard formula is:

    $$\text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

    Softmax maps a score vector $z \in \mathbb{R}^n$ to a probability distribution on the
    **probability simplex** $\Delta^{n-1}$.

    For three tokens A, B, C, the simplex $\Delta^2$ is a **triangle**. Each corner represents
    putting *all* attention on one token; the center represents *uniform* attention over all three.

    The **left** panel shows where the distribution lands on the triangle.
    The **right** panel shows the raw score vector in $\mathbb{R}^3$ and the line connecting
    it to its simplex projection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    z_a = mo.ui.slider(-3.0, 3.0, step=0.1, value=1.5, label=r"$z_A$")
    z_b = mo.ui.slider(-3.0, 3.0, step=0.1, value=0.0, label=r"$z_B$")
    z_c = mo.ui.slider(-3.0, 3.0, step=0.1, value=1.0, label=r"$z_C$")
    mapping_single = mo.ui.dropdown(
        ["softmax", "sparsemax", "entmax-1.5"],
        value="softmax",
        label="Mapping",
    )
    mo.vstack([
        mo.md("**Adjust attention scores for tokens A, B, C and choose a mapping:**"),
        mo.hstack([z_a, z_b, z_c, mapping_single], gap=2),
    ])
    return mapping_single, z_a, z_b, z_c


@app.cell
def _(entmax15, entmax_bisect, go, make_subplots, sparsemax, torch):
    def project(z: "torch.Tensor", mapping: str) -> "torch.Tensor":
        if mapping == "softmax":
            return torch.softmax(z, dim=-1)
        elif mapping == "sparsemax":
            return sparsemax(z, dim=-1)
        elif mapping == "entmax-1.5":
            return entmax15(z, dim=-1)
        else:
            alpha = float(mapping.split("-")[1])
            return entmax_bisect(z, alpha=alpha, dim=-1)

    def draw_projection(points: "torch.Tensor", mapping: str = "softmax") -> "go.Figure":
        raw = project(points, mapping).numpy()
        org = points.numpy()

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scatterternary"}, {"type": "scene"}]],
            subplot_titles=["Probability simplex Δ²", "Score → projection in ℝ³"],
        )

        fig.add_trace(go.Scatterternary(
            mode="markers",
            a=raw[:, 0].tolist(),
            b=raw[:, 1].tolist(),
            c=raw[:, 2].tolist(),
            text=[str(i) for i in range(len(raw))],
            marker=dict(symbol=100, color="#DB7365", size=14, line=dict(width=2)),
        ), row=1, col=1)

        for i in range(len(points)):
            fig.add_trace(go.Scatter3d(
                x=[raw[i, 1], org[i, 1]],
                y=[raw[i, 2], org[i, 2]],
                z=[raw[i, 0], org[i, 0]],
                marker=dict(size=2),
                line=dict(width=2, color="blue"),
                showlegend=False,
            ), row=1, col=2)

        fig.add_trace(go.Scatter3d(
            x=[0, 0, 1, 0],
            y=[0, 1, 0, 0],
            z=[1, 0, 0, 1],
            marker=dict(size=4),
            line=dict(width=5, color="red"),
            name="Simplex face",
        ), row=1, col=2)

        fig.update_layout(height=480, margin=dict(t=50, b=10))
        return fig

    return (draw_projection,)


@app.cell
def _(draw_projection, mapping_single, torch, z_a, z_b, z_c):
    _z = torch.tensor([[z_a.value, z_b.value, z_c.value]])
    draw_projection(_z, mapping_single.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The corners of the triangle represent focusing entirely on one token.
    The center represents spreading attention uniformly across all three.

    **Try `sparsemax`** — the projected point lands on an **edge or corner**,
    meaning at least one token receives *exactly zero* attention weight.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 100 Random Score Vectors

    For fun, here is the same projection applied to 100 random score vectors
    $z \sim \text{Uniform}(-1, 1)^3$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rand_mapping = mo.ui.dropdown(
        ["softmax", "sparsemax", "entmax-1.5"],
        value="softmax",
        label="Mapping",
    )
    rand_mapping
    return (rand_mapping,)


@app.cell
def _(draw_projection, rand_mapping, torch):
    _randos = torch.rand(100, 3) * 2 - 1
    draw_projection(_randos, rand_mapping.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sparsemax

    $$\text{sparsemax}(z) = \arg\min_{\delta \in \Delta} \|\delta - z\|^2$$

    Sparsemax finds the simplex point **closest in Euclidean distance** to the score vector $z$.
    This is just an orthogonal projection onto the simplex — geometrically clear and always sparse.
    Solutions lie on the **edges or corners**, so some tokens always receive exactly zero weight.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entropy on the Simplex

    Another key quantity is **Shannon entropy** $H$, which measures how *uncertain* or
    *spread-out* an attention distribution is:

    $$H(\delta) = -\sum_j \delta_j \log \delta_j$$

    High entropy → distribution is near the **center** (uniform attention).
    Low entropy → distribution is near a **corner** (focused on one token).

    The contour plot below shows entropy at every point of the simplex.
    """)
    return


@app.cell
def _(ff, np, torch):
    _eps = 1e-5
    _X = torch.linspace(0, 1.0, 10).repeat(10)
    _Y = torch.linspace(0, 1.0, 10).repeat(10, 1).t().contiguous().view(-1)
    _keep = _X + _Y <= 1.0
    _X, _Y = _X[_keep], _Y[_keep]
    _Z = 1.0 - _X - _Y

    _entropy = (
        -_X.mul((_X + _eps).log())
        - _Y.mul((_Y + _eps).log())
        - _Z.mul((_Z + _eps).log())
    )


    ff.create_ternary_contour(
        np.stack([_X.numpy(), _Y.numpy(), _Z.numpy()]) + _eps,
        _entropy.numpy(),
        interp_mode="cartesian",
        showscale=True,
        title="Shannon Entropy H(δ) on Δ²",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Softmax as Entropy-Regularized Projection

    The entropy picture reveals *why* softmax never assigns zero weight.
    An equivalent formulation of softmax is:

    $$\text{softmax}(z) = \arg\min_{\delta \in \Delta}\; \delta^\top z - H(\delta)$$

    It minimizes the score alignment term ($\delta^\top z$) while simultaneously
    **maximizing entropy** ($H(\delta)$).  This entropy regularizer always pulls the
    solution away from the boundary.

    **Temperature** $\tau$ controls the trade-off.  Scaling scores by larger $\tau$ drives
    the distribution toward a corner; smaller $\tau$ pushes it toward the center.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    temp_slider = mo.ui.slider(0.1, 4.0, step=0.1, value=1.0, label="Temperature τ")
    temp_slider
    return (temp_slider,)


@app.cell
def _(draw_projection, temp_slider, torch):
    _base = torch.tensor([[1.5, 0., 1.]])
    _scaled = torch.arange(1, 20)[:, None].float() * 0.1 * temp_slider.value * _base
    draw_projection(_scaled, "softmax")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Model: Tsallis Entropy and α-entmax

    The paper generalizes the entropy regularizer using **Tsallis entropy** $H_\alpha^T$:

    $$H_\alpha^T(\delta) = \begin{cases}
    \dfrac{1}{\alpha(\alpha-1)} \displaystyle\sum_j \bigl(\delta_j - \delta_j^\alpha\bigr)
      & \alpha \neq 1 \\[8pt]
    H(\delta) & \alpha = 1
    \end{cases}$$

    Plugging this into the projection objective defines **α-entmax**:

    $$\alpha\text{-entmax}(z) = \arg\min_{\delta \in \Delta}\; \delta^\top z - H_\alpha^T(\delta)$$

    | $\alpha$ | Corresponds to |
    |----------|----------------|
    | 1 | Standard softmax (Shannon entropy regularizer) |
    | 1.5 | entmax-1.5 (special closed-form algorithm) |
    | 2 | Sparsemax (no entropy regularizer, pure Euclidean projection) |

    Different $\alpha$ values exert different "pulls" on the simplex.
    The plots below show how the Tsallis entropy landscape changes with $\alpha$.
    """)
    return


@app.cell
def _(ff, mo, np, torch):
    _eps = 1e-5
    _X = torch.linspace(0, 1.0, 10).repeat(10)
    _Y = torch.linspace(0, 1.0, 10).repeat(10, 1).t().contiguous().view(-1)
    _keep = _X + _Y <= 1.0
    _X, _Y = _X[_keep], _Y[_keep]
    _Z = 1.0 - _X - _Y

    _alpha_figs = []
    for _alpha in [0.5, 1.5, 2.0]:
        _tsallis = (1.0 / (_alpha * (_alpha - 1))) * (
            (_X - _X.pow(_alpha))
            + (_Y - (_Y + _eps).pow(_alpha))
            + (_Z - (_Z + _eps).pow(_alpha))
        )
        _alpha_figs.append(ff.create_ternary_contour(
            np.stack([_X.numpy(), _Y.numpy(), _Z.numpy()]) + _eps,
            _tsallis.numpy(),
            title=f"Tsallis H^T_α,  α = {_alpha}",
            height=380,
            interp_mode="cartesian",
            showscale=True,
        ))

    mo.hstack([mo.as_html(_alpha_figs[0]), mo.as_html(_alpha_figs[1]), mo.as_html(_alpha_figs[2])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the entropy landscapes:**

    - **α < 1** — entropy is concentrated *near the center*; the regularizer strongly prefers uniform distributions.
    - **α = 1** — standard Shannon entropy; the familiar softmax regularizer.
    - **α > 1** — entropy grows toward the *edges and corners*; the regularizer allows or even favors sparse solutions.
    - **α = 2** — entropy is maximized exactly at the corners, allowing sparsemax's zero-weight solutions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive α-entmax: Interpolating from Softmax to Sparsemax

    Adjust $\alpha$ below to watch how the simplex trajectory changes as a fixed score vector
    is scaled by increasing temperature values.  Notice when the path first reaches the **edges**
    (partial sparsity) and when it reaches a **corner** (full sparsity).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    alpha_slider = mo.ui.slider(1.0, 2.0, step=0.05, value=1.5, label="α")
    alpha_slider
    return (alpha_slider,)


@app.cell
def _(
    alpha_slider,
    entmax15,
    entmax_bisect,
    go,
    make_subplots,
    sparsemax,
    torch,
):
    def _project_alpha(z: "torch.Tensor", alpha: float) -> "torch.Tensor":
        if abs(alpha - 1.0) < 0.03:
            return torch.softmax(z, dim=-1)
        elif abs(alpha - 1.5) < 0.03:
            return entmax15(z, dim=-1)
        elif abs(alpha - 2.0) < 0.03:
            return sparsemax(z, dim=-1)
        else:
            return entmax_bisect(z, alpha=alpha, dim=-1)

    _base = torch.tensor([[1.5, 0., 1.]])
    _zs = torch.arange(1, 20)[:, None].float() * 0.1 * _base
    _alpha_val = alpha_slider.value
    _raw = _project_alpha(_zs, _alpha_val).numpy()
    _org = _zs.numpy()

    _fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatterternary"}, {"type": "scene"}]],
        subplot_titles=[f"α-entmax  (α = {_alpha_val:.2f})", "Score → projection"],
    )
    _fig.add_trace(go.Scatterternary(
        mode="markers",
        a=_raw[:, 0].tolist(),
        b=_raw[:, 1].tolist(),
        c=_raw[:, 2].tolist(),
        text=[str(i) for i in range(len(_raw))],
        marker=dict(symbol=100, color="#DB7365", size=12, line=dict(width=2)),
    ), row=1, col=1)
    for _i in range(len(_zs)):
        _fig.add_trace(go.Scatter3d(
            x=[_raw[_i, 1], _org[_i, 1]],
            y=[_raw[_i, 2], _org[_i, 2]],
            z=[_raw[_i, 0], _org[_i, 0]],
            marker=dict(size=2),
            line=dict(width=2, color="blue"),
            showlegend=False,
        ), row=1, col=2)
    _fig.add_trace(go.Scatter3d(
        x=[0, 0, 1, 0], y=[0, 1, 0, 0], z=[1, 0, 0, 1],
        marker=dict(size=4), line=dict(width=5, color="red"), name="Simplex",
    ), row=1, col=2)
    _fig.update_layout(height=500, margin=dict(t=50, b=10))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Key observations:**

    - **softmax (α = 1):** The trajectory is a smooth curve that *never* touches an edge or corner,
      no matter how extreme the scores.
    - **entmax-1.5 (α = 1.5):** The trajectory can reach **edges** (one token gets zero weight)
      but does so gradually — intermediate sparsity.
    - **sparsemax (α = 2):** The trajectory snaps to the **nearest face or corner** of the simplex.
      At high scores, exactly one token receives all the attention.

    A crisp visual way to see this is in the side-by-side comparison from the original paper
    (shown for the 2D, 2-token case):

    - Softmax traces a smooth sigmoid-like curve that asymptotes toward 0 and 1 but never reaches them.
    - Sparsemax is piecewise linear — it saturates at 0 and 1 at finite score differences.
    - entmax-1.5 is in between: it reaches 0/1 at finite scores but with a smoother path.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two-Token Case: Side-by-Side Comparison

    For two tokens with scores $(z, -z)$, we can plot the attention weight on token A
    as a function of the score difference $z$.  This is the 1D analogue of the simplex diagrams.
    """)
    return


@app.cell
def _(entmax15, go, sparsemax, torch):
    _z_vals = torch.linspace(-4, 4, 200)
    _pairs = torch.stack([_z_vals, -_z_vals], dim=-1)

    _soft_p  = torch.softmax(_pairs, dim=-1)[:, 0].numpy()
    _sparse_p = sparsemax(_pairs, dim=-1)[:, 0].numpy()
    _ent15_p  = entmax15(_pairs, dim=-1)[:, 0].numpy()

    _zn = _z_vals.numpy()

    _fig2 = go.Figure()
    _fig2.add_trace(go.Scatter(x=_zn, y=_soft_p,   name="softmax (α=1)",   line=dict(width=2.5)))
    _fig2.add_trace(go.Scatter(x=_zn, y=_ent15_p,  name="entmax-1.5 (α=1.5)", line=dict(width=2.5, dash="dash")))
    _fig2.add_trace(go.Scatter(x=_zn, y=_sparse_p, name="sparsemax (α=2)", line=dict(width=2.5, dash="dot")))
    _fig2.update_layout(
        title="Attention weight on token A vs. score difference z",
        xaxis_title="Score z  (scores are [z, −z])",
        yaxis_title="p(A)",
        height=380,
        legend=dict(x=0.05, y=0.95),
    )
    _fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extensions in the Paper

    The paper introduces several key components to deploy α-entmax in full transformer models:

    1. **Efficient algorithms** — Computing α-entmax exactly in $\mathcal{O}(n \log n)$ time
       (for α = 1.5 there is a special-purpose closed-form algorithm).
    2. **Per-head α** — Different attention heads can use different fixed sparsity levels,
       allowing the model to select the appropriate inductive bias per head.
    3. **Learned α** — $\alpha$ is a scalar parameter optimized end-to-end via backpropagation.
       Each head *self-selects* its own sparsity — the model decides how sparse it needs to be.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiments

    Three models compared on machine translation (WMT EN→DE):

    | Model | Description |
    |-------|-------------|
    | **softmax** | Standard Transformer |
    | **1.5-entmax** | Transformer with α = 1.5 (fast special-case algorithm) |
    | **α-entmax** | Transformer with a *different learned* α per head and layer |

    **Computational overhead is small:**
    α-entmax trains at **75%** the speed of softmax; 1.5-entmax at **90%** — a modest price
    for structured sparsity.

    **Interpretability findings from inspecting learned attention heads:**

    - A **BPE-merging head** learns α ≈ 1.91 (near sparsemax).
      The head can afford to be very sparse because it is almost always just merging the next local subword piece.
    - The α-entmax model picks some heads to be **very dense** (α ≈ 1, attending broadly),
      while 1.5-entmax has far fewer such heads.
    - An **interrogation-finding head** learns to search for the `?` token to
      distinguish interrogative clauses from relative clauses — a linguistically meaningful sparse pattern.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### α Distribution Across Heads (Schematic)

    The paper reports that learned α values cluster around interpretable points.
    Below is a schematic reproduction of the distribution of learned α values
    across all heads, as described qualitatively in the paper.
    """)
    return


@app.cell
def _(go, np):
    _rng = np.random.default_rng(42)
    _alphas = np.concatenate([
        _rng.normal(1.05, 0.05, 8),
        _rng.normal(1.5,  0.15, 28),
        _rng.normal(1.9,  0.08, 12),
    ])
    _alphas = np.clip(_alphas, 1.0, 2.0)

    _fig3 = go.Figure()
    _fig3.add_trace(go.Histogram(
        x=_alphas,
        nbinsx=20,
        marker_color="#5577AA",
        opacity=0.8,
        name="learned α",
    ))
    _fig3.add_vline(x=1.0, line_dash="dot",  line_color="green",  annotation_text="softmax (α=1)")
    _fig3.add_vline(x=1.5, line_dash="dash", line_color="orange", annotation_text="entmax-1.5")
    _fig3.add_vline(x=2.0, line_dash="dot",  line_color="red",    annotation_text="sparsemax (α=2)")
    _fig3.update_layout(
        title="Schematic distribution of learned α values across attention heads",
        xaxis_title="α",
        yaxis_title="Number of heads",
        height=360,
    )
    _fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    - **α-entmax** is a principled generalization of softmax producing **partial or full sparsity**.
    - When α is **learned**, each attention head self-selects its sparsity level — the model
      decides how focused it needs to be.
    - The method is **not much slower** than softmax (75–90% speed) and reliably yields sparse outputs.
    - Sparse attention heads are more **interpretable**: zero-weight tokens are explicitly excluded.
    - The Tsallis entropy family provides a clean mathematical bridge:
      $\alpha = 1$ recovers softmax, $\alpha = 2$ recovers sparsemax, and everything in between
      is a tunable knob on the sparsity–density trade-off.
    """)
    return


if __name__ == "__main__":
    app.run()

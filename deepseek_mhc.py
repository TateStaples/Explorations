# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==6.0.0",
#     "marimo",
#     "numpy==2.4.2",
#     "pandas==3.0.1",
#     "pyarrow==23.0.1",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", app_title="deepseek_mhc")


@app.cell
def _():
    import sys
    from pathlib import Path

    _python_dir = Path(__file__).parent / "mhc-visualizer" / "python"
    if _python_dir.exists() and str(_python_dir) not in sys.path:
        sys.path.insert(0, str(_python_dir))

    import numpy as np
    import altair as alt
    import pandas as pd
    import marimo as mo

    from mhc import run_comparison, sinkhorn_knopp, compute_all_metrics

    return alt, mo, np, pd, run_comparison, sinkhorn_knopp


@app.cell
def _(mo):
    mo.md(r"""
    # The Manifold Dial: Visualizing Why DeepSeek's mHC Stabilizes Deep Networks

    Interactive Demo: Explore how mHC stabilizes deep networks with the [Manifold Dial visualizer](https://subhadipmitra.com/mhc-visualizer/) ↗
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Nine Years of “Good Enough”
    Residual connections haven’t changed since 2016. He et al. introduced them in ResNet, the formula stuck (output = layer(x) + x), and we’ve been using the same thing ever since. Attention mechanisms evolved. Normalization techniques multiplied. FFN architectures got reworked a dozen times. But skip connections? Untouched.

    ```python
    output = layer(x) + x
    ```

    It’s not that nobody tried. There’s been work on dense connections, highway networks, various gating mechanisms. Most added complexity without clear wins. The simple additive skip connection kept winning.
    Then Hyper-Connections came along and showed genuine improvements by expanding the residual stream - multiple parallel paths instead of one, with learned mixing between them. Promising results. But also a problem that becomes obvious only at scale: the networks become unstable during training. Loss spikes. Gradient explosions. The deeper you go, the worse it gets.
    DeepSeek’s mHC paper explains why this happens and how to fix it. The fix involves projecting matrices onto something called the Birkhoff polytope using an algorithm from 1967. I built an interactive tool to visualize what’s actually going on, because the equations alone don’t convey how dramatic the difference is.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What Hyper-Connections Actually Do
    Standard residual: you compute a layer’s output and add back the input. One stream in, one stream out.
    Hyper-Connections expand this to $n$ parallel streams (typically 4). Instead of simple addition, you get learned mixing matrices that control how information flows between streams:
    Three matrices per layer: one to mix the residual streams ($H^{res}$), one to aggregate streams into the layer input ($H^{pre}$), one to distribute the layer output back to streams ($H^{post}$).
    The paper’s ablation study shows $H^{res}$ matters most. That’s the mixing within the residual stream itself - how information from different streams combines as it flows through the network.
    More expressivity should mean better performance, and it does. HC improves over standard residuals in their experiments. The catch is what happens when you stack 60+ layers.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Composite Mapping Problem
    Each layer multiplies by its $H^{res}$ matrix. Through $L$ layers, the effective transformation is:
    This product determines how signals from early layers reach later ones. With unconstrained learned matrices, small amplifications compound. A matrix with spectral norm 1.05 seems harmless. Sixty of them multiplied together? That’s $1.05^{60} \approx 18$. And real HC matrices aren’t limited to 1.05.
    The paper measured this directly. Figure 3 shows the “Amax Gain Magnitude” - essentially the worst-case amplification through the composite mapping. For HC at depth 64, gains can reach 10³ to 10⁵ depending on initialization. In our toy simulation with random matrices, it’s even more extreme - up to 10¹⁶. The composite mapping amplifies signals catastrophically.
    That’s why training becomes unstable. Gradients flow backward through the same composite mapping. A 3000x amplification in the forward pass means 3000x amplification in the backward pass. Gradient clipping helps, but you’re fighting the architecture itself.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Fix: Doubly Stochastic Matrices
    mHC constrains $H^{res}$ to be doubly stochastic - all entries non-negative, all rows sum to 1, all columns sum to 1.
    Why this specific constraint? Three properties matter:
    Spectral norm is bounded by 1. A doubly stochastic matrix cannot amplify signals. Each row summing to 1 means the weighted combination of inputs never exceeds the maximum input. No amplification, no explosion.
    Closure under multiplication. Multiply two doubly stochastic matrices and you get another doubly stochastic matrix. This is the key insight. It doesn’t matter how many layers you stack - the composite mapping stays doubly stochastic, stays bounded.
    Geometric interpretation. The set of doubly stochastic matrices forms the Birkhoff polytope, which is the convex hull of permutation matrices. Every doubly stochastic matrix can be written as a weighted average of permutations. Permutations just shuffle; they don’t amplify. Weighted averages of shuffles don’t amplify either.
    The result: composite gains stay near 1 regardless of depth. The paper shows mHC at depth 64 has composite gain around 1.6. Compare that to HC’s explosive growth.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Sinkhorn-Knopp: 1967 Meets 2025
    To make a learned matrix doubly stochastic, mHC uses the Sinkhorn-Knopp algorithm. Published in 1967 for balancing matrices in numerical analysis, it turns out to be exactly what’s needed here.
    The algorithm is simple: exponentiate entries to make them positive, then alternate between normalizing rows and normalizing columns. Repeat until convergence. The iteration provably converges to a doubly stochastic matrix.
    """)
    return


@app.cell
def _(np):
    def demo_sinkhorn_knopp(matrix, iterations=20, eps=1e-8):
        # Exponentiate (subtract max for numerical stability)
        P = np.exp(matrix - matrix.max())
        for _ in range(iterations):
            P = P / (P.sum(axis=1, keepdims=True) + eps)  # Row normalize
            P = P / (P.sum(axis=0, keepdims=True) + eps)  # Column normalize
        return P

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Controls

    Adjust the parameters below to see how they affect signal propagation stability.
    """)
    return


@app.cell(hide_code=True)
def _(active_config, mo):
    sinkhorn_slider = mo.ui.slider(
        start=0,
        stop=30,
        value=active_config["k"],
        step=1,
        label="Sinkhorn Iterations (k)",
        show_value=True,
    )
    depth_slider = mo.ui.slider(
        start=10,
        stop=200,
        value=active_config["depth"],
        step=10,
        label="Network Depth",
        show_value=True,
    )
    n_dropdown = mo.ui.dropdown(
        options={"2": 2, "4": 4, "8": 8},
        value=active_config["n"],
        label="Number of Streams (n)",
    )
    seed_input = mo.ui.number(
        value=active_config["seed"], start=0, stop=10000, label="Random Seed"
    )

    controls = mo.hstack(
        [sinkhorn_slider, depth_slider, n_dropdown, seed_input], justify="start", gap=2
    )
    controls
    return depth_slider, n_dropdown, seed_input, sinkhorn_slider


@app.cell(hide_code=True)
def _(mo):
    preset_default = mo.ui.run_button(label="Default")
    preset_explosion = mo.ui.run_button(label="HC Explosion (k=0)")
    preset_minimal = mo.ui.run_button(label="Minimal Projection (k=5)")
    preset_deep = mo.ui.run_button(label="Deep Network (200)")
    randomize_btn = mo.ui.run_button(label="Randomize Seed")

    presets = mo.hstack(
        [preset_default, preset_explosion, preset_minimal, preset_deep, randomize_btn],
        justify="start",
        gap=1,
    )
    presets
    return preset_deep, preset_explosion, preset_minimal, randomize_btn


@app.cell
def _(depth_slider, n_dropdown, run_comparison, seed_input, sinkhorn_slider):
    results = run_comparison(
        depth=depth_slider.value,
        n=int(n_dropdown.value),
        sinkhorn_iters=sinkhorn_slider.value,
        seed=seed_input.value,
    )
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Signal Propagation: The Explosion Problem

    The real issue isn't single-layer behavior - it's what happens when we **compose many layers**.

    In a deep network, the effective transformation is:
    $$H_{composite} = H_L \cdot H_{L-1} \cdot ... \cdot H_1$$

    Watch the chart below: **HC (red) explodes exponentially, while mHC (blue) stays bounded!**
    """)
    return


@app.cell
def _(base_chart, mo):
    mo.ui.altair_chart(base_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Eigenvalue Decay: The Convergence Story

    The commenter on Reddit raised an excellent point: while doubly stochastic matrices keep gains bounded,
    their eigenvalues (except λ₁=1) get pushed toward zero with each multiplication.

    This means the composite matrix converges toward the **uniform averaging matrix** (all entries = 1/n).

    The chart below shows |λ₂| (second-largest eigenvalue magnitude) - this controls the rate of convergence:
    - **Baseline (Identity)**: |λ₂| = 1 forever (no mixing)
    - **HC**: Eigenvalues can grow unboundedly
    - **mHC**: |λ₂| < 1, decays toward 0 (gradual averaging)

    This is the fundamental tradeoff: **bounded gains** (stable training) vs **information mixing** (convergence to uniformity).
    """)
    return


@app.cell
def _(eigenvalue_chart, mo):
    mo.ui.altair_chart(eigenvalue_chart)
    return


@app.cell
def _(alt, depth_slider, pd, results):
    eig_data = []
    for _method in ["baseline", "hc", "mhc"]:
        for _i, _m in enumerate(results[_method]["composite"]):
            eig_data.append(
                {
                    "layer": _i + 1,
                    "eigenvalue": _m.get("second_eigenvalue_mag", 0),
                    "method": _method.upper() if _method != "mhc" else "mHC",
                }
            )
    eig_df = pd.DataFrame(eig_data)

    eigenvalue_chart = (
        alt.Chart(eig_df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(
                "layer:Q",
                title="Layer Depth",
                scale=alt.Scale(domain=[1, depth_slider.value]),
            ),
            y=alt.Y(
                "eigenvalue:Q",
                scale=alt.Scale(type="log", domain=[1e-16, 1e10]),
                title="|λ₂| Second Eigenvalue (log scale)",
            ),
            color=alt.Color(
                "method:N",
                scale=alt.Scale(
                    domain=["BASELINE", "HC", "mHC"],
                    range=["#10b981", "#ef4444", "#3b82f6"],
                ),
                legend=alt.Legend(title="Method"),
            ),
        )
        .properties(
            width=700,
            height=300,
            title="Eigenvalue Decay: Convergence to Uniform Matrix",
        )
    )
    return (eigenvalue_chart,)


@app.cell(hide_code=True)
def _(layer_selector, mo, results):
    layer_idx = layer_selector.value - 1
    baseline_m = (
        results["baseline"]["composite"][layer_idx]
        if layer_idx < len(results["baseline"]["composite"])
        else {}
    )
    hc_m = (
        results["hc"]["composite"][layer_idx]
        if layer_idx < len(results["hc"]["composite"])
        else {}
    )
    mhc_m = (
        results["mhc"]["composite"][layer_idx]
        if layer_idx < len(results["mhc"]["composite"])
        else {}
    )

    def _format_gain(g):
        if g is None:
            return "N/A"
        if g > 1000:
            return f"{g:.2e}"
        return f"{g:.4f}"

    def _format_dev(g):
        if g is None:
            return "N/A"
        return f"{g:.2e}"

    metrics_md = mo.md(f"""
    ### Stability Metrics at Layer {layer_selector.value}

    | Metric | Baseline | HC (Unconstrained) | mHC (Sinkhorn) |
    |--------|----------|-------------------|----------------|
    | Forward Gain | {_format_gain(baseline_m.get("forward_gain"))} | {_format_gain(hc_m.get("forward_gain"))} | {_format_gain(mhc_m.get("forward_gain"))} |
    | Backward Gain | {_format_gain(baseline_m.get("backward_gain"))} | {_format_gain(hc_m.get("backward_gain"))} | {_format_gain(mhc_m.get("backward_gain"))} |
    | Spectral Norm | {_format_gain(baseline_m.get("spectral_norm"))} | {_format_gain(hc_m.get("spectral_norm"))} | {_format_gain(mhc_m.get("spectral_norm"))} |
    | \\|λ₁\\| (largest) | {_format_gain(baseline_m.get("largest_eigenvalue_mag"))} | {_format_gain(hc_m.get("largest_eigenvalue_mag"))} | {_format_gain(mhc_m.get("largest_eigenvalue_mag"))} |
    | \\|λ₂\\| (2nd) | {_format_gain(baseline_m.get("second_eigenvalue_mag"))} | {_format_gain(hc_m.get("second_eigenvalue_mag"))} | {_format_gain(mhc_m.get("second_eigenvalue_mag"))} |
    | Row Sum Dev | {_format_dev(baseline_m.get("row_sum_max_dev"))} | {_format_dev(hc_m.get("row_sum_max_dev"))} | {_format_dev(mhc_m.get("row_sum_max_dev"))} |
    | Col Sum Dev | {_format_dev(baseline_m.get("col_sum_max_dev"))} | {_format_dev(hc_m.get("col_sum_max_dev"))} | {_format_dev(mhc_m.get("col_sum_max_dev"))} |
    """)

    metrics_md
    return


@app.cell
def _(heatmaps):
    heatmaps
    return


@app.cell(hide_code=True)
def _(hc_sample, mhc_sample, mo, np):
    hc_row_sums = hc_sample.sum(axis=1) if hc_sample is not None else np.zeros(1)
    hc_col_sums = hc_sample.sum(axis=0) if hc_sample is not None else np.zeros(1)
    mhc_row_sums = mhc_sample.sum(axis=1) if mhc_sample is not None else np.zeros(1)
    mhc_col_sums = mhc_sample.sum(axis=0) if mhc_sample is not None else np.zeros(1)

    sums_md = mo.md(f"""
    **Row/Column Sums:**

    | | HC | mHC |
    |---|---|---|
    | Row Sums | {np.array2string(hc_row_sums, precision=2)} | {np.array2string(mhc_row_sums, precision=3)} |
    | Col Sums | {np.array2string(hc_col_sums, precision=2)} | {np.array2string(mhc_col_sums, precision=3)} |
    | Max Row Dev from 1 | {np.abs(hc_row_sums - 1).max():.4f} | {np.abs(mhc_row_sums - 1).max():.2e} |
    | Max Col Dev from 1 | {np.abs(hc_col_sums - 1).max():.4f} | {np.abs(mhc_col_sums - 1).max():.2e} |

    Notice how mHC row/column sums are all ~1.0 (doubly stochastic)!
    """)
    sums_md
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Manifold Dial: Varying Sinkhorn Iterations

    Watch how the matrix transforms as we increase iterations:
    - **k=0**: Raw random matrix (same as HC) - unconstrained
    - **k=1-5**: Partial projection, rapid stabilization
    - **k=10-20**: Fully doubly stochastic
    """)
    return


@app.cell(hide_code=True)
def _(alt, np, pd, seed_input, sinkhorn_knopp):
    iters_to_show = [0, 1, 2, 5, 10, 20]
    rng_dial = np.random.default_rng(seed_input.value)
    dial_base = rng_dial.standard_normal((4, 4))

    dial_data = []
    for k in iters_to_show:
        if k == 0:
            mat = dial_base
        else:
            mat = sinkhorn_knopp(dial_base, iterations=k)

        for _i in range(4):
            for _j in range(4):
                dial_data.append(
                    {
                        "row": str(_i),
                        "col": str(_j),
                        "value": float(mat[_i, _j]),
                        "k": f"k={k}",
                    }
                )

    dial_df = pd.DataFrame(dial_data)

    dial_chart = (
        alt.Chart(dial_df)
        .mark_rect()
        .encode(
            x=alt.X("col:O", title=None, axis=alt.Axis(labels=False)),
            y=alt.Y("row:O", title=None, axis=alt.Axis(labels=False)),
            color=alt.Color(
                "value:Q", scale=alt.Scale(scheme="blues", domain=[0, 0.6]), legend=None
            ),
            tooltip=["row", "col", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(width=100, height=100)
        .facet(
            column=alt.Column(
                "k:N",
                title="Sinkhorn Iterations",
                sort=[f"k={k}" for k in iters_to_show],
            )
        )
        .properties(
            title="The Manifold Dial: Sinkhorn Iterations Transform Random to Doubly Stochastic"
        )
    )

    dial_chart
    return


@app.cell
def _(alt, depth_slider, mo, pd, results):
    data = []
    for _method in ["baseline", "hc", "mhc"]:
        for _i, _m in enumerate(results[_method]["composite"]):
            data.append(
                {
                    "layer": _i + 1,
                    "gain": _m.get("forward_gain", 1),
                    "method": _method.upper() if _method != "mhc" else "mHC",
                }
            )
    df = pd.DataFrame(data)

    layer_selector = mo.ui.slider(
        start=1,
        stop=depth_slider.value,
        value=depth_slider.value,
        label="Inspect Layer",
        show_value=True,
    )

    base_chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(
                "layer:Q",
                title="Layer Depth",
                scale=alt.Scale(domain=[1, depth_slider.value]),
            ),
            y=alt.Y(
                "gain:Q",
                scale=alt.Scale(type="log"),
                title="Composite Forward Gain (log scale)",
            ),
            color=alt.Color(
                "method:N",
                scale=alt.Scale(
                    domain=["BASELINE", "HC", "mHC"],
                    range=["#10b981", "#ef4444", "#3b82f6"],
                ),
                legend=alt.Legend(title="Method"),
            ),
            strokeDash=alt.StrokeDash(
                "method:N",
                scale=alt.Scale(
                    domain=["BASELINE", "HC", "mHC"], range=[[1, 0], [1, 0], [1, 0]]
                ),
                legend=None,
            ),
        )
        .properties(
            width=700,
            height=400,
            title="The Manifold Dial: HC Explosion vs mHC Stability",
        )
    )
    return base_chart, layer_selector


@app.cell
def _(alt, n_dropdown, np, pd, seed_input, sinkhorn_knopp, sinkhorn_slider):
    n_val = int(n_dropdown.value)
    rng = np.random.default_rng(seed_input.value)

    base_matrix = rng.standard_normal((n_val, n_val))
    hc_sample = base_matrix
    mhc_sample = sinkhorn_knopp(base_matrix, iterations=sinkhorn_slider.value)

    def matrix_to_df(mat, name):
        rows = []
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                rows.append(
                    {
                        "row": str(i),
                        "col": str(j),
                        "value": float(mat[i, j]),
                        "type": name,
                    }
                )
        return pd.DataFrame(rows)

    hc_df = matrix_to_df(hc_sample, "HC")
    mhc_df = matrix_to_df(mhc_sample, "mHC")

    hc_heatmap = (
        alt.Chart(hc_df)
        .mark_rect()
        .encode(
            x=alt.X("col:O", title="Column", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("row:O", title="Row"),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="redblue", domain=[-2, 2]),
                legend=alt.Legend(title="Value"),
            ),
            tooltip=["row", "col", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(width=180, height=180, title="HC (Random)")
    )

    mhc_heatmap = (
        alt.Chart(mhc_df)
        .mark_rect()
        .encode(
            x=alt.X("col:O", title="Column", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("row:O", title="Row"),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="blues", domain=[0, 0.6]),
                legend=alt.Legend(title="Value"),
            ),
            tooltip=["row", "col", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(width=180, height=180, title=f"mHC (k={sinkhorn_slider.value})")
    )

    heatmaps = hc_heatmap | mhc_heatmap
    return hc_sample, heatmaps, mhc_sample


@app.cell(hide_code=True)
def _(np, preset_deep, preset_explosion, preset_minimal, randomize_btn):
    if preset_explosion.value:
        active_config = {"k": 0, "depth": 64, "n": "4", "seed": 42}
    elif preset_minimal.value:
        active_config = {"k": 5, "depth": 64, "n": "4", "seed": 42}
    elif preset_deep.value:
        active_config = {"k": 20, "depth": 200, "n": "4", "seed": 42}
    elif randomize_btn.value:
        active_config = {
            "k": 20,
            "depth": 64,
            "n": "4",
            "seed": int(np.random.randint(0, 10000)),
        }
    else:
        active_config = {"k": 20, "depth": 64, "n": "4", "seed": 42}
    return (active_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Community Discussion: Convergence to Uniformity

    A great question was raised in a discussion regarding the math behind this:

    > **Question:** "Something I was interested in was seeing what multiplying 64 of these matrices look like? On the one hand, they are closed under multiplication, but the eigenvalues are systematically pushed down to 0 except one that is by construction always equal to 1. So do the last layers necessarily only have the average of the first layer? (the matrix with 1/N on each entry)"

    > **Subhadip Mitra:** "Products of doubly stochastic matrices do converge to the uniform 1/N matrix. eigenvalue 1 sticks around (eigenvector = all ones) but everything else has |λ| < 1 so it decays toward zero. classic ergodicity stuff
    >
    > but a few things worth noting:
    >
    > 1. rate of convergence matters
    > at 64 layers with random DS matrices we're not actually AT the uniform matrix yet - somewhere in between. depends on the spectral gap (second largest eigenvalue).
    >
    > 2. the real mHC arch isnt pure matrix multiplication
    > Looking at the pytorch impl, the actual update is more like:
    > `output = x + α * (H @ x) + layer_output`
    > so you've got:
    > - residual identity term (`+ x`) preserving original signal
    > - tiny $α$ (init at 0.01) so its basically identity + small perturbation
    > - fresh info injection every layer from the actual layer computation
    > even if the $H$ products trend toward averaging, new signal keeps getting added"

    Let's visualize exactly what the composite matrix looks like over 64 layers to see this convergence to $1/N$ in action!
    """)
    return


@app.cell
def _(
    depth_slider,
    mo,
    n_dropdown,
    np,
    seed_input,
    sinkhorn_knopp,
    sinkhorn_slider,
):
    n_val_conv = int(n_dropdown.value)
    rng_conv = np.random.default_rng(seed_input.value)

    depth_conv = depth_slider.value
    conv_matrices = []

    for _ in range(depth_conv):
        base_mat = rng_conv.standard_normal((n_val_conv, n_val_conv))
        conv_matrices.append(sinkhorn_knopp(base_mat, iterations=sinkhorn_slider.value))

    composite_conv = np.eye(n_val_conv)
    progression = []
    for i in range(depth_conv):
        composite_conv = composite_conv @ conv_matrices[i]
        progression.append(composite_conv.copy())

    layer_slider_conv = mo.ui.slider(
        1,
        depth_conv,
        value=1,
        label="Layer for Composite Visualization",
        show_value=True,
    )
    return layer_slider_conv, n_val_conv, progression


@app.cell
def _(alt, layer_slider_conv, mo, n_val_conv, pd, progression):
    curr_mat = progression[layer_slider_conv.value - 1]

    p_rows = []
    for _i in range(curr_mat.shape[0]):
        for _j in range(curr_mat.shape[1]):
            p_rows.append(
                {"row": str(_i), "col": str(_j), "value": float(curr_mat[_i, _j])}
            )
    df_conv = pd.DataFrame(p_rows)

    conv_heatmap = (
        alt.Chart(df_conv)
        .mark_rect()
        .encode(
            x=alt.X("col:O", title="Column", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("row:O", title="Row"),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="blues", domain=[0, 1]),
                legend=alt.Legend(title="Probability"),
            ),
            tooltip=["row", "col", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(
            width=300,
            height=300,
            title=f"mHC Composite Product at Layer {layer_slider_conv.value} (Target is 1/N = {1 / n_val_conv:.3f})",
        )
    )

    conv_ui = mo.vstack([layer_slider_conv, mo.ui.altair_chart(conv_heatmap)])
    conv_ui
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What I Find Interesting
    A few things stood out reading this paper:
    The instability isn’t subtle. Three orders of magnitude in signal amplification isn’t a minor numerical issue you can tune away. It’s a fundamental architectural problem. HC was probably hitting this wall in ways that weren’t always diagnosed correctly.
    The fix comes from constraints, not regularization. You could try to penalize large gains with loss terms, but that’s fighting the architecture. Constraining to doubly stochastic matrices makes explosion structurally impossible. The geometry of the constraint does the work.
    The 1967 algorithm works. Machine learning keeps rediscovering techniques from optimization and numerical analysis. Sinkhorn-Knopp wasn’t designed for neural networks, but it slots in perfectly here. There’s probably more useful machinery sitting in old papers.
    Macro-architecture gets less attention than it deserves. We spend enormous effort on attention variants and FFN structures, but how layers connect to each other - the topology of the network - might have similar headroom for improvement.

    ## Code
    I implemented both the visualization and a PyTorch module you can actually use:
    """)
    return


@app.cell
def _():
    # from mhc import mHCResidual

    # Drop-in residual connection replacement
    # residual = mHCResidual(dim=512, n_streams=4, sinkhorn_iters=20)

    # In your forward pass
    # hidden = residual(hidden_states, layer_output)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The repository includes the interactive demo source, Python implementation with tests, and a Colab notebook if you want to experiment without local setup.

    ## Links
    - [Interactive Demo](https://subhadipmitra.com/mhc-visualizer) - the manifold dial visualization
    - [GitHub Repository](https://github.com/bassrehab/mhc-visualizer) - full source, PyTorch module, tests
    - [Colab Notebook](https://colab.research.google.com/github/bassrehab/mhc-visualizer/blob/main/notebook/mhc_exploration.ipynb) - run it yourself
    - [mHC Paper](https://arxiv.org/abs/2512.24880) - the original DeepSeek paper

    ## References
    Xie, Z., Wei, Y., Cao, H., et al. (2025). mHC: Manifold-Constrained Hyper-Connections. arXiv preprint arXiv:2512.24880.
    He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
    Sinkhorn, R., & Knopp, P. (1967). Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of Mathematics, 21(2), 343-348.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

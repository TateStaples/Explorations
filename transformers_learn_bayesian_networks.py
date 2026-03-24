# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch>=2.0.0",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    return F, mo, mpatches, nn, np, plt, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Transformers Learn Bayesian Networks Autoregressively In-Context

    **Paper:** [arXiv:2501.02547](https://arxiv.org/pdf/2501.02547) (PDF)

    This notebook implements the key results from the paper, which demonstrates that:

    1. **Theoretically**, a 2-layer transformer can estimate conditional probability tables of a Bayesian network from in-context examples (Theorem 4.1)
    2. **Empirically**, transformers trained from scratch learn to perform Bayesian inference, outperforming Naive Bayes on structured graphs

    The core mechanism: **Layer 1** acts as a *parent selector* (zeroing out irrelevant variables via the feed-forward network), and **Layer 2** performs *frequency counting* over the selected parent configurations via softmax attention.
    """)
    return


@app.cell
def _():
    # Graph structures from Figure 15 of the paper
    GRAPH_STRUCTURES: dict[str, dict[int, list[int]]] = {
        "general": {0: [], 1: [], 2: [0, 1], 3: [0, 2], 4: [3]},
        "tree": {0: [], 1: [0], 2: [0], 3: [1], 4: [1], 5: [2], 6: [2]},
        "chain": {i: [i - 1] if i > 0 else [] for i in range(10)},
    }
    # Node positions for visualization
    GRAPH_POSITIONS: dict[str, dict[int, tuple[float, float]]] = {
        "general": {0: (0, 2), 1: (2, 2), 2: (1, 1), 3: (1, 0), 4: (1, -1)},
        "tree": {
            0: (3, 3),
            1: (1.5, 2),
            2: (4.5, 2),
            3: (0.5, 1),
            4: (2.5, 1),
            5: (3.5, 1),
            6: (5.5, 1),
        },
        "chain": {i: (i, 0) for i in range(10)},
    }
    return GRAPH_POSITIONS, GRAPH_STRUCTURES


@app.cell
def _(np):
    class BayesianNetwork:
        """Discrete Bayesian network with binary variables."""

        def __init__(
            self,
            parents: dict[int, list[int]],
            d: int = 2,
            rng: np.random.Generator | None = None,
        ):
            self.parents = parents
            self.M = len(parents)
            self.d = d
            self.rng = rng or np.random.default_rng()
            self.cpts: dict[int, np.ndarray] = self._random_cpts()

        def _random_cpts(self) -> dict[int, np.ndarray]:
            """CPT probs drawn from U(0.15,0.3) | U(0.7,0.85)."""
            cpts = {}
            for var in range(self.M):
                n_configs = self.d ** len(self.parents[var])
                probs = np.empty(n_configs)
                for c in range(n_configs):
                    if self.rng.random() < 0.5:
                        probs[c] = self.rng.uniform(0.15, 0.3)
                    else:
                        probs[c] = self.rng.uniform(0.7, 0.85)
                cpts[var] = probs
            return cpts

        def parent_config_index(self, var: int, samples: np.ndarray) -> np.ndarray:
            pv = self.parents[var]
            if not pv:
                return np.zeros(samples.shape[:-1], dtype=int)
            idx = np.zeros(samples.shape[:-1], dtype=int)
            for j, p in enumerate(pv):
                idx += samples[..., p].astype(int) * (self.d ** j)
            return idx

        def sample(self, n: int) -> np.ndarray:
            data = np.zeros((n, self.M), dtype=int)
            for var in range(self.M):
                idx = self.parent_config_index(var, data)
                p0 = self.cpts[var][idx]
                data[:, var] = (self.rng.random(n) >= p0).astype(int)
            return data

        def true_conditional_prob(self, var: int, query: np.ndarray) -> np.ndarray:
            idx = self.parent_config_index(var, query[np.newaxis])[0]
            p0 = self.cpts[var][idx]
            return np.array([p0, 1 - p0])

        def optimal_mode_accuracy(self, var: int, query: np.ndarray) -> float:
            idx = self.parent_config_index(var, query[np.newaxis])[0]
            p0 = self.cpts[var][idx]
            return float(max(p0, 1.0 - p0))

    return (BayesianNetwork,)


@app.cell
def _(np):
    def build_input_tokens(
        context: np.ndarray, query: np.ndarray, m0: int, M: int, d: int = 2
    ) -> np.ndarray:
        """Build (N+1, M*(d+1)) input token matrix for predicting variable m0."""
        N = context.shape[0]
        token_dim = M * (d + 1)
        tokens = np.zeros((N + 1, token_dim), dtype=np.float32)
        for i in range(N):
            for m in range(M):
                tokens[i, m * d + context[i, m]] = 1.0
        for m in range(m0):
            tokens[N, m * d + query[m]] = 1.0
        tokens[N, M * d + m0] = 1.0
        return tokens

    return (build_input_tokens,)


@app.cell
def _(np):
    def naive_bayes_predict(
        context: np.ndarray, query: np.ndarray, m0: int, d: int = 2
    ) -> np.ndarray:
        """Frequency counting conditioned on ALL preceding variables."""
        N = context.shape[0]
        mask = np.ones(N, dtype=bool)
        for m in range(m0):
            mask &= context[:, m] == query[m]
        matching = context[mask]
        if len(matching) == 0:
            return np.ones(d) / d
        return np.array(
            [np.sum(matching[:, m0] == j) / len(matching) for j in range(d)]
        )

    def bayesian_inference_predict(
        context: np.ndarray,
        query: np.ndarray,
        m0: int,
        parents: dict[int, list[int]],
        d: int = 2,
    ) -> np.ndarray:
        """Frequency counting conditioned on PARENT variables only."""
        N = context.shape[0]
        mask = np.ones(N, dtype=bool)
        for pv in parents[m0]:
            mask &= context[:, pv] == query[pv]
        matching = context[mask]
        if len(matching) == 0:
            return np.ones(d) / d
        return np.array(
            [np.sum(matching[:, m0] == j) / len(matching) for j in range(d)]
        )

    return bayesian_inference_predict, naive_bayes_predict


@app.cell
def _(F, nn):
    class BNTransformer(nn.Module):
        """Transformer for learning Bayesian networks in-context."""

        def __init__(
            self,
            input_dim: int,
            d: int = 2,
            hidden_dim: int = 256,
            n_heads: int = 8,
            n_layers: int = 6,
        ):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
                activation="relu",
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.output_proj = nn.Linear(hidden_dim, d)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.input_proj(x)
            h = self.encoder(h)
            return self.output_proj(h[:, -1, :])

        def predict_probs(self, x: "torch.Tensor") -> "torch.Tensor":
            return F.softmax(self.forward(x), dim=-1)

    return (BNTransformer,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Interactive Bayesian Network Explorer

    Select a graph structure and a target variable to explore:
    - The **highlighted node** is the variable being predicted ($m_0$)
    - **Red nodes** are its parents $\mathcal{P}(m_0)$ — the variables the optimal predictor conditions on
    - Use the slider to vary the number of context examples and see how Bayesian inference vs Naive Bayes change
    """)
    return


@app.cell
def _(mo):
    explore_graph = mo.ui.dropdown(
        ["general", "tree", "chain"], value="general", label="Graph structure"
    )
    explore_graph
    return (explore_graph,)


@app.cell
def _(GRAPH_STRUCTURES: dict[str, dict[int, list[int]]], explore_graph, mo):
    _M = len(GRAPH_STRUCTURES[explore_graph.value])
    explore_var = mo.ui.slider(
        0, _M - 1, value=min(3, _M - 1), label="Target variable (m₀)"
    )
    explore_N = mo.ui.slider(5, 200, value=50, step=5, label="Context examples (N)")
    explore_seed = mo.ui.number(value=42, start=0, stop=9999, label="Random seed")
    mo.hstack([explore_var, explore_N, explore_seed], justify="start", gap=1)
    return explore_N, explore_seed, explore_var


@app.cell
def _(
    BayesianNetwork,
    GRAPH_POSITIONS: dict[str, dict[int, tuple[float, float]]],
    GRAPH_STRUCTURES: dict[str, dict[int, list[int]]],
    bayesian_inference_predict,
    build_input_tokens,
    explore_N,
    explore_graph,
    explore_seed,
    explore_var,
    mo,
    mpatches,
    naive_bayes_predict,
    np,
    plt,
):
    # ---- Resolve selections ----
    _gtype = explore_graph.value
    _parents = GRAPH_STRUCTURES[_gtype]
    _M = len(_parents)
    _m0 = explore_var.value
    _N = explore_N.value
    _pos = GRAPH_POSITIONS[_gtype]
    _parent_set = set(_parents[_m0])
    _d = 2

    # ---- Generate data ----
    _bn = BayesianNetwork(_parents, d=_d, rng=np.random.default_rng(int(explore_seed.value)))
    _samples = _bn.sample(_N + 1)
    _context = _samples[:_N]
    _query = _samples[_N]

    # ---- Compute predictions ----
    _true_prob = _bn.true_conditional_prob(_m0, _query)
    _bi_prob = bayesian_inference_predict(_context, _query, _m0, _parents, _d)
    _nb_prob = naive_bayes_predict(_context, _query, _m0, _d)

    # Count matching examples
    _bi_mask = np.ones(_N, dtype=bool)
    for _pv in _parents[_m0]:
        _bi_mask &= _context[:, _pv] == _query[_pv]
    _n_bi_match = int(np.sum(_bi_mask))

    _nb_mask = np.ones(_N, dtype=bool)
    for _m in range(_m0):
        _nb_mask &= _context[:, _m] == _query[_m]
    _n_nb_match = int(np.sum(_nb_mask))

    # ---- Build figure: graph + bar chart + input matrix side by side ----
    _fig = plt.figure(figsize=(16, 5))
    _gs = _fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1.3])
    _ax_graph = _fig.add_subplot(_gs[0])
    _ax_bar = _fig.add_subplot(_gs[1])
    _ax_mat = _fig.add_subplot(_gs[2])

    # -- Panel 1: Graph with highlighted parents --
    for child, plist in _parents.items():
        for p in plist:
            _lw = 2.5 if (child == _m0 and p in _parent_set) else 1.0
            _ec = "red" if (child == _m0 and p in _parent_set) else "gray"
            _ax_graph.annotate(
                "",
                xy=_pos[child],
                xytext=_pos[p],
                arrowprops=dict(arrowstyle="->", color=_ec, lw=_lw),
            )
    for node, (x, y) in _pos.items():
        if node == _m0:
            _color = "#4CAF50"
        elif node in _parent_set:
            _color = "#FF7043"
        else:
            _color = "lightsteelblue"
        _ax_graph.add_patch(
            plt.Circle((x, y), 0.35, color=_color, ec="black", lw=1.5, zorder=5)
        )
        _ax_graph.text(
            x, y, str(node), ha="center", va="center",
            fontsize=11, fontweight="bold", zorder=6,
        )
    xs = [v[0] for v in _pos.values()]
    ys = [v[1] for v in _pos.values()]
    _ax_graph.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    _ax_graph.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    _ax_graph.set_aspect("equal")
    _ax_graph.set_title(f"{_gtype.capitalize()} graph", fontsize=11)
    _ax_graph.axis("off")
    _ax_graph.legend(
        handles=[
            mpatches.Patch(color="#4CAF50", label=f"Target (X{_m0})"),
            mpatches.Patch(color="#FF7043", label=f"Parents {_parents[_m0]}"),
            mpatches.Patch(color="lightsteelblue", label="Other"),
        ],
        fontsize=8,
        loc="lower left",
    )

    # -- Panel 2: Probability comparison bar chart --
    _x_pos = np.arange(_d)
    _w = 0.22
    _ax_bar.bar(
        _x_pos - _w, _true_prob, _w,
        label="True CPT", color="#4CAF50", alpha=0.85,
    )
    _ax_bar.bar(
        _x_pos, _bi_prob, _w,
        label=f"Bayesian Inf. ({_n_bi_match} match)", color="#2196F3", alpha=0.85,
    )
    _ax_bar.bar(
        _x_pos + _w, _nb_prob, _w,
        label=f"Naive Bayes ({_n_nb_match} match)", color="#FF9800", alpha=0.85,
    )
    _ax_bar.set_xticks(_x_pos)
    _ax_bar.set_xticklabels([f"X{_m0}=0", f"X{_m0}=1"])
    _ax_bar.set_ylabel("P(X)")
    _ax_bar.set_ylim(0, 1.05)
    _ax_bar.set_title("Predicted distributions", fontsize=11)
    _ax_bar.legend(fontsize=7.5)

    # -- Panel 3: Input matrix heatmap --
    _tokens = build_input_tokens(_context, _query, _m0, _M, _d)
    _ax_mat.imshow(_tokens.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    _labels = []
    for m in range(_M):
        for v in range(_d):
            _labels.append(f"X{m}={v}")
    for m in range(_M):
        _labels.append(f"pos_{m}")
    _ax_mat.set_yticks(range(len(_labels)))
    _ax_mat.set_yticklabels(_labels, fontsize=6)
    _ax_mat.set_xlabel("Token (context | query)")
    _ax_mat.set_title(f"Input matrix X (predicting X{_m0})", fontsize=11)

    plt.tight_layout()

    # -- Info callout --
    _parent_str = ", ".join(f"X{p}" for p in _parents[_m0]) if _parents[_m0] else "none (root)"
    _query_str = ", ".join(f"X{i}={_query[i]}" for i in range(_M))

    mo.vstack([
        _fig,
        mo.callout(
            mo.md(
                f"**Query:** {_query_str}  \n"
                f"**Predicting:** X{_m0} | Parents = {{{_parent_str}}}  \n"
                f"**Bayesian Inf.** conditions on {len(_parents[_m0])} parent(s) → "
                f"{_n_bi_match}/{_N} matching examples  \n"
                f"**Naive Bayes** conditions on {_m0} predecessor(s) → "
                f"{_n_nb_match}/{_N} matching examples  \n"
                f"{'**More matches → lower variance → better estimate!**' if _n_bi_match > _n_nb_match else ''}"
            ),
            kind="info",
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Theoretical Construction (Theorem 4.1)

    The paper proves a **2-layer transformer** with bounded weights can approximate the optimal MLE:

    $$f(\mathbf{X}) = \text{Linear}_\mathbf{A}\!\left[\text{Read}\!\left(\text{TF}_{\theta^{(2)}}\!\left(\text{TF}_{\theta^{(1)}}(\mathbf{X})\right)\right)\right]$$

    | Layer | Role | Mechanism |
    |-------|------|-----------|
    | **Layer 1** | Parent selection | FFN uses $\mathbf{p}_q$ to zero out non-parent variables: $\widetilde{\mathbf{x}}_{mi} = \mathbf{x}_{mi} \cdot \mathbb{1}[m \in \{m_0\} \cup \mathcal{P}(m_0)]$ |
    | **Layer 2** | Frequency counting | Attention computes: $\widehat{\mathbf{x}}_{m_0 q}= \frac{1}{|\mathcal{I}(m_0)|}\sum\limits_{i \in \mathcal{I}(m_0)}\mathbf{x}_{m_0 i}$ |

    where $\mathcal{I}(m_0) = \{i : \mathbf{x}_{mi} = \mathbf{x}_{mq}\ \forall m \in \mathcal{P}(m_0)\}$ — context examples matching the query's parent values.

    **Try it above:** increase $N$ to see both estimates converge to the true CPT, or pick variables with more parents to see the gap between Naive Bayes and Bayesian inference grow.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Training

    Architecture from Table 1: **6 layers, 8 heads, hidden dim 256, AdamW, cross-entropy loss.**

    **Curriculum learning**: reveal variables progressively (start with 2, add one when loss stabilizes).
    Each step samples a fresh BN from a pool of 5k, generates $N_\text{train}+1$ observations.
    """)
    return


@app.cell
def _(mo):
    graph_type_select = mo.ui.dropdown(
        ["general", "tree", "chain"], value="general", label="Graph"
    )
    n_steps_slider = mo.ui.slider(200, 5000, value=2000, step=200, label="Steps")
    n_train_slider = mo.ui.slider(10, 200, value=100, step=10, label="N_train")
    batch_size_slider = mo.ui.slider(16, 128, value=64, step=16, label="Batch")
    lr_input = mo.ui.number(value=5e-4, start=1e-5, stop=1e-2, step=1e-4, label="LR")
    train_button = mo.ui.run_button(label="Train Model")
    mo.vstack([
        mo.hstack([graph_type_select, n_steps_slider, n_train_slider], justify="start", gap=1),
        mo.hstack([batch_size_slider, lr_input, train_button], justify="start", gap=1),
    ])
    return (
        batch_size_slider,
        graph_type_select,
        lr_input,
        n_steps_slider,
        n_train_slider,
        train_button,
    )


@app.cell
def _(
    BNTransformer,
    BayesianNetwork,
    GRAPH_STRUCTURES: dict[str, dict[int, list[int]]],
    batch_size_slider,
    build_input_tokens,
    graph_type_select,
    lr_input,
    mo,
    n_steps_slider,
    n_train_slider,
    np,
    plt,
    torch,
    train_button,
):
    mo.stop(
        not train_button.value,
        mo.md("*Click **Train Model** to start. General graph / 2000 steps ≈ 2-5 min on CPU.*"),
    )

    _graph_type = graph_type_select.value
    _parents = GRAPH_STRUCTURES[_graph_type]
    _M = len(_parents)
    _d = 2
    _N_train = n_train_slider.value
    _batch_size = batch_size_slider.value
    _n_steps = n_steps_slider.value
    _lr = lr_input.value
    _token_dim = _M * (_d + 1)

    _device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Pre-generate BN pool
    _rng = np.random.default_rng(0)
    _bn_pool = [
        BayesianNetwork(_parents, d=_d, rng=np.random.default_rng(_rng.integers(1 << 31)))
        for _ in range(5000)
    ]

    trained_model = BNTransformer(
        input_dim=_token_dim, d=_d, hidden_dim=256, n_heads=8, n_layers=6
    ).to(_device)
    _optimizer = torch.optim.AdamW(trained_model.parameters(), lr=_lr, weight_decay=5e-2)
    _criterion = torch.nn.CrossEntropyLoss()

    _revealed = 2
    _curriculum_window: list[float] = []
    _curriculum_patience = 50
    _curriculum_threshold = 0.68

    train_losses: list[tuple[int, float]] = []
    train_accs: list[tuple[int, float]] = []
    train_curriculum_history: list[tuple[int, int]] = []

    with mo.status.progress_bar(total=_n_steps) as _bar:
        for _step in range(_n_steps):
            trained_model.train()
            _batch_X = np.zeros((_batch_size, _N_train + 1, _token_dim), dtype=np.float32)
            _batch_y = np.zeros(_batch_size, dtype=np.int64)
            for _b in range(_batch_size):
                _bn = _bn_pool[_rng.integers(5000)]
                _samples = _bn.sample(_N_train + 1)
                _m0 = int(_rng.integers(_revealed))
                _batch_X[_b] = build_input_tokens(_samples[:_N_train], _samples[_N_train], _m0, _M, _d)
                _batch_y[_b] = _samples[_N_train, _m0]

            _X_t = torch.from_numpy(_batch_X).to(_device)
            _y_t = torch.from_numpy(_batch_y).to(_device)
            _logits = trained_model(_X_t)
            _loss = _criterion(_logits, _y_t)
            _optimizer.zero_grad()
            _loss.backward()
            _optimizer.step()

            _lv = _loss.item()
            _av = (_logits.argmax(dim=-1) == _y_t).float().mean().item()
            if _step % 50 == 0:
                train_losses.append((_step, _lv))
                train_accs.append((_step, _av))
                train_curriculum_history.append((_step, _revealed))

            _curriculum_window.append(_lv)
            if len(_curriculum_window) > _curriculum_patience:
                _curriculum_window.pop(0)
            if (
                len(_curriculum_window) >= _curriculum_patience
                and np.mean(_curriculum_window) < _curriculum_threshold
                and _revealed < _M
            ):
                _revealed += 1
                _curriculum_window.clear()
            _bar.update()

    trained_model.eval()
    trained_graph_type = _graph_type
    trained_parents = _parents

    # Plot training curves
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _sl, _vl = zip(*train_losses)
    _ax1.plot(_sl, _vl, color="steelblue", alpha=0.8)
    _ax1.set_xlabel("Step")
    _ax1.set_ylabel("Loss")
    _ax1.set_title("Training Loss")
    _sa, _va = zip(*train_accs)
    _ax2.plot(_sa, _va, color="seagreen", alpha=0.8)
    _ax2.set_xlabel("Step")
    _ax2.set_ylabel("Accuracy")
    _ax2.set_title("Training Accuracy")
    for _i in range(1, len(train_curriculum_history)):
        if train_curriculum_history[_i][1] > train_curriculum_history[_i - 1][1]:
            _s = train_curriculum_history[_i][0]
            _ax1.axvline(_s, color="red", ls="--", alpha=0.5)
            _ax2.axvline(_s, color="red", ls="--", alpha=0.5, label=f"reveal var {train_curriculum_history[_i][1]}")
    _ax2.legend(fontsize=8)
    plt.tight_layout()

    mo.vstack([
        mo.md(
            f"**Done** — `{_graph_type}` graph, {_n_steps} steps. "
            f"Final loss: {train_losses[-1][1]:.4f}, acc: {train_accs[-1][1]:.3f}. "
            f"Red lines = curriculum advances."
        ),
        _fig,
    ])
    return trained_graph_type, trained_model, trained_parents


@app.cell
def _(trained_model):
    next(trained_model.parameters()).device
    return


@app.cell
def _(trained_model):
    trained_model
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Prediction Explorer

    After training, use the controls below to **explore individual predictions**.
    Generate random test cases and see how the transformer's output compares to the baselines in real time.
    """)
    return


@app.cell
def _(mo, trained_parents):
    _M = len(trained_parents)
    pred_var = mo.ui.slider(0, _M - 1, value=min(3, _M - 1), label="Variable to predict")
    pred_N = mo.ui.slider(5, 200, value=50, step=5, label="N_test (context)")
    pred_seed = mo.ui.number(value=7, start=0, stop=9999, label="Seed (change to resample)")
    mo.hstack([pred_var, pred_N, pred_seed], justify="start", gap=1)
    return pred_N, pred_seed, pred_var


@app.cell
def _(
    BayesianNetwork,
    bayesian_inference_predict,
    build_input_tokens,
    mo,
    naive_bayes_predict,
    np,
    plt,
    pred_N,
    pred_seed,
    pred_var,
    torch,
    trained_model,
    trained_parents,
):
    _parents = trained_parents
    _M = len(_parents)
    _d = 2
    _m0 = pred_var.value
    _N = pred_N.value
    _device = next(trained_model.parameters()).device

    # Generate test data
    _bn = BayesianNetwork(_parents, d=_d, rng=np.random.default_rng(int(pred_seed.value)))
    _samples = _bn.sample(_N + 1)
    _context = _samples[:_N]
    _query = _samples[_N]
    _true_val = _query[_m0]

    # Predictions
    _tokens = build_input_tokens(_context, _query, _m0, _M, _d)
    _x_t = torch.from_numpy(_tokens).unsqueeze(0).to(_device)
    with torch.no_grad():
        _tf_probs = trained_model.predict_probs(_x_t)[0].cpu().numpy()
    _bi_probs = bayesian_inference_predict(_context, _query, _m0, _parents, _d)
    _nb_probs = naive_bayes_predict(_context, _query, _m0, _d)
    _true_probs = _bn.true_conditional_prob(_m0, _query)

    # Matching counts
    _bi_mask = np.ones(_N, dtype=bool)
    for _pv in _parents[_m0]:
        _bi_mask &= _context[:, _pv] == _query[_pv]
    _nb_mask = np.ones(_N, dtype=bool)
    for _m in range(_m0):
        _nb_mask &= _context[:, _m] == _query[_m]

    # ---- Plot ----
    _fig, (_ax_bar, _ax_ctx) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Bar chart
    _x = np.arange(_d)
    _w = 0.18
    _ax_bar.bar(_x - 1.5 * _w, _true_probs, _w, label="True CPT", color="#4CAF50", alpha=0.85)
    _ax_bar.bar(_x - 0.5 * _w, _bi_probs, _w, label=f"Bayesian Inf. ({int(np.sum(_bi_mask))} match)", color="#2196F3", alpha=0.85)
    _ax_bar.bar(_x + 0.5 * _w, _nb_probs, _w, label=f"Naive Bayes ({int(np.sum(_nb_mask))} match)", color="#FF9800", alpha=0.85)
    _ax_bar.bar(_x + 1.5 * _w, _tf_probs, _w, label="Transformer", color="#9C27B0", alpha=0.85)
    _ax_bar.axhline(0.5, color="gray", ls=":", alpha=0.4)
    _ax_bar.set_xticks(_x)
    _ax_bar.set_xticklabels([f"X{_m0}=0", f"X{_m0}=1"])
    _ax_bar.set_ylim(0, 1.05)
    _ax_bar.set_ylabel("Probability")
    _ax_bar.set_title(f"Predictions for X{_m0} (true value = {_true_val})")
    _ax_bar.legend(fontsize=8)

    # Context visualization: scatter showing matching examples
    _ax_ctx.set_title(f"Context examples (N={_N})", fontsize=11)
    _colors_ctx = np.full(_N, "lightgray", dtype=object)
    _colors_ctx[_bi_mask] = "#2196F3"
    _both = _bi_mask & _nb_mask
    _colors_ctx[_both] = "#FF9800"
    _ax_ctx.scatter(range(_N), _context[:, _m0], c=_colors_ctx, s=20, alpha=0.7, edgecolors="none")
    _ax_ctx.set_xlabel("Example index")
    _ax_ctx.set_ylabel(f"X{_m0} value")
    _ax_ctx.set_yticks([0, 1])
    _ax_ctx.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", label="Match parents", markersize=8),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF9800", label="Match all predecessors", markersize=8),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray", label="No match", markersize=8),
        ],
        fontsize=7,
        loc="upper right",
    )
    plt.tight_layout()

    # Correctness indicators
    _tf_pred = int(np.argmax(_tf_probs))
    _bi_pred = int(np.argmax(_bi_probs))
    _nb_pred = int(np.argmax(_nb_probs))

    def _check(pred: int) -> str:
        return "correct" if pred == _true_val else "wrong"

    _parent_str = ", ".join(f"X{p}={_query[p]}" for p in _parents[_m0]) if _parents[_m0] else "none"

    mo.vstack([
        _fig,
        mo.hstack([
            mo.stat(value=f"{_tf_probs[_true_val]:.3f}", label=f"Transformer P(X{_m0}={_true_val})", bordered=True),
            mo.stat(value=f"{_bi_probs[_true_val]:.3f}", label=f"Bayesian Inf. P(X{_m0}={_true_val})", bordered=True),
            mo.stat(value=f"{_nb_probs[_true_val]:.3f}", label=f"Naive Bayes P(X{_m0}={_true_val})", bordered=True),
            mo.stat(value=f"{_true_probs[_true_val]:.3f}", label="True CPT", bordered=True),
        ], justify="start", gap=0.5),
        mo.callout(
            mo.md(
                f"**True value:** X{_m0} = {_true_val} | "
                f"**Parent values:** {_parent_str}  \n"
                f"Transformer: **{_check(_tf_pred)}** | "
                f"Bayesian Inf.: **{_check(_bi_pred)}** | "
                f"Naive Bayes: **{_check(_nb_pred)}**"
            ),
            kind="success" if _tf_pred == _true_val else "warn",
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Evaluation — Accuracy vs. Number of Examples

    Reproducing Figure 2: for each $N_\text{test}$ value, we run 500 trials on a held-out BN and report average accuracy. This runs automatically after training.
    """)
    return


@app.cell
def _(
    BayesianNetwork,
    bayesian_inference_predict,
    build_input_tokens,
    mo,
    naive_bayes_predict,
    np,
    plt,
    torch,
    trained_graph_type,
    trained_model,
    trained_parents,
):
    _parents = trained_parents
    _gtype = trained_graph_type
    _M = len(_parents)
    _d = 2
    _device = next(trained_model.parameters()).device

    _eval_vars_map = {
        "general": [0, 2, 3],
        "tree": [0, 2, 4],
        "chain": [0, 4, 7],
    }
    _eval_vars = _eval_vars_map.get(_gtype, list(range(min(3, _M))))
    _N_vals = [5, 10, 20, 30, 50, 75, 100]
    _n_runs = 10  # independent runs (different test BNs) for error bars
    _n_trials_per_run = 150  # trials per run

    _methods = ["Transformer", "Naive Bayes", "Bayesian Inference", "Optimal Acc"]

    # results[var][method] = array of shape (n_runs, len(N_vals))
    _results: dict[int, dict[str, np.ndarray]] = {
        v: {m: np.zeros((_n_runs, len(_N_vals))) for m in _methods}
        for v in _eval_vars
    }

    _total = len(_eval_vars) * len(_N_vals) * _n_runs
    with mo.status.progress_bar(total=_total) as _bar:
        for _var in _eval_vars:
            for _ni, _Nt in enumerate(_N_vals):
                for _run in range(_n_runs):
                    _run_rng = np.random.default_rng(1000 + _run * 137)
                    _test_bn = BayesianNetwork(
                        _parents, d=_d, rng=np.random.default_rng(100 + _run)
                    )
                    _tf_c = _nb_c = _bi_c = 0
                    _opt_c = 0.0
                    for _ in range(_n_trials_per_run):
                        _s = _test_bn.sample(_Nt + 1)
                        _ctx, _q = _s[:_Nt], _s[_Nt]
                        _tv = _q[_var]
                        _tok = build_input_tokens(_ctx, _q, _var, _M, _d)
                        _xt = torch.from_numpy(_tok).unsqueeze(0).to(_device)
                        with torch.no_grad():
                            _p = trained_model.predict_probs(_xt)[0]
                        _tf_c += int(_p.argmax().item() == _tv)
                        _nb_c += int(np.argmax(naive_bayes_predict(_ctx, _q, _var, _d)) == _tv)
                        _bi_c += int(np.argmax(bayesian_inference_predict(_ctx, _q, _var, _parents, _d)) == _tv)
                        _opt_c += _test_bn.optimal_mode_accuracy(_var, _q)
                    _results[_var]["Transformer"][_run, _ni] = _tf_c / _n_trials_per_run
                    _results[_var]["Naive Bayes"][_run, _ni] = _nb_c / _n_trials_per_run
                    _results[_var]["Bayesian Inference"][_run, _ni] = _bi_c / _n_trials_per_run
                    _results[_var]["Optimal Acc"][_run, _ni] = _opt_c / _n_trials_per_run
                    _bar.update()

    # ---- Plot with error bands (mean ± 1 std across runs) ----
    _colors = {"Naive Bayes": "#FF9800", "Bayesian Inference": "#2196F3", "Optimal Acc": "gray", "Transformer": "#9C27B0"}
    _markers = {"Naive Bayes": "s", "Bayesian Inference": "^", "Transformer": "o"}

    _nv = len(_eval_vars)
    _fig, _axes = plt.subplots(1, _nv, figsize=(5 * _nv, 4))
    if _nv == 1:
        _axes = [_axes]
    for _ax, _var in zip(_axes, _eval_vars):
        for _m in _methods:
            _mean = _results[_var][_m].mean(axis=0)
            _std = _results[_var][_m].std(axis=0)
            _ls = "--" if _m == "Optimal Acc" else "-"
            _mk = _markers.get(_m)
            _ax.plot(
                _N_vals, _mean, _ls, marker=_mk, color=_colors[_m],
                label=_m, markersize=5, alpha=0.85,
            )
            _ax.fill_between(
                _N_vals, _mean - _std, _mean + _std,
                color=_colors[_m], alpha=0.15,
            )
        _ax.set_xlabel("# examples")
        _ax.set_ylabel("Test Accuracy")
        _ax.set_title(f"Var. {_var}  (parents={_parents[_var]})")
        _ax.legend(fontsize=7.5)
        _ax.set_ylim(0.2, 0.95)
        _ax.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.vstack([
        mo.md(
            f"**Figure 2 reproduction** — `{_gtype}` graph, "
            f"{_n_runs} runs x {_n_trials_per_run} trials per point. "
            f"Shaded regions = ±1 std across runs (different test BNs)."
        ),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Key Takeaways

    1. **Transformers learn graph structure** — for variables with non-trivial parents (e.g. Var 3 in the general graph), the transformer matches or exceeds Bayesian inference, demonstrating it learns the correct conditioning set.

    2. **Better than Naive Bayes** — by focusing on parents rather than all predecessors, the transformer uses more matching examples → lower variance.

    3. **Generalization across $N$** — trained on $N_\text{train}=100$, generalizes to different $N_\text{test}$.

    4. **Curriculum matters** — progressively revealing variables helps learn the full graph structure.

    ---

    **Next: recovering the BN from weights** — the theoretical construction (Layer 1 = parent selection) suggests attention patterns encode parent relationships. Inspect `trained_model.encoder.layers[0]` weights to look for this structure.
    """)
    return


if __name__ == "__main__":
    app.run()

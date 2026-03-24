# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "networkx",
#     "anywidget",
#     "traitlets"
# ]
# requires-python = ">=3.10"
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx

    return mo, np, nx, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Graphical Models and Inference: A Comprehensive Overview

    This notebook provides an interactive and technically rigorous exploration of the foundational concepts in Graphical Models and Inference (aligned with ECE 590.17). We bridge the perspectives of probability theory, statistical physics, and machine learning to understand how structured representations of uncertainty allow for efficient reasoning in complex, high-dimensional systems.

    ---

    ## Chapter 1: Foundations of Probability and Inference

    **Goal:** Establish the formal axiomatic framework of probability and the limit theorems that justify statistical inference and macroscopic modeling.

    ### Probability Spaces and Random Variables
    The formal foundation of inference rests upon a **Probability Space**, defined as a triplet $(\Omega, \mathcal{F}, \mathbb{P})$. Here, $\Omega$ is the sample space (all possible outcomes), $\mathcal{F}$ is a $\sigma$-algebra of events (subsets of $\Omega$), and $\mathbb{P}: \mathcal{F} \to [0,1]$ is a probability measure satisfying Kolmogorov's axioms: non-negativity, $\mathbb{P}(\Omega) = 1$, and countable additivity for disjoint events.

    A **Random Variable** $X$ is a measurable function $X: \Omega \to \mathcal{X}$ mapping outcomes to a measurable space (often $\mathbb{R}$). For discrete random variables, we characterize the distribution via a **Probability Mass Function (PMF)** $p_X(x) = \mathbb{P}(X = x)$.

    ### Joint, Marginal, and Conditional Distributions
    When dealing with multiple variables, say $X$ and $Y$, their combined behavior is captured by the **Joint Distribution** $p_{X,Y}(x,y)$. We can recover the individual behavior of $X$ through **Marginalization**, summing over the state space of $Y$:
    $$ p_X(x) = \sum_{y \in \mathcal{Y}} p_{X,Y}(x,y) $$

    The cornerstone of Bayesian reasoning is the **Conditional Distribution**, which quantifies how our belief about $X$ updates given the realization of $Y = y$:
    $$ p_{X|Y}(x|y) = \frac{p_{X,Y}(x,y)}{p_Y(y)} \quad \text{for } p_Y(y) > 0 $$
    This naturally leads to **Bayes' Theorem**, which reverses the conditioning: $p_{Y|X}(y|x) = \frac{p_{X|Y}(x|y)p_X(x)}{p_Y(y)}$.
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md("**Planned Interactive Widget:** An interactive conditional probability visualizer (e.g., a 2D grid or Venn diagram) where users can drag sliders to change priors and visually watch the posterior distribution update in real-time."),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Expectation, Variance, and Limit Theorems
    The **Expectation** (or mean) of a function $f(X)$ with respect to distribution $p$ is defined as $\mathbb{E}_p[f(X)] = \sum_{x \in \mathcal{X}} f(x)p(x)$. The **Variance** measures the expected squared deviation from the mean: $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$.

    For a sequence of independent and identically distributed (i.i.d.) random variables $X_1, X_2, \dots, X_n$, asymptotic behavior is governed by two critical theorems:
    1. **Law of Large Numbers (LLN):** The sample average $\frac{1}{n}\sum_{i=1}^n X_i$ converges to the expected value $\mathbb{E}[X_1]$ as $n \to \infty$.
    2. **Central Limit Theorem (CLT):** The normalized sum $\frac{1}{\sqrt{n}}\sum_{i=1}^n (X_i - \mathbb{E}[X_i])$ converges in distribution to a Gaussian $\mathcal{N}(0, \text{Var}(X_1))$. This explains the ubiquity of the Gaussian distribution in macroscopic phenomena.

    **Takeaways:**
    *   Understand the Kolmogorov axioms as the rigorous basis for all probability.
    *   Distinguish between Joint, Marginal, and Conditional distributions.
    *   Apply Bayes' Theorem to invert conditional relationships.
    *   Recognize that the LLN justifies sample-based estimation, while the CLT characterizes the error distribution.
    """)
    return


@app.cell
def _(mo, np):
    n_samples_slider = mo.ui.slider(steps=np.logspace(1, 3, 50), value=100, label="Samples per Trial (n)")
    n_trials_slider = mo.ui.slider(steps=np.logspace(1, np.log10(5000), 50), value=1000, label="Number of Trials (N)")

    mo.md(f"""
    **Interactive Simulation: LLN and CLT**
    Adjust the sliders to observe how individual trials converge (LLN) and how the overall distribution of means approaches a Gaussian (CLT).

    {mo.hstack([n_samples_slider, n_trials_slider])}
    """)
    return n_samples_slider, n_trials_slider


@app.cell
def _(n_samples_slider, n_trials_slider, np, plt):
    _n = n_samples_slider.value
    _trials = n_trials_slider.value

    # Simulation: Uniform(0,1)
    _data = np.random.rand(_trials, _n)
    _trial_means = np.mean(_data, axis=1)

    # LLN: Running average of a single trial
    _single_trial = np.random.rand(_n)
    _running_avg = np.cumsum(_single_trial) / np.arange(1, _n + 1)

    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4))

    # Left: LLN Convergence
    _ax[0].plot(_running_avg, color='steelblue', alpha=0.8, linewidth=1.5)
    _ax[0].axhline(0.5, color='red', linestyle='--', label='Theoretical Mean (0.5)')
    _ax[0].set_title("Law of Large Numbers\n(Running Mean of 1 Sequence)")
    _ax[0].set_xlabel("Sample Size (n)")
    _ax[0].set_ylabel("Mean Value")
    _ax[0].set_ylim(0, 1)
    _ax[0].legend()
    _ax[0].grid(alpha=0.3)

    # Right: CLT Histogram
    _ax[1].hist(_trial_means, bins=35, density=True, color='skyblue', edgecolor='white', alpha=0.7)

    # Theoretical Gaussian Overlay: Mean = 0.5, Var = (1/12)/n (variance of Uniform(0,1) is 1/12)
    _mu = 0.5
    _sigma = np.sqrt(1/12 / _n)
    _x = np.linspace(_mu - 4*_sigma, _mu + 4*_sigma, 100)
    _p = (1 / (np.sqrt(2 * np.pi) * _sigma)) * np.exp(-0.5 * ((_x - _mu) / _sigma)**2)
    _ax[1].plot(_x, _p, 'r', linewidth=2, label=rf'Gaussian: $\mathcal{{N}}(0.5, \sigma^2/n)$')
    _ax[1].set_title(f"Central Limit Theorem\n(Dist. of {_trials} Averages)")
    _ax[1].set_xlabel("Trial Mean")
    _ax[1].set_ylabel("Probability Density")
    _ax[1].legend()
    _ax[1].grid(alpha=0.3)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Chapter 2: Factor Graphs and Structural Equivalences

    **Goal:** Introduce the structured factorization of high-dimensional distributions and the various graphical representations (Directed, Undirected, and Bipartite) used to exploit local dependencies for efficient inference.

    High-dimensional probability distributions $p(x_1, \dots, x_N)$ suffer from the curse of dimensionality. Graphical models mitigate this by asserting that the global joint distribution factorizes into a product of local functions.

    ### Factor Graphs
    A **Factor Graph** is a bipartite graph $\mathcal{G} = (\mathcal{V}, \mathcal{F}, \mathcal{E})$ where $\mathcal{V}$ is a set of variable nodes, $\mathcal{F}$ is a set of factor nodes, and edges $\mathcal{E}$ connect variables to the factors they participate in. The global distribution is factored as:
    $$ p(\mathbf{x}) = \frac{1}{Z} \prod_{a \in \mathcal{F}} f_a(\mathbf{x}_{\partial a}) $$
    where $\mathbf{x}_{\partial a}$ denotes the subset of variables connected to factor $a$, and $Z = \sum_{\mathbf{x}} \prod_{a} f_a(\mathbf{x}_{\partial a})$ is the **Partition Function** that normalizes the distribution.
    """)
    return


@app.cell
def _(nx, plt):
    _G = nx.Graph()
    _var_nodes = ["x1", "x2", "x3", "x4"]
    _factor_nodes = ["f1", "f2", "f3"]
    _G.add_nodes_from(_var_nodes, bipartite=0)
    _G.add_nodes_from(_factor_nodes, bipartite=1)
    _G.add_edges_from([
        ("x1", "f1"), ("x2", "f1"),
        ("x2", "f2"), ("x3", "f2"),
        ("x3", "f3"), ("x4", "f3"),
    ])
    _pos = {
        "x1": (0, 1), "x2": (1, 1), "x3": (2, 1), "x4": (3, 1),
        "f1": (0.5, 0), "f2": (1.5, 0), "f3": (2.5, 0),
    }
    _fig, _ax = plt.subplots(figsize=(7, 3))
    nx.draw_networkx_edges(_G, _pos, ax=_ax, edge_color="gray", width=1.5)
    nx.draw_networkx_nodes(_G, _pos, nodelist=_var_nodes, node_color="steelblue",
                           node_shape="o", node_size=900, ax=_ax)
    nx.draw_networkx_nodes(_G, _pos, nodelist=_factor_nodes, node_color="gold",
                           node_shape="s", node_size=700, ax=_ax)
    nx.draw_networkx_labels(_G, _pos, ax=_ax, font_color="white", font_size=10)
    _ax.set_title("Factor Graph: Bipartite Structure\n(circles = variables, squares = factors)")
    _ax.axis("off")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Bayesian Networks and Markov Random Fields
    - **Directed Acyclic Graphs (Bayesian Networks):** Variables are connected by directed edges representing causal or generative relationships. The joint distribution factorizes into conditional probabilities: $p(\mathbf{x}) = \prod_{i=1}^N p(x_i | \text{pa}(x_i))$, where $\text{pa}(x_i)$ are the parents of $x_i$. By definition, $Z=1$. Independence is determined using the **d-separation** criterion.
    """)
    return


@app.cell
def _(nx, plt):
    _G = nx.DiGraph()
    _G.add_edges_from([
        ("Rain", "Wet Grass"), ("Rain", "Slippery Road"),
        ("Sprinkler", "Wet Grass"), ("Wet Grass", "Slippery Road"),
        ("Slippery Road", "Traffic"), ("Traffic", "Late"),
        ("Slippery Road", "Accident"),
    ])
    _pos = {
        "Rain": (1, 3), "Sprinkler": (3, 3),
        "Wet Grass": (2, 2), "Slippery Road": (1, 1),
        "Traffic": (0, 0), "Late": (0, -1), "Accident": (2, 0),
    }
    _fig, _ax = plt.subplots(figsize=(6, 5))
    nx.draw(_G, _pos, with_labels=True, node_color="steelblue", node_size=1800,
            font_size=8, font_color="white", edge_color="gray",
            arrows=True, arrowsize=20, ax=_ax)
    _ax.set_title("Bayesian Network (DAG)\nDirected edges = causal/generative relationships")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **Undirected Graphs (Markov Random Fields):** Variables are connected by undirected edges. The distribution factorizes over maximal cliques $C$ of the graph: $p(\mathbf{x}) = \frac{1}{Z} \prod_{C} \psi_C(\mathbf{x}_C)$. Conditional independence corresponds directly to graph reachability: if node set $A$ is separated from $B$ by $C$, then $\mathbf{x}_A \perp\!\!\!\perp \mathbf{x}_B \mid \mathbf{x}_C$.

    The **Hammersley-Clifford Theorem** establishes that any strictly positive distribution satisfying the local Markov properties of an undirected graph can be factorized over its cliques (i.e., it is a Gibbs distribution). The **Markov Blanket** of a node $i$ is the minimal set of nodes that renders $i$ conditionally independent of all other nodes in the network.
    """)
    return


@app.cell
def _():
    import anywidget
    import traitlets

    class MarkovBlanketWidget(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
          const W = 640, H = 480;
          const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
          svg.setAttribute("width", W);
          svg.setAttribute("height", H);
          el.appendChild(svg);

          const legend = document.createElement("div");
          legend.style.cssText = "font-size:12px; margin-top:6px; display:flex; gap:14px;";
          for (const [color, label] of [
            ["#e74c3c", "Selected"], ["#f39c12", "Markov Blanket"], ["#5b9bd5", "Other"],
          ]) {
            legend.innerHTML += `<span><svg width="12" height="12" style="vertical-align:middle">
              <circle cx="6" cy="6" r="6" fill="${color}"/></svg> ${label}</span>`;
          }
          el.appendChild(legend);

          function draw() {
            const nodes = model.get("nodes");
            const edges = model.get("edges");
            const selectedList = model.get("selected_nodes");
            const selected = new Set(selectedList);
            const adj = {};
            for (const n of nodes) adj[n.id] = new Set();
            for (const [s, d] of edges) { adj[s].add(d); adj[d].add(s); }
            const blanket = new Set();
            for (const id of selected) {
              for (const nb of (adj[id] || [])) {
                if (!selected.has(nb)) blanket.add(nb);
              }
            }
            const nodeMap = {};
            for (const n of nodes) nodeMap[n.id] = n;

            while (svg.firstChild) svg.removeChild(svg.firstChild);

            for (const [s, d] of edges) {
              const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
              line.setAttribute("x1", nodeMap[s].x); line.setAttribute("y1", nodeMap[s].y);
              line.setAttribute("x2", nodeMap[d].x); line.setAttribute("y2", nodeMap[d].y);
              line.setAttribute("stroke", "#999");
              line.setAttribute("stroke-width", "1.5");
              svg.appendChild(line);
            }

            for (const n of nodes) {
              const color = selected.has(n.id) ? "#e74c3c" : blanket.has(n.id) ? "#f39c12" : "#5b9bd5";
              const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
              g.style.cursor = "pointer";
              g.addEventListener("click", () => {
                const cur = new Set(model.get("selected_nodes"));
                if (cur.has(n.id)) cur.delete(n.id); else cur.add(n.id);
                model.set("selected_nodes", [...cur]);
                model.save_changes();
              });
              const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
              circle.setAttribute("cx", n.x); circle.setAttribute("cy", n.y);
              circle.setAttribute("r", 16); circle.setAttribute("fill", color);
              circle.setAttribute("stroke", "white"); circle.setAttribute("stroke-width", "2");
              g.appendChild(circle);
              const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
              text.setAttribute("x", n.x); text.setAttribute("y", n.y + 4);
              text.setAttribute("text-anchor", "middle");
              text.setAttribute("font-size", "11");
              text.setAttribute("fill", "white");
              text.setAttribute("pointer-events", "none");
              text.textContent = n.id;
              g.appendChild(text);
              svg.appendChild(g);
            }
          }

          draw();
          model.on("change:selected_nodes", draw);
          model.on("change:nodes", draw);
        }

        export default { render };
        """
        _css = """
        svg { display: block; background: #f8f9fa; border-radius: 8px; border: 1px solid #ddd; }
        @media (prefers-color-scheme: dark) { svg { background: #1e1e2e; border-color: #444; } }
        """
        nodes = traitlets.List([]).tag(sync=True)
        edges = traitlets.List([]).tag(sync=True)
        selected_nodes = traitlets.List([]).tag(sync=True)

    return MarkovBlanketWidget, anywidget, traitlets


@app.cell
def _(MarkovBlanketWidget, mo, np, nx):
    _G = nx.erdos_renyi_graph(n=12, p=0.25, seed=42)
    _rng = np.random.default_rng(42)
    while not nx.is_connected(_G):
        _components = list(nx.connected_components(_G))
        _a = _rng.choice(list(_components[0]))
        _b = _rng.choice(list(_components[1]))
        _G.add_edge(int(_a), int(_b))

    _layout = nx.spring_layout(_G, seed=42, k=3.0)
    _W, _H, _PAD = 640, 480, 45
    _nodes_data = [
        {"id": int(n),
         "x": float((_layout[n][0] + 1) / 2 * (_W - 2 * _PAD) + _PAD),
         "y": float((_layout[n][1] + 1) / 2 * (_H - 2 * _PAD) + _PAD)}
        for n in _G.nodes()
    ]
    _edges_data = [[int(u), int(v)] for u, v in _G.edges()]

    markov_blanket_widget = mo.ui.anywidget(
        MarkovBlanketWidget(nodes=_nodes_data, edges=_edges_data, selected_nodes=[])
    )
    mo.vstack([
        mo.md("**Interactive Markov Blanket Explorer** — Click nodes to select (red); click again to deselect. The union Markov Blanket of all selected nodes is highlighted in gold."),
        markov_blanket_widget,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exact Inference: Variable Elimination and Belief Propagation
    To compute a marginal $p(x_i)$, we must sum over all other variables: $p(x_i) = \sum_{\mathbf{x} \setminus x_i} p(\mathbf{x})$. **Variable Elimination** exploits the distributive law, pushing summations inside products. However, its complexity is exponential in the **Treewidth** of the graph (the size of the largest clique in the induced chordal graph).

    For tree-structured graphs (treewidth 1), exact inference is achieved efficiently via **Belief Propagation (Sum-Product Algorithm)**. Nodes pass "messages" along edges:
    - **Variable to Factor:** $\mu_{i \to a}(x_i) = \prod_{c \in \partial i \setminus a} \mu_{c \to i}(x_i)$
        - Variable's distribution of values
    - **Factor to Variable:** $\mu_{a \to i}(x_i) = \sum_{\mathbf{x}_{\partial a \setminus i}} f_a(\mathbf{x}_{\partial a})
    \prod_{j \in \partial a \setminus i} \mu_{j \to a}(x_j)$
        - Distribution of the recieving variable given all the other factor componenets

    Running BP on graphs with cycles is known as **Loopy Belief Propagation**, an approximate heuristic that often performs remarkably well in practice (e.g., in LDPC decoding).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Example: Robot Localization HMM

    A robot walks along a corridor with 6 discrete positions $\{0,1,2,3,4,5\}$. We observe its (noisy) position at steps $t=1,3,5$ and want to infer the marginal $p(X_5 \mid Y_1, Y_3, Y_5)$ via the **Sum-Product**/**Belief-Propagation** forward pass.

    **Model:**
    - **Prior:** $P(X_0 = 0) = 1$ — robot starts at position 0
    - **Transition:** $P(X_{t+1} = i{+}1 \mid X_t = i) = p_\text{fwd}$, $P(X_{t+1} = i \mid X_t = i) = 1 - p_\text{fwd}$ (absorbing at 5)
    - **Observation:** $P(Y_t = y \mid X_t = x) = p_\text{obs}$ if $x = y$, else $\frac{1-p_\text{obs}}{2}$ split to neighbors $x \pm 1$, zero elsewhere
    """)
    return


@app.cell
def _(mo):
    p_fwd_slider   = mo.ui.slider(0.5, 0.99, step=0.05, value=0.7,  label="P(step forward)")
    obs_acc_slider = mo.ui.slider(0.5, 0.99, step=0.05, value=0.8,  label="Obs. accuracy")
    y1_input = mo.ui.number(start=0, stop=5, step=1, value=1, label="y₁")
    y3_input = mo.ui.number(start=0, stop=5, step=1, value=3, label="y₃")
    y5_input = mo.ui.number(start=0, stop=5, step=1, value=5, label="y₅")
    mo.vstack([
        mo.md("**Robot HMM Parameters**"),
        mo.hstack([p_fwd_slider, obs_acc_slider]),
        mo.hstack([mo.md("Observations:"), y1_input, y3_input, y5_input]),
    ])
    return obs_acc_slider, p_fwd_slider, y1_input, y3_input, y5_input


@app.cell
def _(np, obs_acc_slider, p_fwd_slider, y1_input, y3_input, y5_input):
    def _compute_bp(p_fwd: float, obs_acc: float, y1: int, y3: int, y5: int) -> list[dict]:
        S = 6  # states 0..5

        # Transition matrix T[i,j] = P(X_next=j | X_cur=i)
        T = np.zeros((S, S))
        for i in range(S):
            if i < S - 1:
                T[i, i + 1] = p_fwd
                T[i, i]     = 1.0 - p_fwd
            else:
                T[i, i] = 1.0

        def obs_likelihood(y: int) -> np.ndarray:
            vec = np.zeros(S)
            vec[y] = obs_acc
            neighbors = [n for n in [y - 1, y + 1] if 0 <= n < S]
            for n in neighbors:
                vec[n] = (1.0 - obs_acc) / len(neighbors)
            return vec

        prior = np.zeros(S); prior[0] = 1.0

        # -- forward messages --
        # step 1: f_prior -> X0
        m_fprior_X0 = prior.copy()
        # step 2: Y1 -> f_obs1  (point mass)
        m_Y1_fobs1 = np.zeros(S); m_Y1_fobs1[y1] = 1.0
        # step 3: f_obs1 -> X1
        m_fobs1_X1 = obs_likelihood(y1)
        # step 4: X0 -> f_01
        m_X0_f01 = m_fprior_X0.copy()
        # step 5: f_01 -> X1  (T^T @ prior)
        m_f01_X1 = T.T @ m_X0_f01
        # step 6: X1 -> f_12  (combine prediction + obs)
        m_X1_f12 = m_f01_X1 * m_fobs1_X1
        # step 7: f_12 -> X2
        m_f12_X2 = T.T @ m_X1_f12
        # step 8: Y3 -> f_obs3
        m_Y3_fobs3 = np.zeros(S); m_Y3_fobs3[y3] = 1.0
        # step 9: f_obs3 -> X3
        m_fobs3_X3 = obs_likelihood(y3)
        # step 10: X2 -> f_23
        m_X2_f23 = m_f12_X2.copy()
        # step 11: f_23 -> X3
        m_f23_X3 = T.T @ m_X2_f23
        # step 12: X3 -> f_34  (combine prediction + obs)
        m_X3_f34 = m_f23_X3 * m_fobs3_X3
        # step 13: f_34 -> X4
        m_f34_X4 = T.T @ m_X3_f34
        # step 14: X4 -> f_45
        m_X4_f45 = m_f34_X4.copy()
        # step 15: Y5 -> f_obs5
        m_Y5_fobs5 = np.zeros(S); m_Y5_fobs5[y5] = 1.0
        # step 16: f_obs5 -> X5
        m_fobs5_X5 = obs_likelihood(y5)
        # step 17: f_45 -> X5
        m_f45_X5 = T.T @ m_X4_f45
        # step 18: final belief at X5
        belief_X5 = m_f45_X5 * m_fobs5_X5
        belief_X5 = belief_X5 / belief_X5.sum()

        def d(arr: np.ndarray) -> list[float]:
            return arr.tolist()

        schedule = [
            {"from": "f_prior", "to": "X0",     "dist": d(m_fprior_X0), "formula": "Prior: P(X₀=0)=1, all others 0."},
            {"from": "Y1",      "to": "f_obs1",  "dist": d(m_Y1_fobs1),  "formula": f"Point mass at y₁={y1}: observed value sent to obs factor."},
            {"from": "f_obs1",  "to": "X1",      "dist": d(m_fobs1_X1),  "formula": f"Likelihood P(Y₁={y1}|X₁=x): {obs_acc:.2f} at x={y1}, remaining mass split to neighbors ±1."},
            {"from": "X0",      "to": "f_01",    "dist": d(m_X0_f01),    "formula": "Variable-to-factor: X₀ forwards the prior message (only neighbor is f_prior)."},
            {"from": "f_01",    "to": "X1",      "dist": d(m_f01_X1),    "formula": "Factor-to-variable: Tᵀ @ μ(X₀→f₀₁). Forward prediction from X₀ through transition."},
            {"from": "X1",      "to": "f_12",    "dist": d(m_X1_f12),    "formula": "Variable-to-factor: μ(f₀₁→X₁) ⊙ μ(f_obs₁→X₁). Combines prediction + obs likelihood at t=1."},
            {"from": "f_12",    "to": "X2",      "dist": d(m_f12_X2),    "formula": "Factor-to-variable: Tᵀ @ μ(X₁→f₁₂). Forward prediction from X₁."},
            {"from": "Y3",      "to": "f_obs3",  "dist": d(m_Y3_fobs3),  "formula": f"Point mass at y₃={y3}: observed value sent to obs factor."},
            {"from": "f_obs3",  "to": "X3",      "dist": d(m_fobs3_X3),  "formula": f"Likelihood P(Y₃={y3}|X₃=x): {obs_acc:.2f} at x={y3}, remaining mass split to neighbors ±1."},
            {"from": "X2",      "to": "f_23",    "dist": d(m_X2_f23),    "formula": "Variable-to-factor: X₂ forwards the prediction from f₁₂ (no obs at t=2)."},
            {"from": "f_23",    "to": "X3",      "dist": d(m_f23_X3),    "formula": "Factor-to-variable: Tᵀ @ μ(X₂→f₂₃). Forward prediction from X₂."},
            {"from": "X3",      "to": "f_34",    "dist": d(m_X3_f34),    "formula": "Variable-to-factor: μ(f₂₃→X₃) ⊙ μ(f_obs₃→X₃). Combines prediction + obs likelihood at t=3."},
            {"from": "f_34",    "to": "X4",      "dist": d(m_f34_X4),    "formula": "Factor-to-variable: Tᵀ @ μ(X₃→f₃₄). Forward prediction from X₃."},
            {"from": "X4",      "to": "f_45",    "dist": d(m_X4_f45),    "formula": "Variable-to-factor: X₄ forwards the prediction from f₃₄ (no obs at t=4)."},
            {"from": "Y5",      "to": "f_obs5",  "dist": d(m_Y5_fobs5),  "formula": f"Point mass at y₅={y5}: observed value sent to obs factor."},
            {"from": "f_obs5",  "to": "X5",      "dist": d(m_fobs5_X5),  "formula": f"Likelihood P(Y₅={y5}|X₅=x): {obs_acc:.2f} at x={y5}, remaining mass split to neighbors ±1."},
            {"from": "f_45",    "to": "X5",      "dist": d(m_f45_X5),    "formula": "Factor-to-variable: Tᵀ @ μ(X₄→f₄₅). Forward prediction from X₄."},
            {"from": None,      "to": "X5",      "dist": d(belief_X5),   "formula": "Final belief: μ(f₄₅→X₅) ⊙ μ(f_obs₅→X₅), normalized. This is p(X₅|y₁,y₃,y₅)."},
        ]
        return schedule

    bp_schedule = _compute_bp(
        p_fwd_slider.value, obs_acc_slider.value,
        int(y1_input.value), int(y3_input.value), int(y5_input.value),
    )
    return (bp_schedule,)


@app.cell
def _(anywidget, traitlets):
    class BPRobotWidget(anywidget.AnyWidget):
        _esm = """
        const NS = "http://www.w3.org/2000/svg";

        // Fixed node positions
        const NODES = {
          f_prior: {x:45,  y:80,  kind:"factor", label:"f₀"},
          X0:      {x:115, y:80,  kind:"var",    label:"X₀"},
          f_01:    {x:185, y:80,  kind:"factor", label:"f₀₁"},
          X1:      {x:255, y:80,  kind:"var",    label:"X₁"},
          f_12:    {x:325, y:80,  kind:"factor", label:"f₁₂"},
          X2:      {x:395, y:80,  kind:"var",    label:"X₂"},
          f_23:    {x:465, y:80,  kind:"factor", label:"f₂₃"},
          X3:      {x:535, y:80,  kind:"var",    label:"X₃"},
          f_34:    {x:605, y:80,  kind:"factor", label:"f₃₄"},
          X4:      {x:675, y:80,  kind:"var",    label:"X₄"},
          f_45:    {x:745, y:80,  kind:"factor", label:"f₄₅"},
          X5:      {x:815, y:80,  kind:"var",    label:"X₅"},
          f_obs1:  {x:255, y:165, kind:"factor", label:"f_y₁"},
          f_obs3:  {x:535, y:165, kind:"factor", label:"f_y₃"},
          f_obs5:  {x:815, y:165, kind:"factor", label:"f_y₅"},
          Y1:      {x:255, y:250, kind:"obs",    label:"Y₁"},
          Y3:      {x:535, y:250, kind:"obs",    label:"Y₃"},
          Y5:      {x:815, y:250, kind:"obs",    label:"Y₅"},
        };

        const EDGES = [
          ["f_prior","X0"],["X0","f_01"],["f_01","X1"],["X1","f_12"],
          ["f_12","X2"],["X2","f_23"],["f_23","X3"],["X3","f_34"],
          ["f_34","X4"],["X4","f_45"],["f_45","X5"],
          ["X1","f_obs1"],["f_obs1","Y1"],
          ["X3","f_obs3"],["f_obs3","Y3"],
          ["X5","f_obs5"],["f_obs5","Y5"],
        ];

        function svgEl(tag, attrs) {
          const el = document.createElementNS(NS, tag);
          for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, v);
          return el;
        }

        function makeMarker(defs, id, color) {
          const m = svgEl("marker", {id, markerWidth:"8", markerHeight:"6", refX:"8", refY:"3", orient:"auto"});
          m.appendChild(svgEl("polygon", {points:"0 0, 8 3, 0 6", fill:color}));
          defs.appendChild(m);
        }

        function arrowPts(a, b) {
          const dx = b.x - a.x, dy = b.y - a.y;
          const len = Math.sqrt(dx*dx + dy*dy);
          // offset perpendicular to avoid overlap on same edge
          const ox = -dy/len * 4, oy = dx/len * 4;
          const rA = a.kind === "var" ? 16 : 12;
          const rB = b.kind === "var" ? 16 : 12;
          return {
            x1: a.x + ox + dx/len*rA, y1: a.y + oy + dy/len*rA,
            x2: b.x + ox - dx/len*rB, y2: b.y + oy - dy/len*rB,
          };
        }

        function msgLabel(step) {
          if (!step || step.from === null) return step ? `μ→${step.to}` : "";
          return `μ(${step.from}→${step.to})`;
        }

        function barChartSVG(dist) {
          const W = 360, H = 80, n = dist.length;
          const bw = 40, gap = (W - n*bw) / (n+1);
          const maxV = Math.max(...dist, 1e-9);
          const svg = svgEl("svg", {width:W, height:H, viewBox:`0 0 ${W} ${H}`});
          dist.forEach((v, i) => {
            const bh = Math.max(2, (v/maxV)*(H - 22));
            const x = gap + i*(bw+gap);
            const y = H - bh - 16;
            svg.appendChild(svgEl("rect", {x, y, width:bw, height:bh, fill:"#5b9bd5", rx:"3"}));
            // probability label above bar
            const pct = svgEl("text", {x: x+bw/2, y: y-2, "text-anchor":"middle", "font-size":"9", fill:"#555"});
            pct.textContent = v.toFixed(2);
            svg.appendChild(pct);
            // position label below bar
            const lbl = svgEl("text", {x: x+bw/2, y: H-2, "text-anchor":"middle", "font-size":"10", fill:"#333"});
            lbl.textContent = i;
            svg.appendChild(lbl);
          });
          return svg;
        }

        function render({ model, el }) {
          // --- SVG graph ---
          const svg = svgEl("svg", {width:"880", height:"300"});
          el.appendChild(svg);
          const defs = svgEl("defs", {});
          svg.appendChild(defs);
          makeMarker(defs, "arr-green",  "#27ae60");
          makeMarker(defs, "arr-orange", "#e67e22");

          // Draw static edges
          for (const [a, b] of EDGES) {
            const na = NODES[a], nb = NODES[b];
            svg.appendChild(svgEl("line", {
              x1:na.x, y1:na.y, x2:nb.x, y2:nb.y,
              stroke:"#ccc", "stroke-width":"1.5",
            }));
          }

          // Arrow layer (redrawn on update)
          const arrowLayer = svgEl("g", {});
          svg.appendChild(arrowLayer);

          // Draw static nodes on top
          for (const [id, n] of Object.entries(NODES)) {
            if (n.kind === "factor") {
              svg.appendChild(svgEl("rect", {
                x:n.x-11, y:n.y-11, width:22, height:22,
                fill:"#f0c040", stroke:"#b8860b", "stroke-width":"1.5", rx:"3",
              }));
            } else {
              const fill = n.kind === "obs" ? "#85c1e9" : "#5b9bd5";
              svg.appendChild(svgEl("circle", {cx:n.x, cy:n.y, r:"15", fill, stroke:"white", "stroke-width":"1.5"}));
            }
            const txt = svgEl("text", {
              x:n.x, y:n.y+4, "text-anchor":"middle",
              "font-size":"10", fill: n.kind==="factor" ? "#333" : "white",
              "pointer-events":"none",
            });
            txt.textContent = n.label;
            svg.appendChild(txt);
          }

          // --- Info panel ---
          const info = document.createElement("div");
          info.className = "bp-info";
          el.appendChild(info);

          // --- Controls ---
          const controls = document.createElement("div");
          controls.className = "bp-controls";
          const btnPrev = document.createElement("button");
          const btnNext = document.createElement("button");
          const stepLabel = document.createElement("span");
          btnPrev.textContent = "← Prev";
          btnNext.textContent = "Next →";
          controls.append(btnPrev, stepLabel, btnNext);
          el.appendChild(controls);

          function draw() {
            const schedule = model.get("schedule");
            const step = model.get("step");
            const total = schedule.length;

            stepLabel.textContent = `Step ${step} / ${total}`;
            btnPrev.disabled = step <= 0;
            btnNext.disabled = step >= total;

            // Redraw arrows
            while (arrowLayer.firstChild) arrowLayer.removeChild(arrowLayer.firstChild);
            for (let i = 0; i < step; i++) {
              const s = schedule[i];
              if (s.from === null) continue;
              const na = NODES[s.from], nb = NODES[s.to];
              if (!na || !nb) continue;
              const isActive = (i === step - 1);
              const color = isActive ? "#e67e22" : "#27ae60";
              const mid = isActive ? "arr-orange" : "arr-green";
              const pts = arrowPts(na, nb);
              const line = svgEl("line", {
                x1:pts.x1, y1:pts.y1, x2:pts.x2, y2:pts.y2,
                stroke:color, "stroke-width":"2", "marker-end":`url(#${mid})`,
              });
              arrowLayer.appendChild(line);
              // midpoint label
              const mx = (pts.x1+pts.x2)/2, my = (pts.y1+pts.y2)/2 - 5;
              const lbl = svgEl("text", {x:mx, y:my, "text-anchor":"middle", "font-size":"9", fill:color});
              lbl.textContent = msgLabel(s);
              arrowLayer.appendChild(lbl);
            }

            // Update info panel
            if (step === 0) {
              info.innerHTML = `<div class="bp-step-title">Step 0 / ${total} — Initial state</div><div class="bp-formula">Press Next to begin stepping through the forward-pass messages.</div>`;
            } else {
              const s = schedule[step - 1];
              const title = step === total
                ? `Step ${step}/${total}: Final belief at X₅`
                : `Step ${step}/${total}: ${s.from ?? ""}→${s.to}`;
              info.innerHTML = "";
              const titleEl = document.createElement("div");
              titleEl.className = "bp-step-title";
              titleEl.textContent = title;
              info.appendChild(titleEl);
              info.appendChild(barChartSVG(s.dist));
              const formula = document.createElement("div");
              formula.className = "bp-formula";
              formula.textContent = s.formula;
              info.appendChild(formula);
            }
          }

          btnPrev.addEventListener("click", () => {
            const s = model.get("step");
            if (s > 0) { model.set("step", s-1); model.save_changes(); }
          });
          btnNext.addEventListener("click", () => {
            const s = model.get("step");
            const total = model.get("schedule").length;
            if (s < total) { model.set("step", s+1); model.save_changes(); }
          });

          draw();
          model.on("change:step", draw);
          model.on("change:schedule", draw);
        }

        export default { render };
        """
        _css = """
        .bp-info { margin-top: 8px; padding: 8px 10px; border-radius: 6px; background: #f8f9fa; border: 1px solid #e0e0e0; font-size: 13px; }
        .bp-step-title { font-weight: 600; margin-bottom: 4px; }
        .bp-formula { margin-top: 6px; color: #555; font-size: 12px; }
        .bp-info svg { display: block; }
        .bp-controls { display: flex; align-items: center; gap: 10px; margin-top: 6px; font-size: 13px; }
        svg { display: block; }
        button { padding: 4px 10px; border-radius: 4px; border: 1px solid #ccc; cursor: pointer; background: #fff; }
        button:disabled { opacity: 0.4; cursor: default; }
        @media (prefers-color-scheme: dark) {
          .bp-info { background: #1e1e2e; border-color: #444; color: #eee; }
          .bp-formula { color: #aaa; }
          button { background: #2a2a3e; color: #eee; border-color: #555; }
        }
        """
        schedule = traitlets.List([]).tag(sync=True)
        step     = traitlets.Int(0).tag(sync=True)

    return (BPRobotWidget,)


@app.cell
def _(BPRobotWidget, bp_schedule, mo):
    bp_robot_widget = mo.ui.anywidget(BPRobotWidget(schedule=bp_schedule, step=0))
    mo.vstack([
        mo.md("**BP Forward Pass** — Step through the Sum-Product messages. Bar chart shows the distribution carried by each message."),
        bp_robot_widget,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Takeaways:**
    *   Identify the factorization implied by different graph structures (Factor Graphs, BNs, MRFs).
    *   Understand the role of the Partition Function $Z$.
    *   Relate graph separation (d-separation, Markov Blanket) to conditional independence.
    *   Recognize that computational complexity is governed by Treewidth.
    *   Master the Sum-Product message updates for tree-structured graphs.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Chapter 3: Information Theory in Inference

    **Goal:** Quantify uncertainty, information loss, and the asymptotic behavior of long sequences through entropy and divergence measures.

    Information theory provides the fundamental limits on compression, communication, and the inherent uncertainty in probability distributions.

    ### Shannon Entropy and Cross-Entropy
    The **Shannon Entropy** $H(X)$ quantifies the expected amount of "surprise" or uncertainty in a random variable:
    $$ H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x) $$
    It represents the lower bound on the average number of bits required to encode samples from $X$.

    If we encode data generated from a true distribution $p$ using a mismatched model distribution $q$, the expected message length is given by the **Cross-Entropy**:
    $$ H(p, q) = -\sum_x p(x) \log q(x) $$

    ### Kullback-Leibler Divergence
    The penalty paid for using the wrong distribution $q$ instead of $p$ is the **Kullback-Leibler (KL) Divergence**, also known as relative entropy:
    $$ D_{KL}(p \| q) = \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p) $$
    $D_{KL}$ is non-negative ($D_{KL}(p\|q) \ge 0$ with equality iff $p=q$) and acts as a statistical distance measure, though it is not a true metric because it is asymmetric and lacks the triangle inequality.

    ### Mutual Information
    **Mutual Information** $I(X; Y)$ measures the reduction in uncertainty about $X$ obtained by observing $Y$ (and vice versa). It quantifies the dependence between variables:
    $$ I(X; Y) = H(X) - H(X|Y) = \sum_{x, y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} $$
    $I(X;Y) \ge 0$, with equality if and only if $X$ and $Y$ are independent. The **Conditional Mutual Information** $I(X;Y|Z)$ measures the dependence remaining after observing $Z$.

    ### The Asymptotic Equipartition Property (AEP)
    The AEP is the information-theoretic analog of the Law of Large Numbers. It states that for a sequence of i.i.d. variables $(X_1, \dots, X_n) \sim p(x)$, the joint probability $p(X_1, \dots, X_n)$ is close to $2^{-nH(X)}$ with high probability.
    This defines the **Typical Set** $A_\epsilon^{(n)}$: the relatively small set of sequences that concentrate nearly all the probability mass as $n \to \infty$. The volume of the typical set is roughly $2^{nH(X)}$.
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md("**Planned Interactive Widget:** An interactive visualization of the Typical Set where users can drag a slider to increase N, watching the probability mass concentrate tightly into the typical set while the total volume of the state space explodes exponentially."),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Takeaways:**
    *   Define Entropy as the fundamental limit of compression.
    *   Use KL Divergence to measure the discrepancy between distributions.
    *   Interpret Mutual Information as the information shared between random variables.
    *   Understand how AEP guarantees that high-dimensional probability mass concentrates in a "typical" subset.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Chapter 4: Statistical Mechanics and Gibbs Measures

    **Goal:** Connect probabilistic graphical models to the physical principles of energy, temperature, and equilibrium in large interacting systems.

    Many concepts in graphical models originated in statistical physics, specifically in modeling the macroscopic properties of large systems of interacting particles.

    ### The Boltzmann Distribution
    In a physical system in thermal equilibrium with a heat bath, the probability of finding the system in a specific microscopic state or configuration $x$ is given by the **Boltzmann Distribution** (or Gibbs Measure):
    $$ \mu_\beta(x) = \frac{1}{Z(\beta)} e^{-\beta E(x)} $$
    where $E(x)$ is the **Energy** of state $x$, and $\beta = \frac{1}{k_B T}$ is the **Inverse Temperature**. $Z(\beta) = \sum_x e^{-\beta E(x)}$ is the partition function.

    - **High Temperature (Low $\beta$):** The distribution approaches uniformity. Entropy dominates, and the system easily transitions between states of vastly different energies.
    - **Low Temperature (High $\beta$):** The distribution concentrates sharply on the global minimum of the energy $E(x)$ (the ground states). The system "freezes".
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md("**Planned Interactive Widget:** An interactive 3D energy landscape where users can adjust the temperature slider and watch a simulated particle visually 'freeze' into the ground state or jump randomly between modes."),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    The connection to machine learning is profound: any factorized probability model $p(x) \propto \prod_a f_a(x)$ can be recast as a Boltzmann distribution by defining the energy as $E(x) = -\sum_a \log f_a(x)$ with $\beta=1$.

    ### Thermodynamics and Free Energy
    The macroscopic properties of the system are derivable from the partition function. The **Helmholtz Free Energy** is defined as $F = U - TS$, where $U = \mathbb{E}_\beta[E(x)]$ is the average internal energy, $T = \beta^{-1}$ is temperature, and $S$ is the thermodynamic entropy. Equivalently, $F = -\frac{1}{\beta} \log Z(\beta)$.
    The physical system naturally seeks to minimize its Free Energy, which is mathematically equivalent to maximizing the probability.

    ### The Ising Model
    The canonical example of a Markov Random Field is the **Ising Model** of ferromagnetism. Variables (spins) $s_i \in \{-1, +1\}$ are arranged on a lattice. The energy function includes pairwise interactions (couplings) $J_{ij}$ and external magnetic fields $h_i$:
    $$ E(\mathbf{s}) = -\sum_{\langle i,j \rangle} J_{ij} s_i s_j - \sum_i h_i s_i $$
    If $J_{ij} > 0$, neighboring spins prefer to align (ferromagnetic). For a 1D chain, the partition function can be computed exactly using a **Transfer Matrix**. In 2D and higher, the system exhibits critical phase transitions.

    **Interactive Demonstration:** Adjust the inverse temperature $\beta$ below to observe how the Boltzmann distribution over four hypothetical energy states shifts. At low $\beta$ (high temp), states are nearly equiprobable; at high $\beta$ (low temp), probability mass collapses onto the lowest energy state.

    **Takeaways:**
    *   Map probability distributions to energy-based Boltzmann measures.
    *   Understand the role of Inverse Temperature $\beta$ in controlling distribution "sharpness."
    *   Relate Partition Function $Z$ to Free Energy $F$.
    *   Recognize the Ising Model as a prototypical MRF on a lattice.
    """)
    return


@app.cell
def _(mo):
    beta_slider = mo.ui.slider(start=0.01, stop=5.0, step=0.1, value=1.0, label="Inverse Temperature (\\beta)")
    beta_slider
    return (beta_slider,)


@app.cell
def _(beta_slider, np, plt):
    _energies = np.array([1.0, 2.0, 4.0, 8.0])
    _beta = beta_slider.value

    _unnormalized = np.exp(-_beta * _energies)
    _probs = _unnormalized / np.sum(_unnormalized)

    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.bar(["State 1\n(Low E)", "State 2", "State 3", "State 4\n(High E)"], _probs, color='skyblue')
    _ax.set_ylabel("Probability")
    _ax.set_ylim(0, 1.05)
    _ax.set_title(f"Boltzmann Distribution (\\beta = {_beta:.2f})")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Chapter 5: Markov Chain Monte Carlo (MCMC)

    **Goal:** Develop stochastic sampling algorithms to approximate intractable expectations in high-dimensional state spaces where exact summation/integration is impossible.

    When exact inference is intractable (e.g., computing the partition function $Z$ involves an exponential sum), we turn to stochastic approximation. **MCMC** methods construct a Markov chain whose stationary distribution is the target distribution $p(x)$. By simulating the chain, we can draw dependent samples from $p(x)$ to estimate expectations $\mathbb{E}_p[f(x)] \approx \frac{1}{N} \sum_{t=1}^N f(x^{(t)})$.

    ### Markov Chains and Stationary Distributions
    A **Markov Chain** is a sequence of random variables where the future state depends only on the current state: $P(x^{(t+1)} | x^{(t)}, \dots, x^{(1)}) = T(x^{(t)} \to x^{(t+1)})$, where $T$ is the transition kernel.
    A distribution $\pi$ is **Stationary** (invariant) if applying the transition kernel leaves the distribution unchanged: $\sum_{x} \pi(x) T(x \to x') = \pi(x')$.

    A sufficient (though not necessary) condition for $\pi$ to be the unique stationary distribution is **Detailed Balance** (reversibility):
    $$ \pi(x) T(x \to x') = \pi(x') T(x' \to x) $$
    If the chain is also **Irreducible** (can reach any state from any other state) and **Aperiodic** (does not cycle deterministically), it is guaranteed to converge to $\pi$.

    ### The Metropolis-Hastings Algorithm
    To sample from $p(x) = \frac{1}{Z} \tilde{p}(x)$, Metropolis-Hastings proposes a move from $x$ to $x'$ via a proposal distribution $Q(x' | x)$. The move is accepted with probability:
    $$ A(x \to x') = \min \left( 1, \frac{\tilde{p}(x') Q(x | x')}{\tilde{p}(x) Q(x' | x)} \right) $$
    Because $Z$ cancels out in the ratio $\frac{\tilde{p}(x')}{\tilde{p}(x)}$, this method works without knowing the normalizer.

    ### Gibbs Sampling and Glauber Dynamics
    **Gibbs Sampling** is a special case of Metropolis-Hastings where variables are updated one at a time by sampling from their conditional distribution given all other variables: $x_i^{(t+1)} \sim p(x_i | \mathbf{x}_{\setminus i}^{(t)})$. The acceptance probability is always 1.
    For binary variables like the Ising model, this is often called **Glauber Dynamics**.

    ### Mixing Time and Critical Slowing Down
    The **Mixing Time** is the number of steps required for the chain to approach the stationary distribution (the **Burn-in** period). Near phase transitions (e.g., the critical temperature of the 2D Ising model), local updates like Gibbs sampling struggle to flip large clusters of aligned spins. This exponential increase in mixing time is known as **Critical Slowing Down**.

    **Interactive Simulation:** The cell below runs Glauber dynamics on a 2D Ising model. Notice that around the critical inverse temperature $\beta_c \approx 0.44$, the system exhibits large, slow-moving clusters—a visual manifestation of critical slowing down.
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md("**Planned Interactive Widget:** A high-performance interactive WebGL/Canvas implementation of the 2D Ising model where users can draw on the grid, perturb spins directly, and watch the Glauber dynamics heal the perturbation in real-time, visualizing mixing time and correlation lengths dynamically."),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Takeaways:**
    *   State the conditions for a Markov chain to converge to a stationary distribution.
    *   Master the Metropolis-Hastings acceptance ratio derivation.
    *   Implement Gibbs Sampling for models with tractable local conditionals.
    *   Understand the concepts of Mixing Time, Burn-in, and the impact of Critical Slowing Down.
    """)
    return


@app.cell
def _(mo):
    ising_beta_slider = mo.ui.slider(start=0.05, stop=1.0, step=0.05, value=0.44, label="Inverse Temperature (\\beta)")
    ising_J_slider = mo.ui.slider(start=-1.0, stop=2.0, step=0.1, value=1.0, label="Coupling (J)")
    ising_h_slider = mo.ui.slider(start=-1.0, stop=1.0, step=0.1, value=0.0, label="External Field (h)")
    sweep_button = mo.ui.button(label="Run 50 MCMC Sweeps")

    mo.vstack([
        mo.md("**2D Ising Model MCMC Simulator:**"),
        ising_beta_slider,
        ising_J_slider,
        ising_h_slider,
        sweep_button
    ])
    return ising_J_slider, ising_beta_slider, ising_h_slider, sweep_button


@app.cell
def _(mo):
    get_grid, set_grid = mo.state(None)
    return get_grid, set_grid


@app.cell
def _(
    get_grid,
    ising_J_slider,
    ising_beta_slider,
    ising_h_slider,
    mo,
    np,
    plt,
    set_grid,
    sweep_button,
):
    _L = 32
    _beta = ising_beta_slider.value
    _J = ising_J_slider.value
    _h = ising_h_slider.value

    if get_grid() is None:
        _initial_grid = np.random.choice([-1, 1], size=(_L, _L))
        set_grid(_initial_grid)
        _grid = _initial_grid
    else:
        _grid = get_grid().copy()

    _sweeps = 50 if sweep_button.value else 0
    _is_script_mode = mo.app_meta().mode == "script"
    if _is_script_mode:
        _sweeps = 5

    for _ in range(_sweeps):
        for _k in range(_L * _L):
            _i, _j = np.random.randint(0, _L), np.random.randint(0, _L)
            _neighbor_sum = _grid[(_i-1)%_L, _j] + _grid[(_i+1)%_L, _j] + _grid[_i, (_j-1)%_L] + _grid[_i, (_j+1)%_L]
            _heff = _h + _J * _neighbor_sum
            # Glauber dynamics update probability
            _p_plus = 1.0 / (1.0 + np.exp(-2.0 * _beta * _heff))
            _grid[_i, _j] = 1 if np.random.rand() < _p_plus else -1

    if _sweeps > 0 and not _is_script_mode:
         set_grid(_grid)

    _fig2, _ax2 = plt.subplots(figsize=(5, 5))
    _ax2.imshow(_grid, cmap='coolwarm', vmin=-1, vmax=1)
    _ax2.set_title(f"Spin Lattice Configuration (\\beta = {_beta:.2f})")
    _ax2.axis('off')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Chapter 6: Parameter Learning and the EM Algorithm

    **Goal:** Derive methods for estimating model parameters from empirical data, particularly when dealing with hidden variables or incomplete datasets.

    Given a parameterized family of distributions $p_\theta(x)$ and an empirical dataset $\mathcal{D} = \{x^{(1)}, \dots, x^{(M)}\}$, the goal of learning is to find the parameters $\theta$ that best explain the data.

    ### Maximum Likelihood and Exponential Families
    The standard objective is to maximize the **Log-Likelihood** of the data:
    $$ \ell(\theta) = \frac{1}{M} \sum_{m=1}^M \log p_\theta(x^{(m)}) $$
    Many graphical models belong to the **Exponential Family**, formulated as:
    $$ p_\theta(x) = \exp \left\{ \langle \theta, \phi(x) \rangle - A(\theta) \right\} $$
    where $\theta$ are the natural parameters, $\phi(x)$ are the **Sufficient Statistics**, and $A(\theta) = \log \sum_x \exp(\langle \theta, \phi(x) \rangle)$ is the log-partition function.

    The gradient of the log-likelihood for an exponential family has a remarkably intuitive form: it is the difference between the **Empirical Expected Sufficient Statistics** (Data Moments) and the **Model Expected Sufficient Statistics** (Model Moments):
    $$ \nabla_\theta \ell(\theta) = \mathbb{E}_{\mathcal{D}}[\phi(x)] - \mathbb{E}_{p_\theta}[\phi(x)] $$
    At the optimum (the **Maximum Likelihood Estimator**, MLE), the model moments exactly match the data moments (**Moment Matching**). Computing the model moments requires performing inference, making *inference the inner loop of learning*.

    ### Learning with Hidden Variables: The EM Algorithm
    When datasets have missing values or latent variables $Z$, the marginal likelihood $\log \sum_Z p_\theta(X, Z)$ is no longer convex, and direct maximization is difficult. The **Expectation-Maximization (EM)** algorithm is a principled iterative procedure to maximize a lower bound on the marginal likelihood.

    EM alternates between two steps:
    1. **E-step (Expectation):** Fix parameters $\theta^{(t)}$. Compute the posterior distribution over the hidden variables $q(Z) = p_{\theta^{(t)}}(Z | X)$. Calculate the expected complete-data log-likelihood under this posterior:
       $$ Q(\theta, \theta^{(t)}) = \mathbb{E}_{q(Z)} [ \log p_\theta(X, Z) ] $$
       In discrete models, this involves computing expected sufficient statistics using "Soft Counts" derived from inference (e.g., via Belief Propagation).
    2. **M-step (Maximization):** Update parameters to maximize the expected log-likelihood:
       $$ \theta^{(t+1)} = \arg\max_\theta Q(\theta, \theta^{(t)}) $$

    For Hidden Markov Models (HMMs), the E-step is performed efficiently using the Forward-Backward algorithm, and the overall EM procedure is known as the **Baum-Welch algorithm**.
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md("**Planned Interactive Widget:** An interactive Gaussian Mixture Model (GMM) visualizer where users can drag data points and see the EM algorithm animate step-by-step: first the E-step updating the soft color assignments (posteriors), followed by the M-step moving the Gaussian ellipses to match the new assignments."),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Takeaways:**
    *   Explain why MLE for exponential families leads to moment matching.
    *   Understand that inference is required to compute gradients for learning.
    *   Decompose EM into calculating latent posteriors (E-step) and optimizing parameters (M-step).
    *   Recognize HMM learning (Baum-Welch) as a specific application of EM.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Chapter 7: MAP Inference and Variational Methods

    **Goal:** Formulate inference as an optimization problem over distributions and explore principled relaxations (like the Bethe approximation) for graphs with cycles.

    Instead of computing marginal probabilities, we often want to find the single most likely joint assignment of variables. This is **Maximum A Posteriori (MAP)** inference:
    $$ \mathbf{x}^* = \arg\max_{\mathbf{x}} p(\mathbf{x}) = \arg\max_{\mathbf{x}} \prod_{a} f_a(\mathbf{x}_{\partial a}) $$
    MAP inference is a discrete combinatorial optimization problem. While exact on trees via Max-Product message passing, it is NP-hard for general graphs.

    ### The Marginal Polytope and LP Relaxations
    We can recast MAP inference as a linear optimization problem over distributions. Let $\mu$ be a valid joint distribution over $\mathbf{x}$, and let $\mu_a(\mathbf{x}_{\partial a})$ be its local marginals. The set of all globally consistent local marginals forms the **Marginal Polytope** $\mathcal{M}$. MAP inference is exactly:
    $$ \max_{\mu \in \mathcal{M}} \sum_a \sum_{\mathbf{x}_{\partial a}} \mu_a(\mathbf{x}_{\partial a}) \log f_a(\mathbf{x}_{\partial a}) $$
    Because characterizing $\mathcal{M}$ requires an exponential number of constraints, we relax the problem. We optimize over the **Local Polytope** $\mathcal{L} \supset \mathcal{M}$, which only enforces **Local Consistency** (e.g., pairwise marginals agree with univariate marginals). This is the **LP Relaxation** of MAP inference. On trees, $\mathcal{L} = \mathcal{M}$ (the relaxation is tight). On loopy graphs, optimizing over $\mathcal{L}$ yields "Pseudo-Marginals" that provide upper bounds on the true probability.

    ### Variational Inference and the Bethe Approximation
    The partition function $Z$ can be expressed as a variational optimization problem over all valid distributions $q$:
    $$ \log Z = \max_{q \in \mathcal{M}} \left( \mathbb{E}_q[\log \prod f_a] + H(q) \right) $$
    The term maximized is the negative **Gibbs Free Energy**. Computing the exact entropy $H(q)$ is intractable on loopy graphs.

    The **Bethe Approximation** replaces the true entropy with a local approximation based on node and edge marginals, and replaces the marginal polytope $\mathcal{M}$ with the local polytope $\mathcal{L}$. This yields the **Bethe Free Energy**.

    **Theorem (Yedidia, Freeman, Weiss 2001):** The fixed points of Loopy Belief Propagation correspond exactly to the stationary points of the Bethe Free Energy constrained to the local polytope.
    This seminal result bridges statistical physics and message-passing algorithms, proving that Loopy BP is not merely an ad-hoc heuristic, but a principled algorithm for performing approximate variational inference.
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md("**Planned Interactive Widget:** An interactive 3D visualization of the Marginal Polytope versus the Local Polytope for a small 3-node cycle graph, allowing users to rotate the space and visually understand where the LP relaxation produces invalid pseudo-marginals outside the true marginal polytope."),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Takeaways:**
    *   Formulate MAP inference as a Linear Program (LP).
    *   Distinguish between the Marginal Polytope (hard) and the Local Polytope (easy relaxation).
    *   Understand the variational characterization of the Partition Function.
    *   Relate Loopy BP fixed points to the Bethe Free Energy stationary points.
    """)
    return


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.0",
#     "numpy>=1.24.0",
#     "scipy>=1.10.0",
#     "matplotlib>=3.7.0",
#     "networkx>=3.0",
#     "opt-einsum>=3.3.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import scipy
    import matplotlib.pyplot as plt
    import networkx as nx
    import opt_einsum as oe
    from itertools import combinations, product
    from collections import defaultdict
    from typing import Optional
    import time

    return np, nx, plt, scipy, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Beyond Belief Propagation: Cluster-Corrected Tensor Network Contraction with Exponential Convergence

    **Paper:** [arXiv:2510.02290](https://arxiv.org/abs/2510.02290)
    **Authors:** Siddhant Midha (Princeton Quantum Initiative) & Yifan F. Zhang (Dept. of ECE, Princeton University)

    ---

    This notebook is a self-contained, interactive explainer and reimplementation of the
    **cluster-corrected tensor network contraction** method. Starting from the well-known
    Belief Propagation (BP) algorithm, we build up the loop series expansion and then derive
    the cluster expansion that provably converges exponentially fast — the main theoretical
    contribution of the paper.

    **Outline:**
    1. **Motivation & Background** — why tensor network contraction matters
    2. **Belief Propagation for Tensor Networks** — messages, fixed points, the BP vacuum
    3. **The Loop Series Expansion** — exact correction to BP via generalized loops
    4. **The Cluster Expansion** — taking the logarithm to get exponential convergence
    5. **Algorithm & Implementation** — modular code for the full pipeline
    6. **Numerical Results** — 2D Ising model benchmarks reproducing the paper's key figures
    7. **Extensions & Discussion** — QEC, limitations, open questions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part I: Motivation and Background

    ### Why Tensor Network Contraction Matters

    A **tensor network (TN)** is a collection of tensors $\{T_v\}_{v \in V}$ defined on a graph
    $G = (V, E)$. Each edge $e = (v, w)$ carries a **bond Hilbert space** $\mathcal{B}_{vw}$
    of dimension $\chi$ (the **bond dimension**). **Contraction** means summing over all
    internal (bond) indices to produce a scalar:

    $$
    \mathcal{Z}(\mathcal{T}) = \sum_{\text{all bond indices}} \prod_{v \in V} (T_v)_{i_1 i_2 \cdots}
    $$

    This operation is central to:

    - **Quantum simulation** — computing expectation values in PEPS / PEPO states
    - **Quantum error correction** — decoding surface codes and LDPC codes
    - **Statistical mechanics** — evaluating partition functions of classical lattice models
    - **Classical simulation of quantum circuits** — simulating Sycamore-class circuits

    **The problem:** exact contraction is **#P-hard** in general, with cost exponential in system
    size. This motivates polynomial-time approximate algorithms — and that is precisely what
    the cluster expansion provides.
    """)
    return


@app.cell
def _(mo):
    chi_slider = mo.ui.slider(2, 8, value=2, label="Bond dimension χ")
    mo.md(f"**Bond dimension slider:** {chi_slider}")
    return (chi_slider,)


@app.cell
def _(chi_slider, nx, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(5, 4))
    _G = nx.grid_2d_graph(3, 3)
    _pos = nx.nx_agraph.graphviz_layout(_G, prog="neato")
    nx.draw(
        _G,
        _pos,
        ax=_ax,
        with_labels=False,
        node_color="steelblue",
        node_size=400,
        edge_color="gray",
        width=2,
    )
    _chi = chi_slider.value
    _ax.set_title(
        f"3×3 square lattice TN  |  χ = {_chi}\n"
        f"Exact contraction cost: O(χ^{2*3}) = O({_chi**(2*3):.0f})",
        fontsize=11,
    )
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Landscape of Existing Algorithms

    | Family | Complexity | Exact? | Limitations |
    |--------|-----------|--------|-------------|
    | **MPS contraction** (DMRG, TEBD) | $O(N \chi^3)$ | Yes (1D) | Exponential cost for 2D+ |
    | **CTMRG / boundary MPS** | $O(N \chi^6)$ | Approximate | Bond dimension truncation |
    | **Monte Carlo sampling** | Stochastic | Approximate | Sign problem for quantum |
    | **Belief Propagation** | $O(N \chi^3)$ per iter | Exact on trees | Approximate on loopy graphs |
    | **This paper: BP + Cluster Expansion** | $O(N \cdot e^{O(M)})$ | Approximate | Provable exponential convergence! |

    BP is computationally efficient but its accuracy on loopy graphs is poorly understood.
    The **cluster expansion** systematically improves BP with **provable exponential convergence**
    — the main contribution of the paper.

    ### Statistical Mechanics Perspective

    The tensor network contraction computes a **partition function**:
    $$\mathcal{Z}(\mathcal{T}) = \text{tr}(\text{contract all tensors})$$

    The **free energy** $\mathcal{F} = -\ln \mathcal{Z}$ is extensive: $\mathcal{F} \propto N$.
    This extensivity is the key physical insight: working with $\ln \mathcal{Z}$ instead of
    $\mathcal{Z}$ eliminates the combinatorial explosion of disconnected contributions.
    """)
    return


@app.cell
def _(np):
    # Small example: 2D Ising partition function by brute force
    def ising_partition_brute_force(beta, Lx, Ly, J=1.0):
        """Compute Z for Ising model on Lx x Ly grid by exhaustive enumeration."""
        N = Lx * Ly
        Z = 0.0
        for config in range(2**N):
            spins = np.array(
                [2 * ((config >> i) & 1) - 1 for i in range(N)]
            ).reshape(Lx, Ly)
            energy = 0.0
            for i in range(Lx):
                for j in range(Ly):
                    if j + 1 < Ly:
                        energy -= J * spins[i, j] * spins[i, j + 1]
                    if i + 1 < Lx:
                        energy -= J * spins[i, j] * spins[i + 1, j]
            Z += np.exp(-beta * energy)
        return Z

    # Test: 3x3 lattice
    _beta_test = 0.4
    _Z = ising_partition_brute_force(_beta_test, 3, 3)
    print(
        f"2D Ising 3×3 at β={_beta_test}: Z = {_Z:.6f}, "
        f"f = F/N = {-np.log(_Z)/9:.6f}"
    )
    return (ising_partition_brute_force,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part II: Belief Propagation for Tensor Networks

    ### Tensor Network Setup

    We work with a graph $G = (V, E)$ with $N = |V|$ vertices. At each vertex $v$,
    there is a tensor $T_v$ whose legs correspond to the edges incident on $v$.
    Each edge $(v, w)$ carries a bond space of dimension $\chi$. When all bond indices
    are summed, the result is a scalar $\mathcal{Z}$.

    For the 2D Ising model, each vertex has degree 4 (on the bulk of the square lattice)
    and $\chi = 2$ (classical spin).
    """)
    return


@app.cell
def _(np, nx):
    class TensorNetwork:
        """A tensor network on a graph with uniform bond dimension."""

        def __init__(self, graph: nx.Graph, chi: int):
            self.graph = graph
            self.chi = chi
            self.nodes = list(graph.nodes())
            self.edges = list(graph.edges())
            self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
            self.tensors = {}  # node -> numpy array

        def set_tensor(self, node, tensor):
            """Set the tensor at a node. Legs ordered by sorted neighbors."""
            self.tensors[node] = tensor

        def neighbors(self, node):
            """Return sorted neighbors of node."""
            return sorted(self.graph.neighbors(node))

        def contract_exact(self):
            """Brute-force contraction by summing over all bond indices."""
            # Build index labels for opt_einsum
            edge_to_idx = {}
            idx = 0
            for e in self.edges:
                edge_to_idx[e] = idx
                edge_to_idx[(e[1], e[0])] = idx
                idx += 1

            # For each node, build its index list (ordered by sorted neighbors)
            operands = []
            subscripts_list = []
            for node in self.nodes:
                nbrs = self.neighbors(node)
                subs = []
                for nbr in nbrs:
                    subs.append(edge_to_idx[(node, nbr)])
                subscripts_list.append(subs)
                operands.append(self.tensors[node])

            # Build einsum string
            # Use letters a-z, then A-Z for indices
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            input_subs = []
            for subs in subscripts_list:
                input_subs.append("".join(chars[s] for s in subs))
            einsum_str = ",".join(input_subs) + "->"

            return np.einsum(einsum_str, *operands)

    return (TensorNetwork,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The BP Algorithm: Messages and Fixed Points

    **Message tensors** $\mu_{v \to w} \in \mathbb{R}^\chi$ are vectors sent from node $v$
    to neighbor $w$ along edge $(v, w)$.

    The **BP fixed-point condition** says: contracting tensor $T_v$ with all incoming
    messages *except* from $w$ yields the outgoing message $\mu_{v \to w}$:

    $$
    \mu_{v \to w} = \text{contract}\left(T_v,\ \{\mu_{u \to v}\}_{u \in \partial v \setminus w}\right)
    $$

    Messages are **normalized** so that $\mu_{v \to w}^T \mu_{w \to v} = 1$.

    **Iterative algorithm:**
    1. Initialize messages randomly
    2. Update each message using the BP equation
    3. Damping: $\mu^{\text{new}} \leftarrow (1-\alpha)\mu^{\text{old}} + \alpha\,\mu^{\text{update}}$
    4. Normalize, repeat until convergence
    """)
    return


@app.cell
def _(np):
    def belief_propagation(
        tn, max_iter=500, tol=1e-10, damping=0.3, verbose=False
    ):
        """
        Run belief propagation on a tensor network.

        Returns:
            messages: dict of (v, w) -> message vector (numpy array of shape (chi,))
            converged: bool
            history: list of max message changes per iteration
        """
        chi = tn.chi
        graph = tn.graph

        # Initialize messages randomly (positive, normalized)
        messages = {}
        for u, v in graph.edges():
            messages[(u, v)] = np.random.rand(chi) + 0.1
            messages[(v, u)] = np.random.rand(chi) + 0.1

        # Normalize message pairs: mu_{v->w}^T mu_{w->v} = 1
        def normalize_pair(u, v):
            ip = messages[(u, v)] @ messages[(v, u)]
            if ip > 0:
                messages[(u, v)] /= np.sqrt(ip)
                messages[(v, u)] /= np.sqrt(ip)

        for u, v in graph.edges():
            normalize_pair(u, v)

        history = []

        for iteration in range(max_iter):
            max_change = 0.0
            new_messages = {}

            for u, v in list(graph.edges()) + [(v, u) for u, v in graph.edges()]:
                # Compute BP update for message u -> v
                # Contract T_u with all incoming messages except from v
                nbrs_u = tn.neighbors(u)
                T = tn.tensors[u]
                deg = len(nbrs_u)
                v_leg_idx = nbrs_u.index(v)

                # Use einsum: contract all legs except v_leg_idx
                # Build vectors to contract with
                vectors = []
                for leg_idx, w in enumerate(nbrs_u):
                    if w == v:
                        vectors.append(None)  # free leg
                    else:
                        vectors.append(messages[(w, u)])

                # Contract from the last axis backward to avoid index shift
                result = T
                contracted = 0
                for leg_idx in range(deg - 1, -1, -1):
                    if vectors[leg_idx] is not None:
                        result = np.tensordot(
                            result, vectors[leg_idx],
                            axes=([leg_idx], [0])
                        )

                new_msg = result.flatten()

                # Normalize to unit norm for stability
                norm = np.linalg.norm(new_msg)
                if norm > 0:
                    new_msg /= norm

                new_messages[(u, v)] = new_msg

            # Apply damping and update
            for key in new_messages:
                old = messages[key]
                new = new_messages[key]
                # Handle sign ambiguity: align signs
                if old @ new < 0:
                    new = -new
                updated = (1 - damping) * old + damping * new
                norm = np.linalg.norm(updated)
                if norm > 0:
                    updated /= norm
                change = np.linalg.norm(updated - old)
                max_change = max(max_change, change)
                messages[key] = updated

            # Re-normalize pairs
            for u, v in graph.edges():
                normalize_pair(u, v)

            history.append(max_change)
            if verbose and iteration % 50 == 0:
                print(f"  BP iter {iteration}: max_change = {max_change:.2e}")

            if max_change < tol:
                if verbose:
                    print(f"  BP converged at iteration {iteration}")
                return messages, True, history

        if verbose:
            print(f"  BP did not converge after {max_iter} iterations (last change: {max_change:.2e})")
        return messages, max_change < tol * 100, history

    return (belief_propagation,)


@app.cell
def _(np):
    def bp_contract_vertex(tn, messages, node):
        """Contract tensor at node with all incoming messages. Returns scalar z_v."""
        nbrs = tn.neighbors(node)
        T = tn.tensors[node]
        for leg_idx, w in enumerate(nbrs):
            T = np.tensordot(T, messages[(w, node)], axes=([0], [0]))
        return float(T)

    def compute_bp_partition(tn, messages):
        """
        Compute BP vacuum partition function:
        Z_BP = prod_v z_v / prod_{(v,w)} z_{vw}
        where z_v = contract(T_v, all incoming messages)
        and z_{vw} = mu_{v->w}^T mu_{w->v} (should be 1 after normalization)
        """
        log_Z = 0.0

        # Vertex contributions
        for node in tn.nodes:
            z_v = bp_contract_vertex(tn, messages, node)
            if z_v > 0:
                log_Z += np.log(z_v)
            else:
                log_Z += np.log(abs(z_v))  # handle sign

        # Edge contributions (subtract)
        for u, v in tn.edges:
            z_uv = messages[(u, v)] @ messages[(v, u)]
            if abs(z_uv) > 0:
                log_Z -= np.log(abs(z_uv))

        return log_Z  # returns log(Z_BP)

    return (compute_bp_partition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### BP Vacuum: The Zeroth-Order Approximation

    The **BP vacuum partition function** is computed from converged messages:

    $$
    \mathcal{Z}_{\text{BP}} = \frac{\prod_{v \in V} z_v}{\prod_{(v,w) \in E} z_{vw}}
    $$

    where $z_v = \text{contract}(T_v, \text{all incoming messages})$ and
    $z_{vw} = \mu_{v \to w}^T \mu_{w \to v} = 1$ (by normalization convention).

    The BP vacuum free energy $\mathcal{F}_{\text{BP}} = -\ln \mathcal{Z}_{\text{BP}}$ equals
    the **Bethe free energy** — exact on trees, approximate on loopy graphs.

    ### Where BP Fails: The Role of Loops

    BP is exact on **tree graphs** because there are no loops. On graphs with loops (like
    2D lattices), BP "double-counts" correlations propagated around cycles. The error comes
    from **neglecting loop contributions**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part III: The 2D Ising Model as a Tensor Network

    The 2D Ising model on a square lattice has Hamiltonian:
    $$H = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j, \qquad \sigma_i \in \{+1, -1\}$$

    **Tensor network representation:** place a rank-$d$ tensor at each vertex (where $d$ is
    the vertex degree). Each bond carries dimension $\chi = 2$.

    The tensor entries encode the Boltzmann weights. For a vertex with neighbors, we use:
    $$T_{i_1 i_2 \cdots i_d} = \sum_{\sigma = \pm 1} \prod_{k=1}^{d} \sqrt{W_{\sigma, i_k}}$$
    where $W$ is the $2 \times 2$ edge weight matrix $W_{ab} = e^{\beta J (2\delta_{ab} - 1)}$.
    """)
    return


@app.cell
def _(TensorNetwork, np, nx, scipy):
    def make_ising_tn(Lx, Ly, beta, J=1.0):
        """
        Create tensor network for 2D Ising model on Lx x Ly square lattice.
        Uses the standard decomposition with bond dimension chi=2.
        """
        G = nx.grid_2d_graph(Lx, Ly)
        tn = TensorNetwork(G, chi=2)

        # Edge weight matrix: W[a,b] = exp(beta * J * (2*delta_{a,b} - 1))
        W = np.array([
            [np.exp(beta * J), np.exp(-beta * J)],
            [np.exp(-beta * J), np.exp(beta * J)],
        ])
        # Take matrix square root for symmetric decomposition
        sqrtW = scipy.linalg.sqrtm(W).real

        for node in G.nodes():
            nbrs = sorted(G.neighbors(node))
            deg = len(nbrs)
            # T_{i1, i2, ..., id} = sum_sigma sqrt(W)_{sigma, i1} * ... * sqrt(W)_{sigma, id}
            shape = tuple([2] * deg)
            T = np.zeros(shape)
            for sigma in range(2):  # sigma index
                contrib = np.ones(1)
                for _ in range(deg):
                    contrib = np.outer(contrib, sqrtW[sigma, :]).flatten()
                T += contrib.reshape(shape)
            tn.set_tensor(node, T)

        return tn

    return (make_ising_tn,)


@app.cell
def _(np, scipy):
    def onsager_free_energy(beta, J=1.0):
        """
        Onsager's exact free energy per site for the 2D Ising model
        on an infinite square lattice.
        """
        k = 1.0 / (np.sinh(2 * beta * J) ** 2)

        def integrand(theta1, theta2):
            return np.log(
                np.cosh(2 * beta * J) ** 2
                - np.sinh(2 * beta * J) * (np.cos(theta1) + np.cos(theta2))
            )

        result, _ = scipy.integrate.dblquad(
            integrand, 0, np.pi, 0, np.pi
        )
        f = -(np.log(2) + result / (2 * np.pi**2)) / beta
        return f

    return (onsager_free_energy,)


@app.cell
def _(mo, np, onsager_free_energy, plt):
    # Plot exact Ising free energy and specific heat
    _betas = np.linspace(0.05, 1.0, 100)
    _beta_c = 0.5 * np.log(1 + np.sqrt(2))

    _f_exact = [onsager_free_energy(b) for b in _betas]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))

    _ax1.plot(_betas, _f_exact, "b-", linewidth=2)
    _ax1.axvline(_beta_c, color="r", linestyle="--", alpha=0.7, label=f"βc ≈ {_beta_c:.4f}")
    _ax1.set_xlabel("β")
    _ax1.set_ylabel("f (free energy per site)")
    _ax1.set_title("Onsager Exact Free Energy")
    _ax1.legend()

    # Specific heat from numerical differentiation
    _f_arr = np.array(_f_exact)
    _db = _betas[1] - _betas[0]
    _c = -_betas[1:-1] ** 2 * np.gradient(np.gradient(_f_arr, _db), _db)[1:-1]
    _ax2.plot(_betas[1:-1], _c, "r-", linewidth=2)
    _ax2.axvline(_beta_c, color="r", linestyle="--", alpha=0.7)
    _ax2.set_xlabel("β")
    _ax2.set_ylabel("C (specific heat)")
    _ax2.set_title("Specific Heat (diverges at βc)")

    plt.tight_layout()
    mo.md("**Onsager's exact solution** for the 2D Ising model on an infinite square lattice:")
    _fig
    return


@app.cell
def _(
    belief_propagation,
    compute_bp_partition,
    make_ising_tn,
    mo,
    np,
    onsager_free_energy,
    plt,
):
    # Run BP on Ising models and compare with exact
    _beta_c = 0.5 * np.log(1 + np.sqrt(2))
    _betas = np.linspace(0.1, 0.8, 15)
    _L = 5  # lattice size

    _f_bp = []
    _f_exact_finite = []

    for _b in _betas:
        _tn = make_ising_tn(_L, _L, _b)
        _msgs, _conv, _ = belief_propagation(_tn, max_iter=300, damping=0.3)
        _logZ_bp = compute_bp_partition(_tn, _msgs)
        _f_bp.append(-_logZ_bp / (_L * _L))

    _f_onsager = [onsager_free_energy(b) for b in _betas]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))

    _ax1.plot(_betas, _f_onsager, "b-", linewidth=2, label="Onsager (L→∞)")
    _ax1.plot(_betas, _f_bp, "ro--", markersize=4, label=f"BP (L={_L})")
    _ax1.axvline(_beta_c, color="gray", linestyle="--", alpha=0.5)
    _ax1.set_xlabel("β")
    _ax1.set_ylabel("f (free energy per site)")
    _ax1.set_title("BP vs Exact Free Energy")
    _ax1.legend()

    _errors = [abs(a - b) for a, b in zip(_f_bp, _f_onsager)]
    _ax2.semilogy(_betas, _errors, "ro-", markersize=4)
    _ax2.axvline(_beta_c, color="gray", linestyle="--", alpha=0.5)
    _ax2.set_xlabel("β")
    _ax2.set_ylabel("|f_BP - f_exact|")
    _ax2.set_title("BP Error vs Temperature")

    plt.tight_layout()
    mo.md(f"**BP on the {_L}×{_L} Ising model:** accurate at high T (small β), degrades near βc.")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part IV: The Loop Series Expansion

    ### Generalized Loops

    At each edge, decompose the bond space into the **BP ground state** $|0\rangle$
    (defined by the converged BP messages) and **excited states** $|s\rangle$ for $s = 1, \ldots, \chi - 1$.

    An edge is **excited** if it carries an excited state. A **generalized loop** is a subgraph
    $C = (W, F)$ where every vertex has **even degree** — i.e., each vertex is touched by an
    even number of excited edges. This includes simple cycles, figure-eights, and disconnected
    unions of cycles.

    ### The Exact Loop Series

    **Lemma (Loop Series Expansion):**
    $$
    \mathcal{Z}(\mathcal{T}) = \mathcal{Z}_{\text{BP}} \cdot \sum_{l \in \mathcal{L}_G} Z_l
    $$

    where $\mathcal{L}_G$ is the set of all generalized loops (including the empty loop $Z_\emptyset = 1$),
    and $Z_l$ is the **loop tensor** — the contraction of excited-state components along loop $l$.

    This is **exact**: the partition function equals the BP vacuum times a sum over all
    generalized loop corrections.

    For the free energy:
    $$\mathcal{F} = \mathcal{F}_{\text{BP}} - \ln\!\left(\sum_{l \in \mathcal{L}_G} Z_l\right)$$

    ### Why the Naive Loop Series Diverges

    Even if individual loop tensors decay as $|Z_l| \leq e^{-\alpha |l|}$, the **number**
    of disconnected loops grows combinatorially. On an $L \times L$ lattice, there are
    $\sim \binom{L^2}{k}$ ways to place $k$ disjoint plaquettes. This overwhelms the
    exponential decay.

    **Key insight:** work with $\ln \mathcal{Z}$ (free energy) instead of $\mathcal{Z}$ (partition function).
    """)
    return


@app.cell
def _(nx):
    def enumerate_simple_cycles(graph, max_length=None):
        """
        Enumerate simple cycles in the graph up to max_length edges.
        Returns list of cycles, each as a list of edges.
        """
        cycles = []
        for cycle_nodes in nx.simple_cycles(graph, length_bound=max_length):
            if len(cycle_nodes) < 3:
                continue
            # Convert to edge list
            edges = []
            for i in range(len(cycle_nodes)):
                u = cycle_nodes[i]
                v = cycle_nodes[(i + 1) % len(cycle_nodes)]
                edges.append((min(u, v), max(u, v)))
            # Canonical form: sorted edge list
            edges = sorted(set(edges))
            if edges not in cycles:
                cycles.append(edges)
        return cycles

    def enumerate_generalized_loops(graph, max_edges=None):
        """
        Enumerate all subgraphs where every vertex has even degree (generalized loops).
        For small graphs only — exponential in graph size.
        """
        edges = list(graph.edges())
        if max_edges is None:
            max_edges = len(edges)

        loops = []
        # The empty set is the trivial "loop"
        for r in range(2, min(max_edges, len(edges)) + 1):
            from itertools import combinations as _comb
            for edge_subset in _comb(range(len(edges)), r):
                selected = [edges[i] for i in edge_subset]
                # Check if every vertex has even degree
                deg = {}
                for u, v in selected:
                    deg[u] = deg.get(u, 0) + 1
                    deg[v] = deg.get(v, 0) + 1
                if all(d % 2 == 0 for d in deg.values()):
                    loops.append(selected)
        return loops

    return enumerate_generalized_loops, enumerate_simple_cycles


@app.cell
def _(enumerate_generalized_loops, mo, nx, plt):
    # Visualize generalized loops on small graphs
    _G = nx.cycle_graph(4)  # square
    _G.add_edge(0, 2)  # add diagonal -> creates two triangles

    _loops = enumerate_generalized_loops(_G, max_edges=6)

    _fig, _axes = plt.subplots(1, min(len(_loops), 6), figsize=(3 * min(len(_loops), 6), 3))
    if not hasattr(_axes, "__len__"):
        _axes = [_axes]
    _pos = nx.nx_agraph.graphviz_layout(_G, prog="neato")

    for idx, (loop, ax) in enumerate(zip(_loops[:6], _axes)):
        nx.draw(
            _G, _pos, ax=ax, with_labels=True, node_color="lightgray",
            node_size=300, edge_color="lightgray", width=1,
        )
        nx.draw_networkx_edges(
            _G, _pos, edgelist=loop, edge_color="red", width=3, ax=ax
        )
        ax.set_title(f"Loop {idx+1}\n({len(loop)} edges)", fontsize=9)

    plt.suptitle("Generalized loops on K4⁻ graph (even-degree subgraphs)", fontsize=11)
    plt.tight_layout()
    mo.md("**Generalized loops** = subgraphs where every vertex has even degree:")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Computing Loop Tensors

    For each generalized loop $l$, the **loop tensor** $Z_l$ is computed by:

    1. At each edge in the BP fixed point, define a **change-of-basis** that separates the
       BP vacuum direction ($|0\rangle$) from excited directions ($|s\rangle$, $s \geq 1$).
    2. For each vertex $v$ in the loop, compute the **fluctuation tensor**: $T_v$ contracted
       with BP messages, projected onto the excited states of the loop edges.
    3. Contract all fluctuation tensors along the loop edges.

    For $\chi = 2$ (Ising), the excited subspace is 1-dimensional at each edge, so each
    loop tensor is a product of scalars around the loop.
    """)
    return


@app.cell
def _(np):
    def compute_edge_basis(messages, u, v, chi):
        """
        Compute the change-of-basis at edge (u,v) that separates the BP vacuum
        from excited states.

        Returns:
            basis_uv: (chi, chi) matrix where row 0 is the BP vacuum direction
                      (aligned with mu_{u->v}), rows 1..chi-1 are orthogonal excited states
            basis_vu: same for the v->u direction
        """
        mu_uv = messages[(u, v)].copy()
        mu_uv /= np.linalg.norm(mu_uv)

        mu_vu = messages[(v, u)].copy()
        mu_vu /= np.linalg.norm(mu_vu)

        # Build orthonormal basis starting from the message direction
        def gram_schmidt_complete(v0, dim):
            basis = np.zeros((dim, dim))
            basis[0] = v0 / np.linalg.norm(v0)
            for i in range(1, dim):
                # Start with a random vector
                vec = np.zeros(dim)
                vec[i] = 1.0
                # Orthogonalize against previous basis vectors
                for j in range(i):
                    vec -= (vec @ basis[j]) * basis[j]
                norm = np.linalg.norm(vec)
                if norm < 1e-12:
                    # Try another starting vector
                    vec = np.random.randn(dim)
                    for j in range(i):
                        vec -= (vec @ basis[j]) * basis[j]
                    norm = np.linalg.norm(vec)
                basis[i] = vec / norm
            return basis

        basis_uv = gram_schmidt_complete(mu_uv, chi)
        basis_vu = gram_schmidt_complete(mu_vu, chi)

        return basis_uv, basis_vu

    return (compute_edge_basis,)


@app.cell
def _(compute_edge_basis, np):
    def compute_loop_tensor(tn, messages, loop_edges):
        """
        Compute the loop tensor Z_l for a given generalized loop.

        For chi=2, this simplifies significantly: the excited subspace is 1D per edge,
        so the loop tensor is a product of vertex contributions.

        Args:
            tn: TensorNetwork
            messages: dict of (u,v) -> message vector
            loop_edges: list of edges forming the loop

        Returns:
            Z_l: float (the loop tensor value)
        """
        chi = tn.chi

        # Find vertices in the loop and their degree within the loop
        loop_degree = {}
        for u, v in loop_edges:
            loop_degree[u] = loop_degree.get(u, 0) + 1
            loop_degree[v] = loop_degree.get(v, 0) + 1

        loop_vertices = set(loop_degree.keys())

        # Compute edge bases
        edge_bases = {}
        for u, v in loop_edges:
            basis_uv, basis_vu = compute_edge_basis(messages, u, v, chi)
            edge_bases[(u, v)] = basis_uv
            edge_bases[(v, u)] = basis_vu

        # For chi=2, each excited subspace is 1D (index 1 in the basis)
        # The loop tensor is the product over vertices of the fluctuation tensor
        # contracted with excited-state basis vectors on loop edges
        # and BP messages on non-loop edges

        if chi == 2:
            # Simple case: product formula
            Z_l = 1.0
            for node in loop_vertices:
                nbrs = tn.neighbors(node)
                T = tn.tensors[node].copy()

                # For each leg, determine if it's a loop edge or not
                # If loop edge: contract with excited basis vector (index 1)
                # If non-loop edge: contract with incoming message
                loop_edges_set = set()
                for eu, ev in loop_edges:
                    loop_edges_set.add((eu, ev))
                    loop_edges_set.add((ev, eu))

                # Contract legs from last to first to avoid index shifting
                vectors = []
                for nbr in nbrs:
                    if (nbr, node) in loop_edges_set or (node, nbr) in loop_edges_set:
                        # Loop edge: use excited basis vector
                        basis = edge_bases[(nbr, node)]
                        vectors.append(basis[1])  # excited state
                    else:
                        # Non-loop edge: use BP message
                        vectors.append(messages[(nbr, node)])

                # Contract tensor with all vectors
                result = T
                for i, vec in enumerate(vectors):
                    result = np.tensordot(result, vec, axes=([0], [0]))
                Z_l *= float(result)

            return Z_l
        else:
            # General case: need to sum over all excited-state assignments
            # This is more complex and involves actual tensor contraction
            # For now, handle chi=2 case which covers the Ising model
            raise NotImplementedError("Loop tensor for chi > 2 not yet implemented")

    return (compute_loop_tensor,)


@app.cell
def _(
    belief_propagation,
    compute_bp_partition,
    compute_loop_tensor,
    enumerate_generalized_loops,
    make_ising_tn,
    mo,
    np,
):
    # Verify the loop series expansion on a small example
    _tn = make_ising_tn(3, 3, beta=0.3)
    _msgs, _conv, _ = belief_propagation(_tn, max_iter=500, damping=0.2)

    _logZ_bp = compute_bp_partition(_tn, _msgs)
    _Z_bp = np.exp(_logZ_bp)

    # Exact contraction
    _Z_exact = _tn.contract_exact()

    # Enumerate all generalized loops
    _loops = enumerate_generalized_loops(_tn.graph, max_edges=12)

    # Compute loop tensors
    _loop_sum = 1.0  # empty loop contributes 1
    _loop_contributions = []
    for _loop in _loops:
        _Zl = compute_loop_tensor(_tn, _msgs, _loop)
        _loop_sum += _Zl
        _loop_contributions.append((len(_loop), _Zl))

    _Z_reconstructed = _Z_bp * _loop_sum

    mo.md(f"""
    ### Verification: Loop Series on 3×3 Ising (β=0.3)

    | Quantity | Value |
    |----------|-------|
    | Z_exact | {_Z_exact:.8f} |
    | Z_BP | {_Z_bp:.8f} |
    | Z_BP × Σ Z_l | {_Z_reconstructed:.8f} |
    | Number of generalized loops (≤12 edges) | {len(_loops)} |
    | Relative error (BP only) | {abs(_Z_bp - _Z_exact) / abs(_Z_exact):.6e} |
    | Relative error (with loops) | {abs(_Z_reconstructed - _Z_exact) / abs(_Z_exact):.6e} |

    The loop series sum brings us closer to the exact answer, but enumerating all loops
    is exponentially costly — we need the cluster expansion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part VI: The Cluster Expansion

    ### From Loops to Clusters: The Core Idea

    The loop series for $\mathcal{Z}$ includes disconnected loops whose number grows
    combinatorially. The solution: take the **logarithm** and expand $\ln \mathcal{Z}$
    in terms of **connected clusters only**.

    This is the **cluster expansion** (linked cluster expansion / Mayer expansion) from
    statistical mechanics.

    **Key formula:**
    $$
    \mathcal{F} = \mathcal{F}_{\text{BP}} - \sum_{\text{connected clusters } \gamma}
    \frac{1}{|\gamma|!}\, \phi(\gamma) \prod_{l \in \gamma} Z_l
    $$

    where $\phi(\gamma)$ is the **Ursell function** (cumulant). Two loops are "incompatible"
    if they share a vertex or are identical. The Ursell function vanishes for disconnected
    clusters — this eliminates the combinatorial explosion.

    ### The Ursell Function

    Given a cluster $\Gamma = (l_1, \ldots, l_k)$, build the **incompatibility graph**
    $H(\Gamma)$ with edges between loops that share vertices.

    $$
    \phi(\Gamma) = \sum_{\substack{G \subseteq K_k \\ G \text{ connected, spanning}}} (-1)^{|E(G)|}
    $$

    - Single loop: $\phi = 1$
    - Pair of overlapping loops: $\phi = -1$
    - $\phi = 0$ for disconnected clusters (key property!)

    ### Main Convergence Theorem (Theorem III.4)

    If the loop contribution decays sufficiently fast: $|Z_l| \leq \alpha^{|l|}$
    with $\alpha < \alpha^*(\Delta)$ (depending on max degree), then:

    $$
    \left|\mathcal{F} - \mathcal{F}_{\text{BP}} - \sum_{m=1}^{M} C_m\right| \leq A \cdot r^M
    $$

    for some $r < 1$. This is **exponential convergence** — the central result of the paper.
    """)
    return


@app.cell
def _(nx):
    def ursell_function(n_loops, incompatibility_edges):
        """
        Compute the Ursell function for a cluster of n_loops loops
        with given incompatibility edges.

        The Ursell function is: sum over connected spanning subgraphs G of K_n
        of (-1)^{|E(G)|}.

        For small clusters, compute by brute force.
        """
        if n_loops == 1:
            return 1

        # Build the incompatibility graph
        H = nx.Graph()
        H.add_nodes_from(range(n_loops))
        H.add_edges_from(incompatibility_edges)

        # If H is disconnected, Ursell function is 0
        if not nx.is_connected(H):
            return 0

        # For connected H, compute by inclusion-exclusion over spanning connected subgraphs
        # of the complete graph K_n
        # For small n, enumerate all subsets of edges of K_n that form connected spanning graphs
        all_edges = list(nx.complete_graph(n_loops).edges())
        phi = 0

        from itertools import combinations as _comb
        for r in range(n_loops - 1, len(all_edges) + 1):
            for edge_subset in _comb(all_edges, r):
                G = nx.Graph()
                G.add_nodes_from(range(n_loops))
                G.add_edges_from(edge_subset)
                if nx.is_connected(G):
                    phi += (-1) ** len(edge_subset)

        return phi

    # Test: single loop
    assert ursell_function(1, []) == 1
    # Test: pair of overlapping loops
    assert ursell_function(2, [(0, 1)]) == -1
    # Test: pair of non-overlapping loops
    assert ursell_function(2, []) == 0

    print("Ursell function tests passed:")
    print(f"  φ(single loop) = {ursell_function(1, [])}")
    print(f"  φ(overlapping pair) = {ursell_function(2, [(0, 1)])}")
    print(f"  φ(non-overlapping pair) = {ursell_function(2, [])}")
    print(f"  φ(connected triple) = {ursell_function(3, [(0,1),(1,2),(0,2)])}")
    return (ursell_function,)


@app.cell
def _(compute_loop_tensor, enumerate_simple_cycles, nx, ursell_function):
    def cluster_expansion(tn, messages, max_weight=12, verbose=False):
        """
        Compute the cluster expansion correction to the BP free energy.

        This enumerates connected clusters of simple cycles up to a given
        total edge weight, computes their loop tensors and Ursell functions,
        and returns the free energy correction.

        Args:
            tn: TensorNetwork with BP messages
            messages: converged BP messages
            max_weight: maximum total edge weight of clusters to include
            verbose: print progress

        Returns:
            corrections: dict mapping weight -> correction at that weight
            total_correction: float, total free energy correction
        """
        graph = tn.graph

        # Step 1: Enumerate simple cycles up to max_weight edges
        cycles = enumerate_simple_cycles(graph, max_length=max_weight)
        if verbose:
            print(f"Found {len(cycles)} simple cycles up to weight {max_weight}")

        # Step 2: Compute loop tensor for each cycle
        cycle_tensors = {}
        for i, cycle in enumerate(cycles):
            Zl = compute_loop_tensor(tn, messages, cycle)
            cycle_tensors[i] = Zl
            if verbose and i < 5:
                print(f"  Cycle {i} (weight {len(cycle)}): Z_l = {Zl:.6e}")

        # Step 3: Build incompatibility graph between cycles
        # Two cycles are incompatible if they share a vertex
        cycle_vertices = {}
        for i, cycle in enumerate(cycles):
            verts = set()
            for u, v in cycle:
                verts.add(u)
                verts.add(v)
            cycle_vertices[i] = verts

        incomp = nx.Graph()
        incomp.add_nodes_from(range(len(cycles)))
        for i in range(len(cycles)):
            for j in range(i + 1, len(cycles)):
                if cycle_vertices[i] & cycle_vertices[j]:
                    incomp.add_edge(i, j)

        # Step 4: Enumerate connected clusters and compute corrections
        # A cluster is a multiset of cycles whose incompatibility graph is connected
        # For simplicity, we consider sets (multiplicity 1) of cycles

        corrections = {}

        # Order 1: single cycles
        for i, cycle in enumerate(cycles):
            w = len(cycle)
            if w > max_weight:
                continue
            # Single cycle: Ursell = 1, symmetry factor = 1
            contrib = cycle_tensors[i]
            corrections[w] = corrections.get(w, 0) + contrib

        # Order 2: pairs of overlapping cycles
        for i, j in incomp.edges():
            w = len(cycles[i]) + len(cycles[j])
            if w > max_weight:
                continue
            # Pair: Ursell = -1, symmetry factor = 1/2!
            phi = ursell_function(2, [(0, 1)])
            contrib = (1.0 / 2) * phi * cycle_tensors[i] * cycle_tensors[j]
            corrections[w] = corrections.get(w, 0) + contrib

        # Order 3: triples of mutually overlapping cycles
        for i in range(len(cycles)):
            for j in range(i + 1, len(cycles)):
                if not incomp.has_edge(i, j):
                    continue
                for k in range(j + 1, len(cycles)):
                    w = len(cycles[i]) + len(cycles[j]) + len(cycles[k])
                    if w > max_weight:
                        continue
                    edges_ijk = []
                    if incomp.has_edge(i, j):
                        edges_ijk.append((0, 1))
                    if incomp.has_edge(i, k):
                        edges_ijk.append((0, 2))
                    if incomp.has_edge(j, k):
                        edges_ijk.append((1, 2))
                    # Check connectivity
                    H = nx.Graph()
                    H.add_nodes_from([0, 1, 2])
                    H.add_edges_from(edges_ijk)
                    if not nx.is_connected(H):
                        continue
                    phi = ursell_function(3, edges_ijk)
                    contrib = (
                        (1.0 / 6) * phi
                        * cycle_tensors[i] * cycle_tensors[j] * cycle_tensors[k]
                    )
                    corrections[w] = corrections.get(w, 0) + contrib

        total_correction = sum(corrections.values())

        if verbose:
            print(f"\nCluster expansion corrections by weight:")
            for w in sorted(corrections.keys()):
                print(f"  Weight {w}: {corrections[w]:.6e}")
            print(f"Total correction to log(Z)/N: {total_correction:.6e}")

        return corrections, total_correction

    return (cluster_expansion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Toy Example: Triangle Graph

    Consider a triangle (3 vertices, 3 edges, $\chi = 2$). There is exactly one fundamental
    loop: the triangle itself. The loop series is:

    $$\mathcal{Z} = \mathcal{Z}_{\text{BP}} \cdot (1 + Z_{\text{triangle}})$$

    The cluster expansion for the free energy is:

    $$\mathcal{F} = \mathcal{F}_{\text{BP}} - \ln(1 + Z_{\text{triangle}})
    = \mathcal{F}_{\text{BP}} - \sum_{k=1}^{\infty} \frac{(-1)^{k+1}}{k} Z_{\text{triangle}}^k$$

    This converges when $|Z_{\text{triangle}}| < 1$.
    """)
    return


@app.cell
def _(
    TensorNetwork,
    belief_propagation,
    compute_bp_partition,
    compute_loop_tensor,
    mo,
    np,
    nx,
    scipy,
):
    # Triangle Ising tensor network
    _G = nx.cycle_graph(3)
    _beta = 0.3
    _J = 1.0
    _tn = TensorNetwork(_G, chi=2)

    _W = np.array([
        [np.exp(_beta * _J), np.exp(-_beta * _J)],
        [np.exp(-_beta * _J), np.exp(_beta * _J)],
    ])
    _sqrtW = scipy.linalg.sqrtm(_W).real

    for _node in _G.nodes():
        _nbrs = sorted(_G.neighbors(_node))
        _deg = len(_nbrs)
        _shape = tuple([2] * _deg)
        _T = np.zeros(_shape)
        for _sigma in range(2):
            _contrib = np.ones(1)
            for _ in range(_deg):
                _contrib = np.outer(_contrib, _sqrtW[_sigma, :]).flatten()
            _T += _contrib.reshape(_shape)
        _tn.set_tensor(_node, _T)

    # Exact
    _Z_exact = _tn.contract_exact()

    # BP
    _msgs, _, _ = belief_propagation(_tn, max_iter=500, damping=0.2)
    _logZ_bp = compute_bp_partition(_tn, _msgs)
    _Z_bp = np.exp(_logZ_bp)

    # Loop tensor for the triangle
    _triangle_edges = [(0, 1), (0, 2), (1, 2)]
    _Z_tri = compute_loop_tensor(_tn, _msgs, _triangle_edges)

    # Loop series reconstruction
    _Z_loop = _Z_bp * (1 + _Z_tri)

    # Cluster expansion terms
    _cluster_terms = []
    _cluster_sum = 0.0
    for _k in range(1, 20):
        _term = ((-1) ** (_k + 1) / _k) * _Z_tri**_k
        _cluster_sum += _term
        _cluster_terms.append(_cluster_sum)

    mo.md(f"""
    ### Triangle Graph Results (β = {_beta})

    | Quantity | Value |
    |----------|-------|
    | Z_exact | {_Z_exact:.10f} |
    | Z_BP | {_Z_bp:.10f} |
    | Z_triangle (loop tensor) | {_Z_tri:.10f} |
    | Z_BP × (1 + Z_tri) | {_Z_loop:.10f} |
    | Relative error (BP) | {abs(_Z_bp - _Z_exact)/abs(_Z_exact):.6e} |
    | Relative error (loop series) | {abs(_Z_loop - _Z_exact)/abs(_Z_exact):.6e} |

    The cluster expansion for $\\ln(1 + Z_{{\text{{tri}}}}) = {np.log(1 + _Z_tri):.8f}$
    converges in a few terms since $|Z_{{\text{{tri}}}}| = {abs(_Z_tri):.6f} < 1$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part VII: Numerical Results — 2D Ising Model

    We now apply the full pipeline (BP → cluster expansion) to the 2D Ising model
    and reproduce the key results from the paper:

    1. Free energy error vs. cluster expansion order
    2. Comparison of BP, truncated loop series, and cluster expansion
    3. Convergence rate as a function of temperature
    """)
    return


@app.cell
def _(
    belief_propagation,
    cluster_expansion,
    compute_bp_partition,
    ising_partition_brute_force,
    make_ising_tn,
    mo,
    np,
    plt,
):
    # Cluster expansion on small Ising models at various temperatures
    _L = 4
    _N = _L * _L
    _betas = [0.2, 0.35, 0.44, 0.5]
    _beta_c = 0.5 * np.log(1 + np.sqrt(2))
    _max_weights = [4, 6, 8, 10, 12]

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))

    _colors = ["blue", "green", "orange", "red"]

    for _idx, _beta in enumerate(_betas):
        _tn = make_ising_tn(_L, _L, _beta)
        _Z_exact = ising_partition_brute_force(_beta, _L, _L)
        _f_exact = -np.log(_Z_exact) / _N

        _msgs, _conv, _ = belief_propagation(_tn, max_iter=500, damping=0.2)
        _logZ_bp = compute_bp_partition(_tn, _msgs)
        _f_bp = -_logZ_bp / _N

        _errors = [abs(_f_bp - _f_exact)]
        _weights_used = [0]

        for _mw in _max_weights:
            _corr, _total = cluster_expansion(_tn, _msgs, max_weight=_mw)
            _f_cluster = -(_logZ_bp + _total) / _N
            _errors.append(abs(_f_cluster - _f_exact))
            _weights_used.append(_mw)

        _label = f"β={_beta}" + (" (≈βc)" if abs(_beta - _beta_c) < 0.01 else "")
        _axes[0].semilogy(
            _weights_used, _errors, "o-",
            color=_colors[_idx], label=_label, markersize=5
        )

    _axes[0].set_xlabel("Max cluster weight M")
    _axes[0].set_ylabel("|f_cluster - f_exact|")
    _axes[0].set_title(f"Cluster Expansion Convergence ({_L}×{_L} Ising)")
    _axes[0].legend()
    _axes[0].grid(True, alpha=0.3)

    # Compare BP vs cluster expansion across temperatures
    _betas_scan = np.linspace(0.1, 0.7, 12)
    _bp_errors = []
    _cluster_errors = []

    for _b in _betas_scan:
        _tn = make_ising_tn(_L, _L, _b)
        _Z_exact = ising_partition_brute_force(_b, _L, _L)
        _f_exact = -np.log(_Z_exact) / _N

        _msgs, _, _ = belief_propagation(_tn, max_iter=500, damping=0.2)
        _logZ_bp = compute_bp_partition(_tn, _msgs)
        _f_bp_val = -_logZ_bp / _N
        _bp_errors.append(abs(_f_bp_val - _f_exact))

        _corr, _total = cluster_expansion(_tn, _msgs, max_weight=8)
        _f_cl = -(_logZ_bp + _total) / _N
        _cluster_errors.append(abs(_f_cl - _f_exact))

    _axes[1].semilogy(_betas_scan, _bp_errors, "rs-", label="BP only", markersize=4)
    _axes[1].semilogy(_betas_scan, _cluster_errors, "bo-", label="BP + cluster (M=8)", markersize=4)
    _axes[1].axvline(_beta_c, color="gray", linestyle="--", alpha=0.5, label="βc")
    _axes[1].set_xlabel("β")
    _axes[1].set_ylabel("|f - f_exact|")
    _axes[1].set_title(f"BP vs Cluster Expansion ({_L}×{_L} Ising)")
    _axes[1].legend()
    _axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    mo.md("**Key result:** the cluster expansion systematically improves BP, especially away from βc:")
    _fig
    return


@app.cell
def _(
    belief_propagation,
    compute_loop_tensor,
    enumerate_simple_cycles,
    make_ising_tn,
    mo,
    np,
    plt,
):
    # Analyze loop contribution decay
    _L = 4
    _betas_decay = [0.2, 0.35, 0.44]
    _beta_c = 0.5 * np.log(1 + np.sqrt(2))

    _fig, _ax = plt.subplots(figsize=(7, 5))
    _colors = ["blue", "green", "red"]

    for _idx, _beta in enumerate(_betas_decay):
        _tn = make_ising_tn(_L, _L, _beta)
        _msgs, _, _ = belief_propagation(_tn, max_iter=500, damping=0.2)

        _cycles = enumerate_simple_cycles(_tn.graph, max_length=10)

        _by_weight = {}
        for _cycle in _cycles:
            _w = len(_cycle)
            _Zl = abs(compute_loop_tensor(_tn, _msgs, _cycle))
            if _w not in _by_weight:
                _by_weight[_w] = []
            _by_weight[_w].append(_Zl)

        _weights = sorted(_by_weight.keys())
        _means = [np.mean(_by_weight[w]) for w in _weights]

        _ax.semilogy(
            _weights, _means, "o-",
            color=_colors[_idx],
            label=f"β={_beta}" + (" (≈βc)" if abs(_beta - _beta_c) < 0.02 else ""),
            markersize=5,
        )

    _ax.set_xlabel("Loop weight (number of edges)")
    _ax.set_ylabel("Mean |Z_l| (loop contribution)")
    _ax.set_title(f"Loop Contribution Decay ({_L}×{_L} Ising)")
    _ax.legend()
    _ax.grid(True, alpha=0.3)

    plt.tight_layout()
    mo.md("""
    **Loop contribution decay:** the key assumption for convergence. Away from βc,
    loop tensors decay exponentially with loop size:
    """)
    _fig
    return


@app.cell
def _(
    belief_propagation,
    cluster_expansion,
    compute_bp_partition,
    ising_partition_brute_force,
    make_ising_tn,
    mo,
    np,
    plt,
):
    # Convergence rate analysis
    _L = 4
    _N = _L * _L
    _betas_rate = np.linspace(0.1, 0.65, 12)
    _beta_c = 0.5 * np.log(1 + np.sqrt(2))
    _max_weights_conv = [4, 6, 8, 10]

    _rates = []

    for _b in _betas_rate:
        _tn = make_ising_tn(_L, _L, _b)
        _Z_exact = ising_partition_brute_force(_b, _L, _L)
        _f_exact = -np.log(_Z_exact) / _N

        _msgs, _, _ = belief_propagation(_tn, max_iter=500, damping=0.2)
        _logZ_bp = compute_bp_partition(_tn, _msgs)

        _errs = []
        for _mw in _max_weights_conv:
            _corr, _total = cluster_expansion(_tn, _msgs, max_weight=_mw)
            _f_cl = -(_logZ_bp + _total) / _N
            _errs.append(abs(_f_cl - _f_exact) + 1e-16)

        # Fit exponential: error ~ A * r^M
        if len(_errs) >= 2 and _errs[0] > 0 and _errs[-1] > 0:
            _log_errs = np.log(np.array(_errs) + 1e-20)
            _mw_arr = np.array(_max_weights_conv, dtype=float)
            # Linear fit in log space
            _coeffs = np.polyfit(_mw_arr, _log_errs, 1)
            _r = np.exp(_coeffs[0])
            _rates.append(min(_r, 1.0))
        else:
            _rates.append(1.0)

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(_betas_rate, _rates, "bo-", markersize=5)
    _ax.axvline(_beta_c, color="r", linestyle="--", alpha=0.7, label=f"βc ≈ {_beta_c:.4f}")
    _ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    _ax.set_xlabel("β")
    _ax.set_ylabel("Convergence rate r")
    _ax.set_title(f"Cluster Expansion Convergence Rate ({_L}×{_L} Ising)")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _ax.set_ylim(0, 1.2)

    plt.tight_layout()
    mo.md("""
    **Convergence rate** $r$ extracted from fitting error $\\sim A \\cdot r^M$.
    $r < 1$ means exponential convergence; $r \\to 1$ near $\\beta_c$:
    """)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Magnetization via the Cluster Expansion

    The cluster expansion extends to **local observables**. For the magnetization
    $\langle \sigma_i \rangle$, we compute the ratio of the partition function with
    an observable insertion to the bare partition function.

    **Exact result (Yang, 1952)** for the spontaneous magnetization:
    $$\langle \sigma \rangle = \left(1 - \sinh^{-4}(2\beta J)\right)^{1/8}$$
    for $\beta > \beta_c$ (ordered phase).
    """)
    return


@app.cell
def _(mo, np, plt):
    # Yang's exact magnetization
    def yang_magnetization(beta, J=1.0):
        """Exact spontaneous magnetization for 2D Ising (Yang 1952)."""
        x = np.sinh(2 * beta * J)
        if x <= 1:
            return 0.0  # disordered phase
        return (1 - x**(-4)) ** (1.0 / 8)

    _beta_c = 0.5 * np.log(1 + np.sqrt(2))
    _betas = np.linspace(0.01, 1.0, 200)
    _mag = [yang_magnetization(b) for b in _betas]

    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(_betas, _mag, "b-", linewidth=2)
    _ax.axvline(_beta_c, color="r", linestyle="--", alpha=0.7, label=f"βc ≈ {_beta_c:.4f}")
    _ax.set_xlabel("β")
    _ax.set_ylabel("⟨σ⟩")
    _ax.set_title("Spontaneous Magnetization (Yang 1952)")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.md("**Yang's exact magnetization** — the order parameter for the 2D Ising phase transition:")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Performance Benchmarks

    For fixed truncation order $M$, the algorithm scales **linearly** in system size $N$.
    The cost is $O(N \cdot e^{O(M)})$ — exponential in the truncation order but polynomial
    (actually linear) in $N$.
    """)
    return


@app.cell
def _(
    belief_propagation,
    cluster_expansion,
    compute_bp_partition,
    make_ising_tn,
    mo,
    plt,
    time,
):
    _sizes = [3, 4, 5, 6, 7]
    _beta_bench = 0.3
    _max_w = 8

    _bp_times = []
    _cluster_times = []
    _Ns = []

    for _L in _sizes:
        _tn = make_ising_tn(_L, _L, _beta_bench)
        _Ns.append(_L * _L)

        _t0 = time.time()
        _msgs, _, _ = belief_propagation(_tn, max_iter=300, damping=0.3)
        _t1 = time.time()
        _bp_times.append(_t1 - _t0)

        _logZ_bp = compute_bp_partition(_tn, _msgs)

        _t2 = time.time()
        _corr, _total = cluster_expansion(_tn, _msgs, max_weight=_max_w)
        _t3 = time.time()
        _cluster_times.append(_t3 - _t2)

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(_Ns, _bp_times, "bo-", label="BP convergence")
    _ax.plot(_Ns, _cluster_times, "rs-", label=f"Cluster expansion (M={_max_w})")
    _ax.plot(_Ns, [t1 + t2 for t1, t2 in zip(_bp_times, _cluster_times)], "g^-", label="Total")
    _ax.set_xlabel("System size N = L²")
    _ax.set_ylabel("Time (seconds)")
    _ax.set_title("Runtime Scaling (β = 0.3)")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.md("**Runtime scaling:** for fixed truncation order, cost grows roughly linearly in $N$:")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Algorithm Summary

    The full pipeline:

    ```
    Input: Tensor network T, truncation order M
    ─────────────────────────────────────────────
    1. Run BP → converged messages {μ_{v→w}}         O(N χ³) per iteration
    2. Compute BP vacuum → Z_BP, F_BP                O(N χ²)
    3. Enumerate connected clusters up to weight M   O(N · exp(O(M)))
    4. Compute loop tensors for each cluster          O(N · exp(O(M)))
    5. Compute Ursell functions                       O(N · exp(O(M)))
    6. Sum cluster contributions → F_M                O(N · exp(O(M)))
    ─────────────────────────────────────────────
    Output: F_M with error ≤ A · r^M  (r < 1)
    ```

    **Key properties:**
    - **Linear in system size** $N$ for fixed $M$
    - **Exponentially convergent** when loop contributions decay fast enough
    - **Rigorous error bounds** via the Kotecký-Preiss criterion
    - **Systematic improvement** over BP — each order brings exponential error reduction

    ### Comparison with Other Methods

    | Method | Complexity | Convergence | Error bound? |
    |--------|-----------|------------|--------------|
    | **BP** (Bethe approx) | $O(N\chi^3)$ | N/A (fixed point) | No |
    | **Naïve loop series** | $O(N \cdot 2^{|E|})$ | Diverges for large $N$ | No |
    | **Evenbly et al. loop series** | $O(N \cdot e^{O(M)})$ | Convergent (numerical) | No |
    | **This paper: cluster expansion** | $O(N \cdot e^{O(M)})$ | **Exponentially convergent** | **Yes** |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Extensions and Discussion

    ### Application to Quantum Error Correction

    Tensor network contraction is used in **decoders** for quantum error-correcting codes
    (surface codes, LDPC codes). BP is already a fast approximate decoder; the cluster
    expansion can improve decoding accuracy while maintaining near-linear runtime. Typical
    error patterns in QEC may have rapidly decaying loop contributions, making the cluster
    expansion particularly effective.

    ### Application to Quantum Simulation

    Simulating quantum dynamics with 2D tensor networks (PEPS, PEPO) requires contraction
    as a subroutine. BP provides fast approximate contraction; the cluster expansion
    improves accuracy. Particularly relevant for ground state energy, time evolution,
    and thermal state computation.

    ### Limitations

    1. **Requires a BP fixed point** — may not converge for some tensor networks
    2. **Near phase transitions**, convergence slows (loop contributions grow)
    3. **Multiple degenerate BP fixed points** (glassy systems, GHZ states) can cause failure
    4. **Large bond dimension** $\chi$: loop tensor computation becomes expensive

    ### Open Questions

    1. Optimal truncation strategy — which clusters contribute most?
    2. Generalization to tensor networks with open indices
    3. Connection to generalized BP (Kikuchi, CVM) — can the cluster expansion improve these?
    4. Rigorous bounds on loop contributions for physically relevant tensor networks
    5. Extension to complex-valued / non-Hermitian tensor networks

    ### Connection to Polymer Models

    The proof maps the cluster expansion to an **abstract polymer model**:
    - **Polymers** = generalized loops
    - **Polymer weight** = loop tensor $Z_l$
    - **Incompatibility** = vertex overlap
    - **Kotecký-Preiss criterion:** convergence when $\sum_{\gamma' \sim \gamma} |w(\gamma')| e^{a(|\gamma'|)} \leq a(|\gamma|)$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Appendix: Verifying the Kotecký-Preiss Criterion

    The Kotecký-Preiss criterion requires that for a suitable function $a(\cdot)$:

    $$
    \sum_{\gamma' \sim \gamma} |w(\gamma')|\, e^{a(|\gamma'|)} \leq a(|\gamma|)
    $$

    where the sum is over all polymers (loops) $\gamma'$ incompatible with $\gamma$,
    and $w(\gamma') = Z_{\gamma'}$ is the polymer weight.

    With $a(n) = \alpha \cdot n$ (linear in loop size), the criterion becomes:

    $$
    \sum_{\gamma' \sim \gamma} |Z_{\gamma'}|\, e^{\alpha |\gamma'|} \leq \alpha \cdot |\gamma|
    $$

    This is satisfied when loop contributions decay faster than $e^{-\alpha |l|}$ and
    the number of incompatible loops grows slower than exponentially in $\alpha$.
    """)
    return


@app.cell
def _(
    belief_propagation,
    compute_loop_tensor,
    enumerate_simple_cycles,
    make_ising_tn,
    mo,
    np,
    plt,
):
    # Check Kotecký-Preiss criterion for the Ising model
    _L = 4
    _betas_kp = [0.2, 0.3, 0.4]
    _beta_c = 0.5 * np.log(1 + np.sqrt(2))

    _fig, _ax = plt.subplots(figsize=(7, 4))

    for _beta in _betas_kp:
        _tn = make_ising_tn(_L, _L, _beta)
        _msgs, _, _ = belief_propagation(_tn, max_iter=500, damping=0.2)

        _cycles = enumerate_simple_cycles(_tn.graph, max_length=10)

        # For each cycle, compute |Z_l| and find incompatible cycles
        _cycle_data = []
        _cycle_verts = []
        for _c in _cycles:
            _Zl = abs(compute_loop_tensor(_tn, _msgs, _c))
            _verts = set()
            for _u, _v in _c:
                _verts.add(_u)
                _verts.add(_v)
            _cycle_data.append((len(_c), _Zl))
            _cycle_verts.append(_verts)

        # For each loop weight, compute the LHS of Kotecký-Preiss
        # using alpha = 0.5
        _alpha = 0.5
        _by_weight = {}
        for i, (_w, _Zl) in enumerate(_cycle_data):
            if _w not in _by_weight:
                _by_weight[_w] = []
            # Sum |Z_l'| * exp(alpha * |l'|) over incompatible l'
            _lhs = 0.0
            for j, (_w2, _Zl2) in enumerate(_cycle_data):
                if i == j:
                    continue
                if _cycle_verts[i] & _cycle_verts[j]:  # incompatible
                    _lhs += _Zl2 * np.exp(_alpha * _w2)
            _by_weight[_w].append((_lhs, _alpha * _w))

        _weights = sorted(_by_weight.keys())
        _lhs_means = [np.mean([x[0] for x in _by_weight[w]]) for w in _weights]
        _rhs_vals = [_alpha * w for w in _weights]

        _ax.plot(_weights, _lhs_means, "o-", label=f"LHS (β={_beta})", markersize=5)

    _ax.plot(_weights, _rhs_vals, "k--", linewidth=2, label="RHS = α·|γ| (α=0.5)")
    _ax.set_xlabel("Loop weight |γ|")
    _ax.set_ylabel("Kotecký-Preiss bound")
    _ax.set_title(f"Kotecký-Preiss Criterion ({_L}×{_L} Ising, α=0.5)")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.md("""
    **Kotecký-Preiss criterion:** the LHS (sum of weighted incompatible loop contributions)
    must be bounded by the RHS ($\\alpha \\cdot |\\gamma|$). When satisfied, the cluster
    expansion converges absolutely:
    """)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Appendix: Loop Counting on Random Regular Graphs

    On random $\Delta$-regular graphs, the expected number of simple loops of length $k$ is:
    $$
    \mathbb{E}[\text{# loops of length } k] = \frac{(\Delta - 1)^k}{2k}
    $$

    This shows that for large $\Delta$, loops are rare per-vertex and BP becomes more
    accurate. The convergence condition requires loop contribution decay faster than
    $(\Delta - 1)^{-1}$.
    """)
    return


@app.cell
def _(enumerate_simple_cycles, mo, nx, plt):
    # Compare loop counting on random regular graphs with analytical prediction
    _deltas = [3, 4, 5]
    _N_graph = 20
    _n_samples = 10

    _fig, _ax = plt.subplots(figsize=(7, 4))

    for _delta in _deltas:
        _counts_by_k = {}
        for _ in range(_n_samples):
            try:
                _G = nx.random_regular_graph(_delta, _N_graph)
                _cycles = enumerate_simple_cycles(_G, max_length=10)
                for _c in _cycles:
                    _k = len(_c)
                    _counts_by_k[_k] = _counts_by_k.get(_k, 0) + 1
            except nx.NetworkXError:
                continue

        _ks = sorted(_counts_by_k.keys())
        _avg_counts = [_counts_by_k[k] / _n_samples for k in _ks]
        _predicted = [(_delta - 1) ** k / (2 * k) for k in _ks]

        _ax.plot(_ks, _avg_counts, "o-", label=f"Δ={_delta} (empirical)", markersize=5)
        _ax.plot(_ks, _predicted, "x--", label=f"Δ={_delta} (analytical)", markersize=7)

    _ax.set_xlabel("Loop length k")
    _ax.set_ylabel("Number of simple loops")
    _ax.set_title(f"Loop Counting on Random Regular Graphs (N={_N_graph})")
    _ax.legend(fontsize=8)
    _ax.set_yscale("log")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.md("**Loop counting** on random regular graphs — empirical vs. analytical prediction:")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Glossary of Key Terms

    | Term | Definition |
    |------|-----------|
    | **Tensor Network** | Collection of tensors on a graph with shared indices |
    | **Contraction** | Summing over all internal indices to produce a scalar |
    | **Bond dimension** $\chi$ | Dimension of the Hilbert space on each edge |
    | **Belief Propagation (BP)** | Iterative message-passing for approximate contraction |
    | **BP Vacuum** | Zeroth-order approximation: partition function from BP messages |
    | **Bethe Free Energy** | Free energy from BP = exact on trees, approximate on loopy graphs |
    | **Generalized Loop** | Subgraph where every vertex has even degree |
    | **Loop Tensor** $Z_l$ | Contribution of loop $l$ to the partition function correction |
    | **Loop Series** | Exact expansion: $\mathcal{Z} = \mathcal{Z}_{\text{BP}} \cdot \sum_l Z_l$ |
    | **Cluster** | Multiset of loops whose incompatibility graph is connected |
    | **Ursell Function** $\phi(\gamma)$ | Combinatorial weight ensuring only connected contributions survive |
    | **Cluster Expansion** | Systematic expansion of free energy in connected clusters |
    | **Loop Contribution** | Magnitude $|Z_l|$ — controls convergence rate |
    | **Kotecký-Preiss Criterion** | Sufficient condition for convergence of the cluster expansion |
    | **Connective Constant** | Growth rate of self-avoiding walks — determines critical loop decay rate |

    ---

    ## References

    1. **Midha & Zhang**, "Beyond Belief Propagation..." [arXiv:2510.02290](https://arxiv.org/abs/2510.02290) (2025)
    2. **Chertkov & Chernyak**, "Loop Calculus in Statistical Physics" — Phys. Rev. E (2006)
    3. **Evenbly et al.**, "Loop Series Expansions for Tensor Networks" [arXiv:2409.03108](https://arxiv.org/abs/2409.03108) (2025)
    4. **Gray et al.**, "TN Loop Cluster Expansions for Quantum Many-Body Problems" [arXiv:2510.05647](https://arxiv.org/abs/2510.05647) (2025)
    5. **Kotecký & Preiss**, "Cluster Expansion for Abstract Polymer Models" — Commun. Math. Phys. (1986)
    6. **Onsager**, "Crystal Statistics. I." — Phys. Rev. (1944)
    7. **Alkabetz & Arad**, "Tensor Networks Contraction and BP" — Phys. Rev. Res. (2021)
    8. **Yedidia, Freeman, Weiss**, "Understanding Belief Propagation and Its Generalizations" (2003)
    9. **Friedli & Velenik**, *Statistical Mechanics of Lattice Systems* — Cambridge (2017)
    """)
    return


if __name__ == "__main__":
    app.run()

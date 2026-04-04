# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.0",
#     "numpy>=1.24.0",
#     "scipy>=1.10.0",
#     "matplotlib>=3.7.0",
#     "networkx>=3.0",
# ]
# ///

import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import time
    from collections import Counter
    from itertools import combinations
    from typing import Any

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import scipy

    def layout_graph(G: nx.Graph, seed: int = 0) -> dict[Any, Any]:
        try:
            return nx.nx_agraph.graphviz_layout(G, prog="neato")
        except Exception:
            return nx.spring_layout(G, seed=seed, k=0.15)

    return Counter, combinations, layout_graph, np, nx, plt, scipy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Abstract

    *(Verbatim from [arXiv:2510.02290](https://arxiv.org/abs/2510.02290), v2.)*

    > Tensor network contraction on arbitrary graphs is a fundamental computational challenge with applications ranging from quantum simulation to error correction. While belief propagation (BP) provides a powerful approximation algorithm for this task, its accuracy limitations are poorly understood and systematic improvements remain elusive. Here, we develop a rigorous theoretical framework for BP in tensor networks, leveraging insights from statistical mechanics to devise a *cluster expansion* that systematically improves the BP approximation. We prove that the cluster expansion converges exponentially fast if an object called the *loop contribution* decays sufficiently fast with the loop size, giving a rigorous error bound on BP. We also provide a simple and efficient algorithm to compute the cluster expansion to arbitrary order. We demonstrate the efficacy of our method on the two-dimensional Ising model, where we find that our method significantly improves upon BP and existing corrective algorithms such as loop series expansion. Our work opens the door to a systematic theory of BP for tensor networks and its applications in decoding classical and quantum error-correcting codes and simulating quantum systems.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Beyond Belief Propagation: Cluster-Corrected Tensor Network Contraction

    **Paper:** [arXiv:2510.02290](https://arxiv.org/abs/2510.02290) (Midha & Zhang)

    This notebook follows the paper’s pipeline: **belief propagation** → **normalized tensors**
    $\tilde T_v = T_v / Z^{(v)}$ (Eq. (12)) → **loop corrections** $Z_\ell$ (Def. II.2, Eq. (9)) →
    **cluster expansion** of $\mathcal{F}(\tilde{\mathcal{T}}) = \sum_{\text{connected } \mathbf{W}}
    \phi(\mathbf{W}) Z_{\mathbf{W}}$ (Lemma III.1, Eq. (17)), then
    $\mathcal{F}(\mathcal{T}) = \mathcal{F}(\tilde{\mathcal{T}}) + \sum_v \ln Z^{(v)}$ (Eq. (13)).

    **Thermodynamic convention (Sec. V):** free energy density
    $f = -\beta^{-1} \ln \mathcal{Z} / N$.

    **Outline:** (1) Introduction & BP — (2) Loop series (additive, Lemma II.2) — (3) Cluster expansion
    — (4) Algorithm — (5) 2D Ising benchmarks (Figs. 4–5 style) — (6) Discussion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Sec. I — Contributions and loop series vs cluster expansion

    The paper’s two main threads are: **(1)** rigorous control of the error between the exact
    contraction $\mathcal{Z}$ and the BP value, and **(2)** a **cluster expansion** that
    systematically improves BP with **exponential convergence** in the truncation order when
    loop weights $|Z_\ell|$ decay fast enough (Theorem III.1).

    The **naïve loop series** (Lemma II.2) expands the **partition function** itself:
    $\mathcal{Z} = Z_0 + \sum_{\ell \in \mathcal{L}_G} Z_\ell$ (Eq. (10)). Truncating in $|\ell|$
    is spoiled by **combinatorial growth of disconnected** loop configurations on 2D lattices
    (Sec. II.4).

    The **cluster expansion** instead expands the **free energy** $\mathcal{F} \sim \ln \mathcal{Z}$:
    after normalizing tensors, only **connected clusters** of loops contribute (Lemma III.1), with
    **Ursell** weights $\phi(\mathbf{W})$ (Eq. (16)). Extensivity of $\mathcal{F}$ matches additive
    series: local perturbations change $\mathcal{F}$ by $O(1)$, not multiplicative factors across
    $\mathcal{Z}$ (Sec. III.1).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Motivation: tensor contraction and hardness

    A **tensor network** $(\{T_v\}, V, E)$ contracts to a scalar $\mathcal{Z}$. Exact contraction is
    **#P-hard** in general. **BP** gives $O(N)$-per-iteration message passing; this notebook
    implements BP and the **cluster-corrected** free energy on small 2D Ising networks.
    """)
    return


@app.cell
def _(mo):
    chi_slider = mo.ui.slider(2, 8, value=2, label="Bond dimension χ")
    mo.md(f"**Bond dimension (illustrative):** {chi_slider}")
    return (chi_slider,)


@app.cell
def _(chi_slider, layout_graph, nx, plt):
    _G = nx.grid_2d_graph(3, 3)
    _pos = layout_graph(_G)
    _fig, _ax = plt.subplots(1, 1, figsize=(5, 4))
    nx.draw(
        _G, _pos, ax=_ax, with_labels=False, node_color="steelblue",
        node_size=400, edge_color="gray", width=2,
    )
    _chi = chi_slider.value
    _ax.set_title(
        f"3×3 lattice TN  |  χ = {_chi}\n"
        "Exact contraction cost is exponential in |E| at fixed χ",
        fontsize=11,
    )
    plt.tight_layout()
    _fig
    return


@app.cell
def _(np):
    def ising_partition_brute_force(
        beta: float,
        Lx: int,
        Ly: int,
        J: float = 1.0,
        periodic: bool = False,
    ) -> float:
        """Exhaustive Z for Ising; periodic = torus (small L only)."""
        N = Lx * Ly
        Z = 0.0
        for cfg in range(2**N):
            spins = np.array(
                [2 * ((cfg >> i) & 1) - 1 for i in range(N)]
            ).reshape(Lx, Ly)
            e = 0.0
            for i in range(Lx):
                for j in range(Ly):
                    if periodic:
                        e -= J * spins[i, j] * spins[i, (j + 1) % Ly]
                        e -= J * spins[i, j] * spins[(i + 1) % Lx, j]
                    else:
                        if j + 1 < Ly:
                            e -= J * spins[i, j] * spins[i, j + 1]
                        if i + 1 < Lx:
                            e -= J * spins[i, j] * spins[i + 1, j]
            Z += np.exp(-beta * e)
        return Z

    return (ising_partition_brute_force,)


@app.cell
def _(np, nx):
    from typing import Any, Self

    class TensorNetwork:
        """Tensor network on a graph; legs ordered by sorted neighbors."""

        graph: nx.Graph
        chi: int
        nodes: list[Any]
        edges: list[tuple[Any, Any]]
        tensors: dict[Any, np.ndarray]

        def __init__(self, graph: nx.Graph, chi: int) -> None:
            self.graph = graph
            self.chi = chi
            self.nodes = list(graph.nodes())
            self.edges = list(graph.edges())
            self.tensors = {}

        def set_tensor(self, node: Any, tensor: np.ndarray) -> None:
            self.tensors[node] = tensor

        def neighbors(self, node: Any) -> list[Any]:
            return sorted(self.graph.neighbors(node))

        def copy_empty_tensors(self) -> Self:
            """Shallow graph copy; tensors filled by caller."""
            out = TensorNetwork(self.graph, self.chi)
            for n in self.nodes:
                out.tensors[n] = self.tensors[n].copy()
            return out

        def contract_exact(self) -> np.floating:
            edge_to_idx = {}
            idx = 0
            for e in self.edges:
                edge_to_idx[e] = idx
                edge_to_idx[(e[1], e[0])] = idx
                idx += 1
            operands = []
            subscripts_list = []
            for node in self.nodes:
                nbrs = self.neighbors(node)
                subs = [edge_to_idx[(node, nbr)] for nbr in nbrs]
                subscripts_list.append(subs)
                operands.append(self.tensors[node])
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            input_subs = ["".join(chars[s] for s in subs) for subs in subscripts_list]
            einsum_str = ",".join(input_subs) + "->"
            return np.einsum(einsum_str, *operands)

    return (TensorNetwork,)


@app.cell
def _(TensorNetwork, np, nx):
    def w_paper(s: int, x: int, beta: float, J: float = 1.0) -> float:
        """Paper Eq. after (26): w(s,x,β) with s ∈ {+1,-1}, x ∈ {0,1}."""
        b = beta * J
        ch = np.cosh(b)
        th = np.tanh(b)
        if x == 0:
            return float(np.sqrt(ch))
        return float(np.sqrt(ch) * s * np.sqrt(th))

    def make_grid_graph(Lx: int, Ly: int, periodic: bool = False) -> nx.Graph:
        if not periodic:
            return nx.grid_2d_graph(Lx, Ly)
        G = nx.Graph()
        for i in range(Lx):
            for j in range(Ly):
                G.add_node((i, j))
        for i in range(Lx):
            for j in range(Ly):
                G.add_edge((i, j), (i, (j + 1) % Ly))
                G.add_edge((i, j), ((i + 1) % Lx, j))
        return G

    def make_ising_tn(Lx, Ly, beta, J=1.0, periodic=False):
        """2D Ising TN with paper bond weights (χ=2)."""
        G = make_grid_graph(Lx, Ly, periodic=periodic)
        tn = TensorNetwork(G, chi=2)
        spins = (1, -1)
        for node in G.nodes():
            nbrs = sorted(G.neighbors(node))
            deg = len(nbrs)
            T = np.zeros((2,) * deg)
            for sigma in spins:
                contrib = np.ones(1)
                for _ in range(deg):
                    outer = []
                    for x in (0, 1):
                        outer.append(w_paper(sigma, x, beta, J))
                    contrib = np.outer(contrib, outer).flatten()
                T += contrib.reshape((2,) * deg)
            tn.set_tensor(node, T)
        return tn

    def make_ising_ring_tn(n_sites: int, beta: float, J: float = 1.0) -> TensorNetwork:
        """1D ring C_n, χ=2 Ising with the same paper bond weights as ``make_ising_tn``."""
        G = nx.cycle_graph(n_sites)
        tn = TensorNetwork(G, chi=2)
        spins = (1, -1)
        for node in G.nodes():
            nbrs = sorted(G.neighbors(node))
            deg = len(nbrs)
            T = np.zeros((2,) * deg)
            for sigma in spins:
                contrib = np.ones(1)
                for _ in range(deg):
                    outer = []
                    for x in (0, 1):
                        outer.append(w_paper(sigma, x, beta, J))
                    contrib = np.outer(contrib, outer).flatten()
                T += contrib.reshape((2,) * deg)
            tn.set_tensor(node, T)
        return tn

    return make_ising_ring_tn, make_ising_tn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Sec. II.1–II.2 — BP as mean field on tensor networks

    **Messages** $\mu_{v\to w}$ approximate rank-one environments. On trees BP is exact; on loopy
    graphs it implements a **Bethe** / mean-field picture (locally tree-like). Updates contract
    $T_v$ with incoming messages (Eq. (24)); we use damping, random init, and an $\ell_2$ residual
    check (Eq. (25)).
    """)
    return


@app.cell
def _(TensorNetwork, np):
    from typing import Any

    def belief_propagation(
        tn: TensorNetwork,
        max_iter: int = 500,
        tol: float = 1e-9,
        damping: float = 0.3,
        noise_std: float = 1e-6,
        verbose: bool = False,
    ) -> tuple[dict[tuple[Any, Any], np.ndarray], bool, list[float]]:
        chi, graph = tn.chi, tn.graph
        messages: dict[tuple[Any, Any], np.ndarray] = {}
        for u, v in graph.edges():
            messages[(u, v)] = np.random.rand(chi) + 0.1
            messages[(v, u)] = np.random.rand(chi) + 0.1

        def normalize_pair(u: Any, v: Any) -> None:
            ip = messages[(u, v)] @ messages[(v, u)]
            if ip > 0:
                s = np.sqrt(ip)
                messages[(u, v)] /= s
                messages[(v, u)] /= s

        for u, v in graph.edges():
            normalize_pair(u, v)

        history = []
        for iteration in range(max_iter):
            max_change = 0.0
            max_res = 0.0
            new_messages = {}
            for u, v in list(graph.edges()) + [(v, u) for u, v in graph.edges()]:
                nbrs_u = tn.neighbors(u)
                T = tn.tensors[u].copy()
                deg = len(nbrs_u)
                vectors = []
                for w in nbrs_u:
                    if w == v:
                        vectors.append(None)
                    else:
                        vectors.append(messages[(w, u)])
                result = T
                for leg_idx in range(deg - 1, -1, -1):
                    if vectors[leg_idx] is not None:
                        result = np.tensordot(result, vectors[leg_idx], axes=([leg_idx], [0]))
                new_msg = result.flatten()
                nrm = np.linalg.norm(new_msg)
                if nrm > 0:
                    new_msg /= nrm
                new_messages[(u, v)] = new_msg
                max_res = max(max_res, np.linalg.norm(new_msg - messages[(u, v)]))

            for key in new_messages:
                old = messages[key]
                raw = new_messages[key]
                new = raw + noise_std * np.random.randn(*raw.shape)
                nrm = np.linalg.norm(new)
                if nrm > 0:
                    new /= nrm
                if old @ new < 0:
                    new = -new
                upd = (1 - damping) * old + damping * new
                nrm = np.linalg.norm(upd)
                if nrm > 0:
                    upd /= nrm
                max_change = max(max_change, np.linalg.norm(upd - old))
                messages[key] = upd

            for u, v in graph.edges():
                normalize_pair(u, v)

            history.append(max_change)
            if verbose and iteration % 50 == 0:
                print(f"  BP {iteration}: Δ={max_change:.2e} res={max_res:.2e}")
            if max_change < tol and max_res < 10 * tol:
                return messages, True, history
        return messages, max_change < tol * 50, history

    return (belief_propagation,)


@app.cell
def _(TensorNetwork, np):
    from typing import Any

    def bp_contract_vertex(
        tn: TensorNetwork,
        messages: dict[tuple[Any, Any], np.ndarray],
        node: Any,
    ) -> float:
        nbrs = tn.neighbors(node)
        T = tn.tensors[node]
        for w in nbrs:
            T = np.tensordot(T, messages[(w, node)], axes=([0], [0]))
        return float(T)

    def compute_bp_partition(
        tn: TensorNetwork,
        messages: dict[tuple[Any, Any], np.ndarray],
    ) -> float:
        log_Z = 0.0
        for node in tn.nodes:
            z_v = bp_contract_vertex(tn, messages, node)
            log_Z += np.log(abs(z_v) + 1e-30)
        for u, v in tn.edges:
            z_uv = messages[(u, v)] @ messages[(v, u)]
            log_Z -= np.log(abs(z_uv) + 1e-30)
        return float(log_Z)

    def local_z_factors(
        tn: TensorNetwork,
        messages: dict[tuple[Any, Any], np.ndarray],
    ) -> dict[Any, float]:
        return {v: bp_contract_vertex(tn, messages, v) for v in tn.nodes}

    def normalize_tensors(
        tn: TensorNetwork,
        messages: dict[tuple[Any, Any], np.ndarray],
    ) -> tuple[TensorNetwork, float, dict[Any, float]]:
        """T̃_v = T_v / Z^{(v)} (Eq. (12)); return offset Σ ln Z^{(v)}."""
        zv = local_z_factors(tn, messages)
        ttn = tn.copy_empty_tensors()
        log_off = 0.0
        for v in tn.nodes:
            z = zv[v]
            log_off += np.log(abs(z) + 1e-30)
            ttn.tensors[v] = tn.tensors[v] / z
        return ttn, log_off, zv

    def free_energy_density(log_Z: float, beta: float, N: int) -> float:
        """f = -β⁻¹ ln Z / N (paper Sec. V)."""
        return -log_Z / (beta * N)

    return compute_bp_partition, free_energy_density, normalize_tensors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Sec. II.3 — Projectors, Def. II.1 (generalized loops), Lemma II.2

    Expand $\mathbb{1}$ on each bond into BP vacuum plus **orthogonal** complement $\mathcal{P}^\perp$.
    A **generalized loop** (Def. II.1) is an edge-induced subgraph in which **every vertex has
    degree at least two** in that subgraph (no dangling excitations; Lemma II.1).

    **Lemma II.2 (additive loop series):**
    $$\mathcal{Z}(\mathcal{T}) = Z_0 + \sum_{\ell \in \mathcal{L}_G} Z_\ell$$
    with only generalized loops contributing. This notebook uses the paper’s **additive** form;
    $Z_0$ is the all-vacuum contribution (we identify it with the BP Bethe partition function
    $e^{\mathcal{F}_{\mathrm{BP}}}$ in verification cells).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sec. II.4 — Why truncating the loop series fails

    Truncating $\sum_{|\ell|\le m} Z_\ell$ still sums **disconnected** loops; on an $L\times L$
    lattice the count of disjoint plaquettes grows combinatorially in $L^2$, overwhelming
    exponential decay of $|Z_\ell|$ (Fig. 2 in the paper).
    """)
    return


@app.cell
def _(combinations, nx):
    from typing import Any

    def is_paper_generalized_loop(edge_list: list[tuple[Any, Any]]) -> bool:
        """Def. II.1: induced subgraph has min degree ≥ 2."""
        if not edge_list:
            return False
        H = nx.Graph()
        H.add_edges_from(edge_list)
        return all(H.degree(v) >= 2 for v in H.nodes())

    def edge_key(u, v):
        return (u, v) if u < v else (v, u)

    def canonical_loop_edges(edge_list):
        return frozenset(edge_key(u, v) for u, v in edge_list)

    def enumerate_connected_paper_loops(graph, max_weight, max_loops=6000):
        edges = sorted({edge_key(u, v) for u, v in graph.edges()})
        n_e = len(edges)
        seen = set()
        out = []
        for r in range(1, min(max_weight, n_e) + 1):
            for idxs in combinations(range(n_e), r):
                el = [edges[i] for i in idxs]
                if not is_paper_generalized_loop(el):
                    continue
                H = nx.Graph()
                H.add_edges_from(el)
                if not nx.is_connected(H):
                    continue
                key = frozenset(el)
                if key in seen:
                    continue
                seen.add(key)
                out.append(list(el))
                if len(out) >= max_loops:
                    return out
        return out

    def enumerate_simple_cycles_as_edges(graph, max_length):
        cycles = []
        for cyc in nx.simple_cycles(graph, length_bound=max_length):
            if len(cyc) < 3:
                continue
            el = []
            for i in range(len(cyc)):
                u, v = cyc[i], cyc[(i + 1) % len(cyc)]
                el.append(edge_key(u, v))
            el = sorted(set(el))
            if el not in cycles:
                cycles.append(el)
        return cycles

    return enumerate_connected_paper_loops, enumerate_simple_cycles_as_edges


@app.cell
def _(enumerate_connected_paper_loops, layout_graph, mo, nx, plt):
    _G = nx.cycle_graph(4)
    _G.add_edge(0, 2)
    _loops = enumerate_connected_paper_loops(_G, max_weight=8, max_loops=50)
    _fig, _axes = plt.subplots(1, min(len(_loops), 6), figsize=(3 * min(len(_loops), 6), 3))
    if not hasattr(_axes, "__len__"):
        _axes = [_axes]
    _pos = layout_graph(_G)
    for idx, (loop, ax) in enumerate(zip(_loops[:6], _axes)):
        nx.draw(_G, _pos, ax=ax, with_labels=True, node_color="lightgray", node_size=300)
        nx.draw_networkx_edges(_G, _pos, edgelist=loop, edge_color="red", width=3, ax=ax)
        ax.set_title(f"{len(loop)} edges", fontsize=9)
    plt.suptitle("Connected generalized loops (min degree ≥ 2)", fontsize=11)
    plt.tight_layout()
    mo.md("**Def. II.1:** connected edge sets whose induced subgraph has **minimum degree ≥ 2**.")
    _fig
    return


@app.cell
def _(np):
    def compute_edge_basis(messages, u, v, chi):
        mu_uv = messages[(u, v)].copy()
        mu_uv /= np.linalg.norm(mu_uv) + 1e-15
        mu_vu = messages[(v, u)].copy()
        mu_vu /= np.linalg.norm(mu_vu) + 1e-15

        def gram_schmidt(v0, dim):
            basis = np.zeros((dim, dim))
            basis[0] = v0 / (np.linalg.norm(v0) + 1e-15)
            for i in range(1, dim):
                vec = np.zeros(dim)
                vec[i] = 1.0
                for j in range(i):
                    vec -= (vec @ basis[j]) * basis[j]
                nrm = np.linalg.norm(vec)
                if nrm < 1e-12:
                    vec = np.random.randn(dim)
                    for j in range(i):
                        vec -= (vec @ basis[j]) * basis[j]
                    nrm = np.linalg.norm(vec)
                basis[i] = vec / (nrm + 1e-15)
            return basis

        return gram_schmidt(mu_uv, chi), gram_schmidt(mu_vu, chi)

    return (compute_edge_basis,)


@app.cell
def _(compute_edge_basis, np):
    def compute_loop_tensor(tn, messages, loop_edges):
        """Loop correction Z_ℓ for χ=2 (excited direction ⊥ BP message)."""
        chi = tn.chi
        if chi != 2:
            raise NotImplementedError("This notebook implements χ=2 (Ising).")
        loop_degree = {}
        for u, v in loop_edges:
            loop_degree[u] = loop_degree.get(u, 0) + 1
            loop_degree[v] = loop_degree.get(v, 0) + 1
        loop_vertices = set(loop_degree.keys())
        edge_bases = {}
        for u, v in loop_edges:
            buv, bvu = compute_edge_basis(messages, u, v, chi)
            edge_bases[(u, v)] = buv
            edge_bases[(v, u)] = bvu
        loop_es = set()
        for u, v in loop_edges:
            loop_es.add((u, v))
            loop_es.add((v, u))
        Z_l = 1.0
        for node in loop_vertices:
            nbrs = tn.neighbors(node)
            T = tn.tensors[node].copy()
            vecs = []
            for nbr in nbrs:
                if (nbr, node) in loop_es or (node, nbr) in loop_es:
                    vecs.append(edge_bases[(nbr, node)][1])
                else:
                    vecs.append(messages[(nbr, node)])
            res = T
            for vec in vecs:
                res = np.tensordot(res, vec, axes=([0], [0]))
            Z_l *= float(res)
        return Z_l

    return (compute_loop_tensor,)


@app.cell
def _(
    belief_propagation,
    compute_bp_partition,
    compute_loop_tensor,
    enumerate_connected_paper_loops,
    make_ising_tn,
    mo,
    np,
):
    _tn = make_ising_tn(3, 3, beta=0.3, periodic=False)
    _msgs, _, _ = belief_propagation(_tn, max_iter=500, damping=0.2)
    _logZ_bp = compute_bp_partition(_tn, _msgs)
    _Z0 = np.exp(_logZ_bp)
    _Z_exact = float(_tn.contract_exact())
    _loops = enumerate_connected_paper_loops(_tn.graph, max_weight=12, max_loops=4000)
    _s = 0.0
    for _lp in _loops:
        _s += compute_loop_tensor(_tn, _msgs, _lp)
    _Z_rec = _Z0 + _s
    mo.md(f"""
    ### Lemma II.2 check (3×3 open, β=0.3)

    | | |
    |---|---|
    | $Z_{{\\rm exact}}$ | {_Z_exact:.8f} |
    | $Z_0 \\approx e^{{\\mathcal{{F}}_{{\\rm BP}}}}$ | {_Z0:.8f} |
    | $\\sum_{{\\ell\\neq 0}} Z_\\ell$ (connected loops, $\\|\\ell\\|\\le 12$) | {_s:.8f} |
    | $Z_0 + \\sum Z_\\ell$ | {_Z_rec:.8f} |
    | rel. err. | {abs(_Z_rec-_Z_exact)/abs(_Z_exact):.3e} |

    Truncation and finite loop list introduce error; full $\\mathcal{{L}}_G$ + all disconnects are needed for equality.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Sec. III.2 — Cluster expansion (Defs. III.1–III.5)

    **Compatible** loops share no vertex or edge (Def. III.1). A **cluster** is a multiset of loops
    (Def. III.2–III.3) with weight $|\mathbf{W}| = \sum_i \eta_i |\ell_i|$ and
    $Z_{\mathbf{W}} = \prod_i Z_{\ell_i}^{\eta_i}$.

    The **interaction graph** $G_{\mathbf{W}}$ (Def. III.4) has a vertex per loop **instance**;
    an edge connects instances that are **incompatible** or **identical** (same underlying loop).
    **Lemma III.1:** only **connected** $G_{\mathbf{W}}$ contribute,
    $$\mathcal{F}(\tilde{\mathcal{T}}) = \sum_{\mathbf{W}\ \mathrm{connected}} \phi(\mathbf{W}) Z_{\mathbf{W}}.$$
    For $\eta_{\mathbf{W}}>1$, $\phi(\mathbf{W}) = \frac{1}{\mathbf{W}!}\sum_{C} (-1)^{|E(C)|}$ over **spanning connected** subgraphs $C$ of $G_{\mathbf{W}}$ (Eq. (16)).

    **Theorem III.1 (informal):** if $|Z_\ell| \le e^{-c|\ell|}$ with $c>c_0(\Delta)$, the series
    converges absolutely and truncation error is $O(n e^{-d(m+1)})$ (Eqs. (19)–(20)).
    """)
    return


@app.cell
def _(nx):
    def phi_cluster_from_graph(G_W, factorial_W):
        """φ = (1/W!) * sum_{spanning connected} (-1)^|E|."""
        n = G_W.number_of_nodes()
        if n == 0:
            return 0.0
        if n == 1:
            return 1.0 / factorial_W
        edges = list(G_W.edges())
        m = len(edges)
        s = 0
        for mask in range(1 << m):
            el = [edges[i] for i in range(m) if (mask >> i) & 1]
            H = nx.Graph()
            H.add_nodes_from(range(n))
            H.add_edges_from(el)
            if nx.is_connected(H) and H.number_of_nodes() == n:
                s += (-1) ** len(el)
        return s / factorial_W

    def build_interaction_graph(loop_vertex_sets, same_cluster_ids):
        """
        loop_vertex_sets: list of frozenset of graph vertices for each loop instance
        same_cluster_ids: list of int loop-type id (identical loops share id)
        """
        n = len(loop_vertex_sets)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if same_cluster_ids[i] == same_cluster_ids[j]:
                    G.add_edge(i, j)
                elif loop_vertex_sets[i] & loop_vertex_sets[j]:
                    G.add_edge(i, j)
        return G

    def cluster_factorial(multiplicities):
        import math

        p = 1.0
        for eta in multiplicities:
            p *= math.factorial(int(eta))
        return p

    return build_interaction_graph, phi_cluster_from_graph


@app.cell
def _(Counter, build_interaction_graph, nx, phi_cluster_from_graph):
    import math
    from itertools import combinations_with_replacement

    def cluster_expansion_sum(loops_edge_list, loop_Z, max_weight, max_cluster_size=4):
        """Σ_{connected W} φ(W) Z_W (Lemma III.1); Z_ℓ evaluated on T̃."""
        verts = []
        for el in loops_edge_list:
            vs = set()
            for u, v in el:
                vs.add(u)
                vs.add(v)
            verts.append(frozenset(vs))
        nL = len(loops_edge_list)
        total_F = 0.0
        for k in range(1, max_cluster_size + 1):
            for idxs in combinations_with_replacement(range(nL), k):
                if sum(len(loops_edge_list[i]) for i in idxs) > max_weight:
                    continue
                cnt = Counter(idxs)
                Gw = build_interaction_graph([verts[i] for i in idxs], list(idxs))
                if not nx.is_connected(Gw):
                    continue
                fact = math.prod(math.factorial(int(eta)) for eta in cnt.values())
                phi = phi_cluster_from_graph(Gw, fact)
                zpow = 1.0
                for i in idxs:
                    zpow *= loop_Z[i]
                total_F += phi * zpow
        return total_F

    return (cluster_expansion_sum,)


@app.cell
def _(
    cluster_expansion_sum,
    compute_loop_tensor,
    enumerate_connected_paper_loops,
    enumerate_simple_cycles_as_edges,
    normalize_tensors,
):
    def collect_loops(graph, max_weight, cycles_only=False, max_loops=8000):
        if cycles_only or graph.number_of_edges() > 20:
            return enumerate_simple_cycles_as_edges(graph, max_length=max_weight)
        return enumerate_connected_paper_loops(graph, max_weight, max_loops=max_loops)

    def log_partition_cluster(tn, messages, max_weight, max_cluster_size=4, cycles_only=False):
        """log Z ≈ ℱ(T̃) + Σ_v ln Z^(v) with ℱ from cluster sum (Lemma III.1, Eq. (13))."""
        loops = collect_loops(tn.graph, max_weight, cycles_only=cycles_only)
        ttn, log_off, _ = normalize_tensors(tn, messages)
        loop_Z = [compute_loop_tensor(ttn, messages, el) for el in loops]
        F_tilde = cluster_expansion_sum(loops, loop_Z, max_weight, max_cluster_size)
        return F_tilde + log_off, loops, loop_Z

    return (log_partition_cluster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Sec. III.3 — Toy ring (normalized network)

    On a 1D ring with a single loop $\ell$, after normalization one expects
    $\mathcal{Z}(\tilde{\mathcal{T}})\approx 1+Z_\ell$ and $\tilde{\mathcal{F}}=\ln\mathcal{Z}(\tilde{\mathcal{T}})\approx\ln(1+Z_\ell)$.
    The **cluster expansion** (Lemma III.1) truncated at multiset size $K$ approaches $\ln\mathcal{Z}(\tilde{\mathcal{T}})$
    as $K$ grows; for this graph it tracks the Taylor series
    $\sum_{k\ge1} (-1)^{k+1} Z_\ell^k/k$ from Eq. (23)—distinct from the one-line **linked-cluster** expression
    $\ln(1+Z_\ell)$. The next cell fixes $C_n$, runs BP + normalization, and plots **errors vs. $K$** so convergence is visible.
    """)
    return


@app.cell
def _(
    belief_propagation,
    cluster_expansion_sum,
    compute_loop_tensor,
    make_ising_ring_tn,
    mo,
    normalize_tensors,
    np,
    plt,
):
    _n = 5
    _beta = 0.3
    _tn = make_ising_ring_tn(_n, _beta)
    _logZ_full = float(np.log(float(_tn.contract_exact())))
    _msgs, _bp_ok, _ = belief_propagation(
        _tn,
        max_iter=4000,
        tol=1e-12,
        damping=0.5,
        noise_std=0.0,
    )
    _ttn, _log_off, _ = normalize_tensors(_tn, _msgs)
    _Z_tilde = float(_ttn.contract_exact())
    _logZ_tilde = float(np.log(_Z_tilde))

    _ring_edges = []
    for _i in range(_n):
        _a, _b = _i, (_i + 1) % _n
        _ring_edges.append((_a, _b) if _a < _b else (_b, _a))
    _Z_ell = compute_loop_tensor(_ttn, _msgs, _ring_edges)
    _one_plus = 1.0 + _Z_ell
    _log_one_plus = float(np.log(_one_plus))

    _loops = [_ring_edges]
    _loop_Z = [_Z_ell]
    _K_max = 14
    _Ks = list(range(1, _K_max + 1))
    _err_cluster = []
    _err_taylor = []
    for _K in _Ks:
        _F_K = cluster_expansion_sum(
            _loops, _loop_Z, max_weight=9999, max_cluster_size=_K
        )
        _err_cluster.append(abs(_F_K - _logZ_tilde))
        _T_K = sum(
            ((-1) ** (_k + 1)) * (_Z_ell**_k) / _k for _k in range(1, _K + 1)
        )
        _err_taylor.append(abs(_T_K - _logZ_tilde))

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _axes[0].semilogy(
        _Ks,
        _err_cluster,
        "o-",
        ms=4,
        label=r"$|\tilde{\mathcal{F}}_K - \ln\tilde{Z}|$",
    )
    _axes[0].axhline(
        abs(_log_one_plus - _logZ_tilde),
        color="gray",
        ls=":",
        lw=1.2,
        label=r"$|\ln(1+Z_\ell)-\ln\tilde{Z}|$",
    )
    _axes[0].set_xlabel(r"max multiset size $K$")
    _axes[0].set_ylabel("absolute error")
    _axes[0].set_title("Cluster expansion vs truncation")
    _axes[0].legend(fontsize=8)
    _axes[0].grid(True, alpha=0.3)

    _axes[1].semilogy(
        _Ks,
        _err_taylor,
        "s-",
        ms=4,
        color="darkgreen",
        label=r"$|T_K - \ln\tilde{Z}|$",
    )
    _axes[1].axhline(
        abs(_log_one_plus - _logZ_tilde),
        color="gray",
        ls=":",
        lw=1.2,
        label=r"$|\ln(1+Z_\ell)-\ln\tilde{Z}|$",
    )
    _axes[1].set_xlabel(r"Taylor order $K$")
    _axes[1].set_ylabel("absolute error")
    _axes[1].set_title(r"Eq. (23): $T_K=\sum_{k=1}^K (-1)^{k+1} Z_\ell^k/k$")
    _axes[1].legend(fontsize=8)
    _axes[1].grid(True, alpha=0.3)
    plt.suptitle(
        rf"Sec. III.3 — $C_{_n}$ ring at $\beta={_beta}$: convergence in $K$",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()

    mo.md(
        f"""
    ### Toy ring numerics ($C_{_n}$, $\\beta={_beta}$)

    | | |
    |---|---|
    | $\\ln Z$ (unnormalized TN) | {_logZ_full:.12f} |
    | $\\ln\\tilde{{Z}}$ (normalized TN, exact) | {_logZ_tilde:.12f} |
    | $\\sum_v \\ln Z^{{(v)}}$ (offset) | {_log_off:.12f} |
    | BP converged | `{_bp_ok}` |
    | $Z_\\ell$ (single ring loop, $\\tilde{{T}}$) | {_Z_ell:.6e} |
    | $1+Z_\\ell$ | {_one_plus:.10f} |
    | $\\ln(1+Z_\\ell)$ | {_log_one_plus:.12f} |
    | $|\\tilde{{Z}}-(1+Z_\\ell)|\\,/\\,\\tilde{{Z}}$ | {abs(_Z_tilde - _one_plus) / max(abs(_Z_tilde), 1e-30):.2e} |
    | $|\\tilde{{\\mathcal{{F}}}}_{{{_K_max}}} - \\ln\\tilde{{Z}}|$ | {_err_cluster[-1]:.2e} |
    | $|T_{{{_K_max}}} - \\ln\\tilde{{Z}}|$ | {_err_taylor[-1]:.2e} |

    Both curves decay with $K$ when $|Z_\\ell|\\ll 1$: the cluster sum and the Taylor partial sums approach the same limit near $\\ln\\tilde{{Z}}$. The horizontal gray line is the gap between that limit and the one-line linked-cluster value $\\ln(1+Z_\\ell)$ when $\\tilde{{Z}}\\neq 1+Z_\\ell$ at finite $\\chi$ and BP messages.
    """
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Sec. IV — Algorithm (Fig. 3)

    1. **Enumerate** connected loops and (once per graph) connected clusters up to weight $m$
       (Appendix II; complexity $O(n\,e^{O(m)})$ per Lemma III.2).
    2. Run **BP** with damping, random init, optional noise; enforce small edge residual (Eq. (25)).
    3. **Normalize** $\tilde T_v=T_v/Z^{(v)}$ and store $\sum_v \ln Z^{(v)}$.
    4. For each cluster $\mathbf{W}$, compute $Z_{\mathbf{W}}$ and $\phi(\mathbf{W})$ from $G_{\mathbf{W}}$
       (Eq. (16)); **sum** to $\tilde F_m$ (Eq. (17)); output $\tilde F_m+\sum_v\ln Z^{(v)}$.
    5. Contributions parallelize over clusters; for PEPS-scale problems many $\phi$ are $1,-1,-\tfrac12$
       at modest $m$.
    """)
    return


@app.cell
def _(np, scipy):
    def onsager_free_energy(beta, J=1.0):
        k = 1.0 / (np.sinh(2 * beta * J) ** 2)

        def integrand(theta1, theta2):
            return np.log(
                np.cosh(2 * beta * J) ** 2
                - np.sinh(2 * beta * J) * (np.cos(theta1) + np.cos(theta2))
            )

        result, _ = scipy.integrate.dblquad(integrand, 0, np.pi, 0, np.pi)
        return -(np.log(2) + result / (2 * np.pi**2)) / beta

    return (onsager_free_energy,)


@app.cell
def _(
    belief_propagation,
    compute_bp_partition,
    free_energy_density,
    make_ising_tn,
    mo,
    np,
    onsager_free_energy,
    plt,
):
    _beta_c = 0.5 * np.log(1 + np.sqrt(2))
    _beta_bp = np.log(2.0) / 2.0
    _betas = np.linspace(0.1, 0.85, 40)
    _L = 5
    _N = _L * _L
    _f_bp = []
    _f_ons = [onsager_free_energy(b) for b in _betas]
    for _b in _betas:
        _tn = make_ising_tn(_L, _L, _b, periodic=False)
        _m, _, _ = belief_propagation(_tn, max_iter=400, damping=0.25)
        _lz = compute_bp_partition(_tn, _m)
        _f_bp.append(free_energy_density(_lz, _b, _N))
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(_betas, _f_ons, "k-", lw=2, label="Onsager (L→∞)")
    _ax.plot(_betas, _f_bp, "r--", lw=1.5, label=f"BP vacuum (L={_L})")
    _ax.axvline(_beta_c, color="gray", ls=":", label=f"βc (Onsager) ≈ {_beta_c:.3f}")
    _ax.axvline(_beta_bp, color="blue", ls=":", label=f"β_BP (Bethe z=4) ≈ {_beta_bp:.3f}")
    _ax.set_xlabel("β")
    _ax.set_ylabel(r"$f = -\beta^{-1}\ln Z/N$")
    _ax.set_title("Fig. 4(a)-style: BP vs Onsager (consistent f)")
    _ax.legend(fontsize=8)
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.md("**Same** $f=-\\beta^{-1}\\ln Z/N$ for BP and Onsager (Sec. V).")
    _fig
    return


@app.cell
def _(
    belief_propagation,
    free_energy_density,
    ising_partition_brute_force,
    log_partition_cluster,
    make_ising_tn,
    mo,
    np,
    plt,
):
    _L, _N = 4, 16
    _betas = np.linspace(0.26, 0.46, 25)
    _weights = [4, 6, 8]
    _beta_c = 0.5 * np.log(1 + np.sqrt(2))
    _beta_bp = np.log(2.0) / 2.0
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    for _mw in _weights:
        _errs = []
        for _b in _betas:
            _tn = make_ising_tn(_L, _L, _b, periodic=False)
            _Z = ising_partition_brute_force(_b, _L, _L, periodic=False)
            _f_ex = free_energy_density(np.log(_Z), _b, _N)
            _m, _, _ = belief_propagation(_tn, max_iter=450, damping=0.25)
            _logc, _, _ = log_partition_cluster(_tn, _m, _mw, max_cluster_size=3, cycles_only=True)
            _f_c = free_energy_density(_logc, _b, _N)
            _errs.append(abs(_f_c - _f_ex))
        _axes[1].semilogy(_betas, _errs, "-", label=f"cluster m≤{_mw}")
    _axes[1].axvline(_beta_c, color="gray", ls=":", alpha=0.7)
    _axes[1].axvline(_beta_bp, color="blue", ls=":", alpha=0.7)
    _axes[1].set_xlabel("β")
    _axes[1].set_ylabel(r"$|f_{\rm approx}-f_{\rm exact}|$")
    _axes[1].set_title(f"Fig. 4(c)-style: cluster error ({_L}×{_L}, cycle basis)")
    _axes[1].legend(fontsize=8)
    _axes[1].grid(True, alpha=0.3)
    _zoom = (_betas >= 0.28) & (_betas <= 0.42)
    for _mw in _weights:
        _fv = []
        for _b in _betas[_zoom]:
            _tn = make_ising_tn(_L, _L, _b, periodic=False)
            _Z = ising_partition_brute_force(_b, _L, _L, periodic=False)
            _f_ex = free_energy_density(np.log(_Z), _b, _N)
            _m, _, _ = belief_propagation(_tn, max_iter=450, damping=0.25)
            _logc, _, _ = log_partition_cluster(_tn, _m, _mw, max_cluster_size=3, cycles_only=True)
            _fv.append(free_energy_density(_logc, _b, _N))
        _axes[0].plot(_betas[_zoom], _fv, "-", label=f"m≤{_mw}")
    _fexz = [free_energy_density(np.log(ising_partition_brute_force(b, _L, _L)), b, _N) for b in _betas[_zoom]]
    _axes[0].plot(_betas[_zoom], _fexz, "k--", lw=2, label="exact")
    _axes[0].set_xlabel("β")
    _axes[0].set_ylabel(r"$f$")
    _axes[0].set_title("Fig. 4(b)-style: zoom")
    _axes[0].legend(fontsize=8)
    _axes[0].grid(True, alpha=0.3)
    plt.tight_layout()
    mo.md("**Benchmarks** use **simple cycles** as loop basis when $|E|>20$ (full Def. II.1 enumeration is exponential in $|E|$).")
    _fig
    return


@app.cell
def _(mo, np, plt):
    _beta_bp = np.log(2.0) / 2.0
    _Narr = np.array([10, 20, 30, 60, 120])
    _zw = 0.08
    _z0 = 1.2
    _w = np.array([4, 6, 8, 10])
    _loop_err = []
    _clus_err = []
    for _N in _Narr:
        _loop_err.append(
            abs(
                (np.log(_z0) + np.log(1 + _N * _zw) / _N)
                - (np.log(_z0) + _zw)
            )
        )
        _clus_err.append(0.02 * np.exp(-0.15 * (_w[-1] - 4)))
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.semilogy(_Narr, _loop_err, "s--", color="red", label="naïve loop density (Eq. 27 sketch)")
    _ax.axhline(_clus_err[0], color="blue", ls="-", lw=2, label="cluster (size-independent, schematic)")
    _ax.set_xlabel("N")
    _ax.set_ylabel(r"error $|f_{\rm approx}-f_{\rm dense}|$ (illustrative)")
    _ax.set_title("Fig. 4(d)-style: loop vs cluster scaling (analytic cartoon)")
    _ax.legend(fontsize=8)
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.md(
        r"**Eq. (27)–(28):** $\frac1N\ln[Z_0(1+N Z_w)]\to \ln z_0$ kills loop corrections in the "
        r"thermodynamic limit; cluster expansion keeps an $O(1)$ density correction. Numeric values are **schematic**."
    )
    _fig
    return


@app.cell
def _(
    belief_propagation,
    compute_loop_tensor,
    enumerate_simple_cycles_as_edges,
    make_ising_tn,
    mo,
    normalize_tensors,
    np,
    plt,
):
    _L = 4
    _N = _L * _L
    _beta_bp = np.log(2.0) / 2.0
    _betas = np.linspace(0.15, 0.65, 28)
    _ws = [4, 6, 8]
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    for _w in _ws:
        _vals = []
        for _b in _betas:
            _tn = make_ising_tn(_L, _L, _b, periodic=False)
            _m, _, _ = belief_propagation(_tn, max_iter=450, damping=0.25)
            _tt, _, _ = normalize_tensors(_tn, _m)
            _cyc = [c for c in enumerate_simple_cycles_as_edges(_tn.graph, _w) if len(c) == _w]
            if not _cyc:
                _vals.append(np.nan)
                continue
            _zs = [abs(compute_loop_tensor(_tt, _m, c)) for c in _cyc]
            _vals.append(float(np.mean(_zs)))
        _axes[0].plot(_betas, _vals, "-", label=f"|ℓ|={_w}")
    _axes[0].axvline(_beta_bp, color="blue", ls=":", label="β_BP")
    _axes[0].set_xlabel("β")
    _axes[0].set_ylabel(r"mean $|Z_\ell|$")
    _axes[0].set_title("Fig. 5(a)-style")
    _axes[0].legend(fontsize=8)
    _axes[0].grid(True, alpha=0.3)
    for _b, _lab in [(0.2, "β=0.2"), (_beta_bp, "β_BP"), (0.5, "β=0.5")]:
        _tn = make_ising_tn(_L, _L, _b, periodic=False)
        _m, _, _ = belief_propagation(_tn, max_iter=450, damping=0.25)
        _tt, _, _ = normalize_tensors(_tn, _m)
        _byw = {}
        for _w in range(4, 11):
            _cyc = [c for c in enumerate_simple_cycles_as_edges(_tn.graph, _w) if len(c) == _w]
            if not _cyc:
                continue
            _byw[_w] = np.mean([abs(compute_loop_tensor(_tt, _m, c)) for c in _cyc])
        _axes[1].semilogy(
            list(_byw.keys()),
            list(_byw.values()),
            "o-",
            label=_lab,
        )
    _axes[1].set_xlabel("loop weight")
    _axes[1].set_ylabel(r"mean $|Z_\ell|$")
    _axes[1].set_title("Fig. 5(b)-style")
    _axes[1].legend(fontsize=8)
    _axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    mo.md("Loop weights on **normalized** tensors; cycle basis.")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Sec. VI — Discussion

    The cluster expansion formalizes **short-range** corrections to BP; convergence (Theorem III.1)
    is a **strong** assumption on loop decay. **Applications:** improved decoders for LDPC / QEC and
    more accurate TN contractions in simulation (see paper). Concurrent **TN loop cluster** work:
    Gray et al., [arXiv:2510.05647](https://arxiv.org/abs/2510.05647).

    **Magnetization** and other local observables require inserting modified tensors; we do not
    implement them here (Sec. V mentions Yang’s result only as reference physics).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## References

    1. Midha & Zhang, [arXiv:2510.02290](https://arxiv.org/abs/2510.02290)
    2. Evenbly et al., loop series for TN — [arXiv:2409.03108](https://arxiv.org/abs/2409.03108)
    3. Chertkov & Chernyak, loop calculus — Phys. Rev. E (2006)
    4. Kotecký & Preiss — Commun. Math. Phys. (1986)
    5. Onsager — Phys. Rev. (1944)
    """)
    return


if __name__ == "__main__":
    app.run()

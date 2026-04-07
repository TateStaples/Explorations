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

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import time
    from collections import Counter
    from collections.abc import Iterable
    from itertools import combinations, combinations_with_replacement
    from typing import Any, Self

    import matplotlib.pyplot as matplotlib_pyplot
    import networkx as networkx
    import numpy as numpy
    import scipy
    import math

    def layout_graph(graph: networkx.Graph, random_seed: int = 0) -> dict[Any, Any]:
        try:
            return networkx.nx_agraph.graphviz_layout(graph, prog="neato")
        except Exception:
            return networkx.spring_layout(graph, seed=random_seed, k=0.15)

    return (
        Any,
        Counter,
        Iterable,
        Self,
        combinations,
        combinations_with_replacement,
        layout_graph,
        math,
        matplotlib_pyplot,
        networkx,
        numpy,
        scipy,
    )


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

    **Notebook map (paper sections):** **I** Introduction — **II** Belief propagation (tensor
    networks, BP, loop series) — **III** Cluster expansion — **IV** Algorithm — **V** Benchmark on the
    2D Ising model — **VI** Discussion — **References**.
    """)
    return


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
    ---
    ## I Introduction

    ### I.1 Comparison with existing and concurrent work

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

    ### I.2 Organization

    Markdown headings mirror the paper: **§II** builds BP and the loop series; **§III** defines the
    cluster expansion; **§IV** states the algorithm; **§V** benchmarks the 2D Ising model (Figs. 4–5);
    **§VI** discusses scope and related work.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## II Belief propagation

    ### II.1 Tensor networks

    A **tensor network** $(\{T_v\}, V, E)$ contracts to a scalar $\mathcal{Z}$. Exact contraction is
    **#P-hard** in general. The cells below define `TensorNetwork`, small-lattice **exact** Ising
    partition sums, and `make_ising_tn` with the paper’s bond weights (after Eq. (26)). **BP** and
    the **cluster-corrected** free energy follow in **§II.2** onward.
    """)
    return


@app.cell
def _(mo):
    chi_slider = mo.ui.slider(2, 8, value=2, label="Bond dimension χ per bond")
    return (chi_slider,)


@app.cell
def _(chi_slider, layout_graph, matplotlib_pyplot, mo, networkx, numpy):
    """3×3 grid: χ controls how many parallel strands each bond is drawn with."""
    bond_dimension = int(chi_slider.value)

    lattice_graph = networkx.grid_2d_graph(3, 3)
    node_positions = layout_graph(lattice_graph)
    _lattice_demo_figure, _lattice_demo_axes = matplotlib_pyplot.subplots(
        1, 1, figsize=(5, 4.2)
    )

    for tail_vertex, head_vertex in lattice_graph.edges():
        point_tail = numpy.array([node_positions[tail_vertex][0], node_positions[tail_vertex][1]])
        point_head = numpy.array([node_positions[head_vertex][0], node_positions[head_vertex][1]])
        tangent = point_head - point_tail
        edge_length = float(numpy.linalg.norm(tangent))
        if edge_length < 1e-12:
            continue
        tangent /= edge_length
        normal = numpy.array([-tangent[1], tangent[0]])
        strand_spacing = 0.035 * edge_length
        for strand_index in range(bond_dimension):
            offset = (strand_index - (bond_dimension - 1) / 2.0) * strand_spacing
            strand_tail = point_tail + normal * offset
            strand_head = point_head + normal * offset
            _lattice_demo_axes.plot(
                [strand_tail[0], strand_head[0]],
                [strand_tail[1], strand_head[1]],
                color="#4a4a4a",
                lw=1.65,
                solid_capstyle="round",
                zorder=1,
            )

    networkx.draw_networkx_nodes(
        lattice_graph,
        node_positions,
        ax=_lattice_demo_axes,
        node_color="steelblue",
        node_size=440,
        # zorder=2,
    )
    _lattice_demo_axes.set_title(
        f"3×3 lattice TN — each bond drawn with χ = {bond_dimension} strands\n"
        r"(Ising cells below use χ = 2; PEPS uses larger χ)",
        fontsize=10,
    )
    _lattice_demo_axes.axis("off")
    matplotlib_pyplot.tight_layout()

    _caption = mo.md(
        f"**Bond dimension:** {chi_slider}. "
        f"Each edge is the **tensor contraction** over a χ-dimensional index; "
        f"the drawing shows **{bond_dimension}** parallel **virtual bonds** per graph edge."
    )
    mo.vstack([_caption, _lattice_demo_figure])
    return


@app.cell
def _(numpy):
    def ising_partition_brute_force(
        inverse_temperature: float,
        lattice_width: int,
        lattice_height: int,
        coupling_strength: float = 1.0,
        periodic_boundary: bool = False,
    ) -> float:
        """Exhaustive partition sum for Ising; periodic_boundary = torus (small lattices only)."""
        site_count = lattice_width * lattice_height
        partition_sum = 0.0
        for spin_configuration_index in range(2**site_count):
            spin_array = numpy.array(
                [
                    2 * ((spin_configuration_index >> site_index) & 1) - 1
                    for site_index in range(site_count)
                ]
            ).reshape(lattice_width, lattice_height)
            energy = 0.0
            for row_index in range(lattice_width):
                for column_index in range(lattice_height):
                    if periodic_boundary:
                        energy -= (
                            coupling_strength
                            * spin_array[row_index, column_index]
                            * spin_array[row_index, (column_index + 1) % lattice_height]
                        )
                        energy -= (
                            coupling_strength
                            * spin_array[row_index, column_index]
                            * spin_array[(row_index + 1) % lattice_width, column_index]
                        )
                    else:
                        if column_index + 1 < lattice_height:
                            energy -= (
                                coupling_strength
                                * spin_array[row_index, column_index]
                                * spin_array[row_index, column_index + 1]
                            )
                        if row_index + 1 < lattice_width:
                            energy -= (
                                coupling_strength
                                * spin_array[row_index, column_index]
                                * spin_array[row_index + 1, column_index]
                            )
            partition_sum += numpy.exp(-inverse_temperature * energy)
        return partition_sum

    return (ising_partition_brute_force,)


@app.cell
def _(Any, Self, networkx, numpy):
    class TensorNetwork:
        """Tensor network on a graph; legs ordered by sorted neighbors."""

        graph: networkx.Graph
        chi: int
        nodes: list[Any]
        edges: list[tuple[Any, Any]]
        tensors: dict[Any, numpy.ndarray]

        def __init__(self, graph: networkx.Graph, chi: int) -> None:
            self.graph = graph
            self.chi = chi
            self.nodes = list(graph.nodes())
            self.edges = list(graph.edges())
            self.tensors = {}

        def set_tensor(self, node: Any, tensor: numpy.ndarray) -> None:
            self.tensors[node] = tensor

        def neighbors(self, node: Any) -> list[Any]:
            return sorted(self.graph.neighbors(node))

        def copy_empty_tensors(self) -> Self:
            """Shallow graph copy; tensors filled by caller."""
            copied_network = TensorNetwork(self.graph, self.chi)
            for node in self.nodes:
                copied_network.tensors[node] = self.tensors[node].copy()
            return copied_network

        def contract_exact(self) -> numpy.floating:
            edge_index_by_pair: dict[tuple[Any, Any], int] = {}
            next_edge_index = 0
            for undirected_edge in self.edges:
                edge_index_by_pair[undirected_edge] = next_edge_index
                edge_index_by_pair[(undirected_edge[1], undirected_edge[0])] = (
                    next_edge_index
                )
                next_edge_index += 1
            tensor_operands: list[numpy.ndarray] = []
            leg_index_lists: list[list[int]] = []
            for node in self.nodes:
                neighbor_nodes = self.neighbors(node)
                leg_indices = [
                    edge_index_by_pair[(node, neighbor)] for neighbor in neighbor_nodes
                ]
                leg_index_lists.append(leg_indices)
                tensor_operands.append(self.tensors[node])
            einstein_chars = (
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            )
            input_subscript_strings = [
                "".join(einstein_chars[leg_index] for leg_index in leg_indices)
                for leg_indices in leg_index_lists
            ]
            einsum_expression = ",".join(input_subscript_strings) + "->"
            return numpy.einsum(einsum_expression, *tensor_operands)

    return (TensorNetwork,)


@app.cell
def _(TensorNetwork, networkx, numpy):
    def paper_bond_weight(
        physical_spin: int,
        virtual_index: int,
        inverse_temperature: float,
        coupling_strength: float = 1.0,
    ) -> float:
        """Paper Eq. after (26): w(s,x,β) with s ∈ {+1,-1}, x ∈ {0,1}."""
        scaled_coupling = inverse_temperature * coupling_strength
        cosh_term = numpy.cosh(scaled_coupling)
        tanh_term = numpy.tanh(scaled_coupling)
        if virtual_index == 0:
            return float(numpy.sqrt(cosh_term))
        return float(
            numpy.sqrt(cosh_term) * physical_spin * numpy.sqrt(tanh_term)
        )

    def make_grid_graph(
        lattice_width: int,
        lattice_height: int,
        periodic_boundary: bool = False,
    ) -> networkx.Graph:
        if not periodic_boundary:
            return networkx.grid_2d_graph(lattice_width, lattice_height)
        torus_graph = networkx.Graph()
        for row_index in range(lattice_width):
            for column_index in range(lattice_height):
                torus_graph.add_node((row_index, column_index))
        for row_index in range(lattice_width):
            for column_index in range(lattice_height):
                torus_graph.add_edge(
                    (row_index, column_index),
                    (row_index, (column_index + 1) % lattice_height),
                )
                torus_graph.add_edge(
                    (row_index, column_index),
                    ((row_index + 1) % lattice_width, column_index),
                )
        return torus_graph

    def make_ising_tn(
        lattice_width: int,
        lattice_height: int,
        inverse_temperature: float,
        coupling_strength: float = 1.0,
        bond_dimension: int = 2,
        periodic_boundary: bool = False,
    ) -> TensorNetwork:
        """2D Ising tensor network with paper bond weights (χ=2 default)."""
        interaction_graph = make_grid_graph(
            lattice_width, lattice_height, periodic_boundary=periodic_boundary
        )
        tensor_network = TensorNetwork(interaction_graph, chi=bond_dimension)
        physical_spins = (1, -1)
        for node in interaction_graph.nodes():
            neighbor_nodes = sorted(interaction_graph.neighbors(node))
            neighbor_count = len(neighbor_nodes)
            local_tensor = numpy.zeros((2,) * neighbor_count)
            for physical_spin in physical_spins:
                rank_one_contribution = numpy.ones(1)
                for _leg_slot in range(neighbor_count):
                    bond_factors: list[float] = []
                    for virtual_index in (0, 1):
                        bond_factors.append(
                            paper_bond_weight(
                                physical_spin,
                                virtual_index,
                                inverse_temperature,
                                coupling_strength,
                            )
                        )
                    rank_one_contribution = numpy.outer(
                        rank_one_contribution, bond_factors
                    ).flatten()
                local_tensor += rank_one_contribution.reshape((2,) * neighbor_count)
            tensor_network.set_tensor(node, local_tensor)
        return tensor_network

    def make_ising_ring_tn(
        ring_site_count: int,
        inverse_temperature: float,
        coupling_strength: float = 1.0,
    ) -> TensorNetwork:
        """1D ring C_n, χ=2 Ising with the same paper bond weights as ``make_ising_tn``."""
        cycle_graph = networkx.cycle_graph(ring_site_count)
        tensor_network = TensorNetwork(cycle_graph, chi=2)
        physical_spins = (1, -1)
        for node in cycle_graph.nodes():
            neighbor_nodes = sorted(cycle_graph.neighbors(node))
            neighbor_count = len(neighbor_nodes)
            local_tensor = numpy.zeros((2,) * neighbor_count)
            for physical_spin in physical_spins:
                rank_one_contribution = numpy.ones(1)
                for _leg_slot in range(neighbor_count):
                    bond_factors: list[float] = []
                    for virtual_index in (0, 1):
                        bond_factors.append(
                            paper_bond_weight(
                                physical_spin,
                                virtual_index,
                                inverse_temperature,
                                coupling_strength,
                            )
                        )
                    rank_one_contribution = numpy.outer(
                        rank_one_contribution, bond_factors
                    ).flatten()
                local_tensor += rank_one_contribution.reshape((2,) * neighbor_count)
            tensor_network.set_tensor(node, local_tensor)
        return tensor_network

    return make_ising_ring_tn, make_ising_tn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### II.2 Belief propagation

    **Messages** $\mu_{v\to w}$ approximate rank-one environments. On trees BP is exact; on loopy
    graphs it implements a **Bethe** / mean-field picture (locally tree-like). Updates contract
    $T_v$ with incoming messages (Eq. (24)); we use damping, random init, and an $\ell_2$ residual
    check (Eq. (25)). After convergence, `compute_bp_partition` and `normalize_tensors` implement the
    Bethe partition function and vertex factors $Z^{(v)}$ (Eqs. (4)–(5), (12)).
    """)
    return


@app.cell
def _(Any, TensorNetwork, numpy):
    def belief_propagation(
        tensor_network: TensorNetwork,
        max_iterations: int = 500,
        convergence_tolerance: float = 1e-9,
        damping_factor: float = 0.3,
        message_noise_standard_deviation: float = 1e-6,
        verbose: bool = False,
    ) -> tuple[dict[tuple[Any, Any], numpy.ndarray], bool, list[float]]:
        bond_dimension, interaction_graph = tensor_network.chi, tensor_network.graph
        messages: dict[tuple[Any, Any], numpy.ndarray] = {}
        for tail_vertex, head_vertex in interaction_graph.edges():
            messages[(tail_vertex, head_vertex)] = (
                numpy.random.rand(bond_dimension) + 0.1
            )
            messages[(head_vertex, tail_vertex)] = (
                numpy.random.rand(bond_dimension) + 0.1
            )

        def normalize_undirected_message_pair(
            tail_vertex: Any, head_vertex: Any
        ) -> None:
            inner_product = (
                messages[(tail_vertex, head_vertex)]
                @ messages[(head_vertex, tail_vertex)]
            )
            if inner_product > 0:
                scale = numpy.sqrt(inner_product)
                messages[(tail_vertex, head_vertex)] /= scale
                messages[(head_vertex, tail_vertex)] /= scale

        for tail_vertex, head_vertex in interaction_graph.edges():
            normalize_undirected_message_pair(tail_vertex, head_vertex)

        residual_history: list[float] = []
        for iteration_index in range(max_iterations):
            max_message_change = 0.0
            max_message_residual = 0.0
            proposed_messages: dict[tuple[Any, Any], numpy.ndarray] = {}
            directed_edges = list(interaction_graph.edges()) + [
                (head_vertex, tail_vertex)
                for tail_vertex, head_vertex in interaction_graph.edges()
            ]
            for tail_vertex, head_vertex in directed_edges:
                neighbors_of_tail = tensor_network.neighbors(tail_vertex)
                contracted_tensor = tensor_network.tensors[tail_vertex].copy()
                neighbor_count = len(neighbors_of_tail)
                incoming_message_vectors: list[numpy.ndarray | None] = []
                for neighbor_vertex in neighbors_of_tail:
                    if neighbor_vertex == head_vertex:
                        incoming_message_vectors.append(None)
                    else:
                        incoming_message_vectors.append(
                            messages[(neighbor_vertex, tail_vertex)]
                        )
                contraction_result = contracted_tensor
                for leg_index in range(neighbor_count - 1, -1, -1):
                    if incoming_message_vectors[leg_index] is not None:
                        contraction_result = numpy.tensordot(
                            contraction_result,
                            incoming_message_vectors[leg_index],
                            axes=([leg_index], [0]),
                        )
                proposed_message = contraction_result.flatten()
                proposed_norm = numpy.linalg.norm(proposed_message)
                if proposed_norm > 0:
                    proposed_message /= proposed_norm
                proposed_messages[(tail_vertex, head_vertex)] = proposed_message
                max_message_residual = max(
                    max_message_residual,
                    numpy.linalg.norm(
                        proposed_message - messages[(tail_vertex, head_vertex)]
                    ),
                )

            for message_key in proposed_messages:
                previous_message = messages[message_key]
                raw_proposed = proposed_messages[message_key]
                noisy_candidate = raw_proposed + message_noise_standard_deviation * numpy.random.randn(
                    *raw_proposed.shape
                )
                candidate_norm = numpy.linalg.norm(noisy_candidate)
                if candidate_norm > 0:
                    noisy_candidate /= candidate_norm
                if previous_message @ noisy_candidate < 0:
                    noisy_candidate = -noisy_candidate
                damped_update = (
                    (1 - damping_factor) * previous_message
                    + damping_factor * noisy_candidate
                )
                update_norm = numpy.linalg.norm(damped_update)
                if update_norm > 0:
                    damped_update /= update_norm
                max_message_change = max(
                    max_message_change,
                    numpy.linalg.norm(damped_update - previous_message),
                )
                messages[message_key] = damped_update

            for tail_vertex, head_vertex in interaction_graph.edges():
                normalize_undirected_message_pair(tail_vertex, head_vertex)

            residual_history.append(max_message_change)
            if verbose and iteration_index % 50 == 0:
                print(
                    f"  BP {iteration_index}: Δ={max_message_change:.2e} "
                    f"res={max_message_residual:.2e}"
                )
            if (
                max_message_change < convergence_tolerance
                and max_message_residual < 10 * convergence_tolerance
            ):
                return messages, True, residual_history
        return (
            messages,
            max_message_change < convergence_tolerance * 50,
            residual_history,
        )

    return (belief_propagation,)


@app.cell
def _(Any, TensorNetwork, numpy):
    def bp_contract_vertex(
        tensor_network: TensorNetwork,
        messages: dict[tuple[Any, Any], numpy.ndarray],
        node: Any,
    ) -> float:
        neighbor_nodes = tensor_network.neighbors(node)
        contracted_tensor = tensor_network.tensors[node]
        for neighbor_vertex in neighbor_nodes:
            contracted_tensor = numpy.tensordot(
                contracted_tensor,
                messages[(neighbor_vertex, node)],
                axes=([0], [0]),
            )
        return float(contracted_tensor)

    def compute_bp_partition(
        tensor_network: TensorNetwork,
        messages: dict[tuple[Any, Any], numpy.ndarray],
    ) -> float:
        log_partition_function = 0.0
        for node in tensor_network.nodes:
            vertex_normalization = bp_contract_vertex(
                tensor_network, messages, node
            )
            log_partition_function += numpy.log(
                abs(vertex_normalization) + 1e-30
            )
        for tail_vertex, head_vertex in tensor_network.edges:
            edge_inner_product = (
                messages[(tail_vertex, head_vertex)]
                @ messages[(head_vertex, tail_vertex)]
            )
            log_partition_function -= numpy.log(
                abs(edge_inner_product) + 1e-30
            )
        return float(log_partition_function)

    def local_z_factors(
        tensor_network: TensorNetwork,
        messages: dict[tuple[Any, Any], numpy.ndarray],
    ) -> dict[Any, float]:
        return {
            node: bp_contract_vertex(tensor_network, messages, node)
            for node in tensor_network.nodes
        }

    def normalize_tensors(
        tensor_network: TensorNetwork,
        messages: dict[tuple[Any, Any], numpy.ndarray],
    ) -> tuple[TensorNetwork, float, dict[Any, float]]:
        """T̃_v = T_v / Z^{(v)} (Eq. (12)); return offset Σ ln Z^{(v)}."""
        vertex_z_factors = local_z_factors(tensor_network, messages)
        normalized_network = tensor_network.copy_empty_tensors()
        log_normalization_offset = 0.0
        for node in tensor_network.nodes:
            vertex_factor = vertex_z_factors[node]
            log_normalization_offset += numpy.log(abs(vertex_factor) + 1e-30)
            normalized_network.tensors[node] = (
                tensor_network.tensors[node] / vertex_factor
            )
        return normalized_network, log_normalization_offset, vertex_z_factors

    def free_energy_density(
        log_partition_function: float, inverse_temperature: float, site_count: int
    ) -> float:
        """f = -β⁻¹ ln Z / N (paper Sec. V)."""
        return -log_partition_function / (inverse_temperature * site_count)

    return compute_bp_partition, normalize_tensors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### II.3 Loop series expansion

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
    ### II.4 Divergence due to disconnected loops

    Truncating $\sum_{|\ell|\le m} Z_\ell$ still sums **disconnected** loops; on an $L\times L$
    lattice the count of disjoint plaquettes grows combinatorially in $L^2$, overwhelming
    exponential decay of $|Z_\ell|$ (Fig. 2 in the paper).
    """)
    return


@app.cell
def _(Any, combinations, networkx):
    def is_paper_generalized_loop(
        undirected_edge_list: list[tuple[Any, Any]],
    ) -> bool:
        """Def. II.1: induced subgraph has min degree ≥ 2."""
        if not undirected_edge_list:
            return False
        induced_subgraph = networkx.Graph()
        induced_subgraph.add_edges_from(undirected_edge_list)
        return all(
            induced_subgraph.degree(vertex) >= 2
            for vertex in induced_subgraph.nodes()
        )

    def canonical_undirected_edge_key(
        first_vertex: Any, second_vertex: Any
    ) -> tuple[Any, Any]:
        return (
            (first_vertex, second_vertex)
            if first_vertex < second_vertex
            else (second_vertex, first_vertex)
        )

    def canonical_loop_edges(
        edge_list: list[tuple[Any, Any]],
    ) -> frozenset[tuple[Any, Any]]:
        return frozenset(
            canonical_undirected_edge_key(tail, head)
            for tail, head in edge_list
        )

    def enumerate_connected_paper_loops(
        interaction_graph: networkx.Graph,
        max_edge_weight: int,
        max_loop_count: int = 6000,
    ) -> list[list[tuple[Any, Any]]]:
        canonical_edges = sorted(
            {
                canonical_undirected_edge_key(tail, head)
                for tail, head in interaction_graph.edges()
            }
        )
        edge_count = len(canonical_edges)
        seen_edge_sets: set[frozenset[tuple[Any, Any]]] = set()
        loop_edge_lists: list[list[tuple[Any, Any]]] = []
        for subset_size in range(1, min(max_edge_weight, edge_count) + 1):
            for edge_index_tuple in combinations(range(edge_count), subset_size):
                candidate_edges = [
                    canonical_edges[edge_index] for edge_index in edge_index_tuple
                ]
                if not is_paper_generalized_loop(candidate_edges):
                    continue
                connectivity_graph = networkx.Graph()
                connectivity_graph.add_edges_from(candidate_edges)
                if not networkx.is_connected(connectivity_graph):
                    continue
                dedup_key = frozenset(candidate_edges)
                if dedup_key in seen_edge_sets:
                    continue
                seen_edge_sets.add(dedup_key)
                loop_edge_lists.append(list(candidate_edges))
                if len(loop_edge_lists) >= max_loop_count:
                    return loop_edge_lists
        return loop_edge_lists

    def enumerate_paper_loops(
        interaction_graph: networkx.Graph,
        max_edge_weight: int,
        max_loop_count: int = 6000,
    ) -> list[list[tuple[Any, Any]]]:
        """All generalized loops (Def. II.1); Lemma II.2 sums over these, connected or not."""
        canonical_edges = sorted(
            {
                canonical_undirected_edge_key(tail, head)
                for tail, head in interaction_graph.edges()
            }
        )
        edge_count = len(canonical_edges)
        seen_edge_sets: set[frozenset[tuple[Any, Any]]] = set()
        loop_edge_lists: list[list[tuple[Any, Any]]] = []
        for subset_size in range(1, min(max_edge_weight, edge_count) + 1):
            for edge_index_tuple in combinations(range(edge_count), subset_size):
                candidate_edges = [
                    canonical_edges[edge_index] for edge_index in edge_index_tuple
                ]
                if not is_paper_generalized_loop(candidate_edges):
                    continue
                dedup_key = frozenset(candidate_edges)
                if dedup_key in seen_edge_sets:
                    continue
                seen_edge_sets.add(dedup_key)
                loop_edge_lists.append(list(candidate_edges))
                if len(loop_edge_lists) >= max_loop_count:
                    return loop_edge_lists
        return loop_edge_lists

    def enumerate_simple_cycles_as_edges(
        interaction_graph: networkx.Graph,
        maximum_cycle_length: int,
    ) -> list[list[tuple[Any, Any]]]:
        cycle_edge_lists: list[list[tuple[Any, Any]]] = []
        for vertex_cycle in networkx.simple_cycles(
            interaction_graph, length_bound=maximum_cycle_length
        ):
            if len(vertex_cycle) < 3:
                continue
            edges_along_cycle: list[tuple[Any, Any]] = []
            for position in range(len(vertex_cycle)):
                tail_vertex = vertex_cycle[position]
                head_vertex = vertex_cycle[(position + 1) % len(vertex_cycle)]
                edges_along_cycle.append(
                    canonical_undirected_edge_key(tail_vertex, head_vertex)
                )
            unique_sorted_edges = sorted(set(edges_along_cycle))
            if unique_sorted_edges not in cycle_edge_lists:
                cycle_edge_lists.append(unique_sorted_edges)
        return cycle_edge_lists

    return (
        enumerate_connected_paper_loops,
        enumerate_paper_loops,
        enumerate_simple_cycles_as_edges,
    )


@app.cell
def _(
    enumerate_connected_paper_loops,
    layout_graph,
    matplotlib_pyplot,
    mo,
    networkx,
):
    demo_graph = networkx.cycle_graph(4)
    demo_graph.add_edge(0, 2)
    example_loops = enumerate_connected_paper_loops(
        demo_graph, max_edge_weight=8, max_loop_count=50
    )
    subplot_count = min(len(example_loops), 6)
    _generalized_loop_figure, _generalized_loop_axes_array = (
        matplotlib_pyplot.subplots(1, subplot_count, figsize=(3 * subplot_count, 3))
    )
    if not hasattr(_generalized_loop_axes_array, "__len__"):
        _generalized_loop_axes_array = [_generalized_loop_axes_array]
    _generalized_loop_node_positions = layout_graph(demo_graph)
    for panel_index, (loop_edges, loop_diagram_axis) in enumerate(
        zip(example_loops[:6], _generalized_loop_axes_array)
    ):
        networkx.draw(
            demo_graph,
            _generalized_loop_node_positions,
            ax=loop_diagram_axis,
            with_labels=True,
            node_color="lightgray",
            node_size=300,
        )
        networkx.draw_networkx_edges(
            demo_graph,
            _generalized_loop_node_positions,
            edgelist=loop_edges,
            edge_color="red",
            width=3,
            ax=loop_diagram_axis,
        )
        loop_diagram_axis.set_title(f"{len(loop_edges)} edges", fontsize=9)
    matplotlib_pyplot.suptitle(
        "Example connected generalized loops (min degree ≥ 2)", fontsize=11
    )
    matplotlib_pyplot.tight_layout()
    mo.md(
        "**Def. II.1:** edge-induced subgraphs where every vertex has **degree ≥ 2**; "
        "a generalized loop need **not** be connected (paper). Here we draw **connected** examples."
    )
    _generalized_loop_figure
    return


@app.cell
def _(Any, numpy):
    def compute_edge_basis(
        messages: dict[tuple[Any, Any], numpy.ndarray],
        tail_vertex: Any,
        head_vertex: Any,
        bond_dimension: int,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        message_tail_to_head = messages[(tail_vertex, head_vertex)].copy()
        message_tail_to_head /= (
            numpy.linalg.norm(message_tail_to_head) + 1e-15
        )
        message_head_to_tail = messages[(head_vertex, tail_vertex)].copy()
        message_head_to_tail /= (
            numpy.linalg.norm(message_head_to_tail) + 1e-15
        )

        def gram_schmidt_orthonormalize(
            leading_unit_direction: numpy.ndarray, ambient_dimension: int
        ) -> numpy.ndarray:
            orthonormal_basis = numpy.zeros((ambient_dimension, ambient_dimension))
            orthonormal_basis[0] = leading_unit_direction / (
                numpy.linalg.norm(leading_unit_direction) + 1e-15
            )
            for basis_index in range(1, ambient_dimension):
                candidate_vector = numpy.zeros(ambient_dimension)
                candidate_vector[basis_index] = 1.0
                for previous_index in range(basis_index):
                    candidate_vector -= (
                        candidate_vector @ orthonormal_basis[previous_index]
                    ) * orthonormal_basis[previous_index]
                candidate_norm = numpy.linalg.norm(candidate_vector)
                if candidate_norm < 1e-12:
                    candidate_vector = numpy.random.randn(ambient_dimension)
                    for previous_index in range(basis_index):
                        candidate_vector -= (
                            candidate_vector @ orthonormal_basis[previous_index]
                        ) * orthonormal_basis[previous_index]
                    candidate_norm = numpy.linalg.norm(candidate_vector)
                orthonormal_basis[basis_index] = candidate_vector / (
                    candidate_norm + 1e-15
                )
            return orthonormal_basis

        return (
            gram_schmidt_orthonormalize(message_tail_to_head, bond_dimension),
            gram_schmidt_orthonormalize(message_head_to_tail, bond_dimension),
        )

    return (compute_edge_basis,)


@app.cell
def _(Any, TensorNetwork, compute_edge_basis, numpy):
    def compute_loop_tensor(
        tensor_network: TensorNetwork,
        messages: dict[tuple[Any, Any], numpy.ndarray],
        loop_edge_list: list[tuple[Any, Any]],
    ) -> float:
        """Loop correction Z_ℓ for χ=2 (Eq. (9): contract every T_v, ⊥ on loop legs, μ elsewhere)."""
        bond_dimension = tensor_network.chi
        if bond_dimension != 2:
            raise NotImplementedError("This notebook implements χ=2 (Ising).")
        orthonormal_bases_by_directed_edge: dict[
            tuple[Any, Any], numpy.ndarray
        ] = {}
        for tail_vertex, head_vertex in loop_edge_list:
            basis_uv, basis_vu = compute_edge_basis(
                messages, tail_vertex, head_vertex, bond_dimension
            )
            orthonormal_bases_by_directed_edge[(tail_vertex, head_vertex)] = (
                basis_uv
            )
            orthonormal_bases_by_directed_edge[(head_vertex, tail_vertex)] = (
                basis_vu
            )
        directed_loop_edges: set[tuple[Any, Any]] = set()
        for tail_vertex, head_vertex in loop_edge_list:
            directed_loop_edges.add((tail_vertex, head_vertex))
            directed_loop_edges.add((head_vertex, tail_vertex))
        loop_correction_product = 1.0
        for node in tensor_network.nodes:
            neighbor_nodes = tensor_network.neighbors(node)
            local_tensor = tensor_network.tensors[node].copy()
            contraction_vectors: list[numpy.ndarray] = []
            for neighbor_vertex in neighbor_nodes:
                if (
                    (neighbor_vertex, node) in directed_loop_edges
                    or (node, neighbor_vertex) in directed_loop_edges
                ):
                    contraction_vectors.append(
                        orthonormal_bases_by_directed_edge[
                            (neighbor_vertex, node)
                        ][1]
                    )
                else:
                    contraction_vectors.append(
                        messages[(neighbor_vertex, node)]
                    )
            contraction_result = local_tensor
            for direction_vector in contraction_vectors:
                contraction_result = numpy.tensordot(
                    contraction_result, direction_vector, axes=([0], [0])
                )
            loop_correction_product *= float(contraction_result)
        return loop_correction_product

    return (compute_loop_tensor,)


@app.cell
def _(
    belief_propagation,
    compute_bp_partition,
    compute_loop_tensor,
    enumerate_paper_loops,
    make_ising_tn,
    mo,
    numpy,
):
    _tn = make_ising_tn(3, 3, inverse_temperature=0.3, periodic_boundary=False)
    _msgs, _, _ = belief_propagation(
        _tn,
        max_iterations=4000,
        convergence_tolerance=1e-12,
        damping_factor=0.3,
        message_noise_standard_deviation=0.0,
    )
    _logZ_bp = compute_bp_partition(_tn, _msgs)
    _Z0 = numpy.exp(_logZ_bp)
    _Z_exact = float(_tn.contract_exact())
    _loops = enumerate_paper_loops(
        _tn.graph, max_edge_weight=12, max_loop_count=4000
    )
    _s = 0.0
    for _lp in _loops:
        _s += compute_loop_tensor(_tn, _msgs, _lp)
    _Z_rec = _Z0 + _s
    mo.md(f"""
    #### Lemma II.2 (numerical check, 3×3 open, β=0.3)

    | | |
    |---|---|
    | $Z_{{\\rm exact}}$ | {_Z_exact:.8f} |
    | $Z_0 \\approx e^{{\\mathcal{{F}}_{{\\rm BP}}}}$ | {_Z0:.8f} |
    | $\\sum_{{\\ell\\in\\mathcal{{L}}_G}} Z_\\ell$ ($\\|\\ell\\|\\le 12$, {len(_loops)} terms) | {_s:.8f} |
    | $Z_0 + \\sum Z_\\ell$ | {_Z_rec:.8f} |
    | rel. err. | {abs(_Z_rec-_Z_exact)/abs(_Z_exact):.3e} |

    Eq. (9) contracts **every** site tensor $T_v$; off-loop sites use the BP vacuum on all legs. On this graph, every generalized loop with $\\|\\ell\\|\\le 12$ is connected, so the count matches a connected-only enumeration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## III Cluster expansion

    ### III.1 Physical intuition

    The **partition function** $\mathcal{Z}$ is a global product of local tensor data; its logarithm
    $\mathcal{F}=\ln\mathcal{Z}$ is **extensive** on large homogeneous systems. Expanding
    $\mathcal{Z}$ directly (Lemma II.2) keeps **multiplicative** disconnected excitations; expanding
    $\mathcal{F}(\tilde{\mathcal{T}})$ after BP normalization yields an **additive** series over
    **connected** loop clusters (Lemma III.1), better matched to thermodynamic scaling.
    """)
    return


@app.cell(hide_code=True)
def _(layout_graph, matplotlib_pyplot, mo, networkx):
    def _plaquette_cycle_edges(
        top_left_row: int, top_left_col: int
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Unit face (plaquette) as a 4-cycle on grid_2d nodes; CCW from top-left."""
        r, c = top_left_row, top_left_col
        corners = ((r, c), (r + 1, c), (r + 1, c + 1), (r, c + 1))
        return [
            (corners[k], corners[(k + 1) % 4]) for k in range(4)
        ]

    def _canonical_edge(
        tail: tuple[int, int], head: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return tuple(sorted((tail, head)))

    def _edge_list_to_canonical_set(
        edge_list: list[tuple[tuple[int, int], tuple[int, int]]],
    ) -> set[tuple[tuple[int, int], tuple[int, int]]]:
        return {_canonical_edge(u, v) for u, v in edge_list}

    _fig2, (_ax_a, _ax_b) = matplotlib_pyplot.subplots(1, 2, figsize=(9.5, 4.2))

    # (a) Two disjoint plaquettes: compatible; their union is one disconnected loop.
    _grid_a = networkx.grid_2d_graph(4, 2)
    _pos_a = layout_graph(_grid_a)
    _edges_a1 = _plaquette_cycle_edges(0, 0)
    _edges_a2 = _plaquette_cycle_edges(2, 0)
    networkx.draw_networkx_nodes(
        _grid_a, _pos_a, ax=_ax_a, node_color="lightgray", node_size=220
    )
    networkx.draw_networkx_edges(
        _grid_a, _pos_a, ax=_ax_a, edge_color="#dddddd", width=1.2
    )
    networkx.draw_networkx_edges(
        _grid_a,
        _pos_a,
        ax=_ax_a,
        edgelist=_edges_a1,
        edge_color="crimson",
        width=3.0,
    )
    networkx.draw_networkx_edges(
        _grid_a,
        _pos_a,
        ax=_ax_a,
        edgelist=_edges_a2,
        edge_color="steelblue",
        width=3.0,
    )
    _ax_a.set_title(
        "(a) Compatible / disconnected\n"
        "two plaquettes: no shared vertex or edge (Def. III.1)",
        fontsize=10,
    )
    _ax_a.axis("off")

    # (b) Adjacent plaquettes: incompatible (share two vertices and one edge).
    _grid_b = networkx.grid_2d_graph(2, 4)
    _pos_b = layout_graph(_grid_b)
    _edges_b1 = _plaquette_cycle_edges(0, 0)
    _edges_b2 = _plaquette_cycle_edges(0, 1)
    _set_b1 = _edge_list_to_canonical_set(_edges_b1)
    _set_b2 = _edge_list_to_canonical_set(_edges_b2)
    _shared_b = _set_b1 & _set_b2
    _only_b1 = _set_b1 - _set_b2
    _only_b2 = _set_b2 - _set_b1
    networkx.draw_networkx_nodes(
        _grid_b, _pos_b, ax=_ax_b, node_color="lightgray", node_size=220
    )
    networkx.draw_networkx_edges(
        _grid_b, _pos_b, ax=_ax_b, edge_color="#dddddd", width=1.2
    )
    if _only_b1:
        networkx.draw_networkx_edges(
            _grid_b,
            _pos_b,
            ax=_ax_b,
            edgelist=list(_only_b1),
            edge_color="crimson",
            width=3.0,
        )
    if _only_b2:
        networkx.draw_networkx_edges(
            _grid_b,
            _pos_b,
            ax=_ax_b,
            edgelist=list(_only_b2),
            edge_color="steelblue",
            width=3.0,
        )
    if _shared_b:
        networkx.draw_networkx_edges(
            _grid_b,
            _pos_b,
            ax=_ax_b,
            edgelist=list(_shared_b),
            edge_color="mediumpurple",
            width=3.5,
        )
    _ax_b.set_title(
        "(b) Incompatible\n" r"two plaquettes overlap on an edge (Def. III.1)",
        fontsize=10,
    )
    _ax_b.axis("off")

    matplotlib_pyplot.suptitle(
        "Paper Fig. 2: disconnectedness vs. incompatibility",
        fontsize=12,
        y=1.02,
    )
    matplotlib_pyplot.tight_layout()
    mo.md(
        "**Paper Fig. 2** (Sec. III, before Def. III.1): **(a)** two **compatible** plaquette loops "
        "form a single **disconnected** generalized loop—this is the combinatorics that blows up the "
        "naïve loop sum (§II.4). **(b)** two **incompatible** loops share a bond; such pairs are "
        "adjacent in the cluster **interaction graph** $G_{\\mathbf{W}}$ (Def. III.4). "
        "*(The paper’s **Fig. 4** is the Ising free-energy benchmark; this schematic is **Fig. 2**.)*"
    )
    _fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### III.2 Formal definition (Defs. III.1–III.5)

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
def _(Any, Iterable, networkx):
    def phi_cluster_from_graph(
        loop_interaction_graph: networkx.Graph, multiset_factorial_product: float
    ) -> float:
        """φ = (1/W!) * sum_{spanning connected} (-1)^|E|."""
        import math

        instance_count = loop_interaction_graph.number_of_nodes()
        if instance_count == 0:
            return 0.0
        if instance_count == 1:
            return 1.0 / multiset_factorial_product
        graph_edge_count = loop_interaction_graph.number_of_edges()
        complete_graph_edge_count = instance_count * (instance_count - 1) // 2
        if graph_edge_count == complete_graph_edge_count:
            # Identical loop instances ⇒ interaction graph is K_n. Brute force is 2^{Θ(n²)};
            # the signed sum over connected spanning subgraphs of K_n is (-1)^{n-1} (n-1)!.
            signed_connected_sum = (
                (-1) ** (instance_count - 1)
                * math.factorial(instance_count - 1)
            )
            return float(signed_connected_sum) / multiset_factorial_product

        # Relabel nodes to 0..n-1 (interaction graphs are always built that way; this is defensive).
        nodes_sorted = sorted(loop_interaction_graph.nodes())
        index_by_node = {node: index for index, node in enumerate(nodes_sorted)}
        if len(index_by_node) != instance_count:
            raise ValueError("Unexpected duplicate or missing node in interaction graph")
        adjacency_bitmasks = [0] * instance_count
        for tail_vertex, head_vertex in loop_interaction_graph.edges():
            tail_index = index_by_node[tail_vertex]
            head_index = index_by_node[head_vertex]
            adjacency_bitmasks[tail_index] |= 1 << head_index
            adjacency_bitmasks[head_index] |= 1 << tail_index

        def induced_subgraph_has_no_edges(vertex_mask: int) -> bool:
            """True iff vertex set is an independent set (so ∑_{F⊆E(S)}(-1)^|F| = 1)."""
            remaining = vertex_mask
            while remaining:
                lowest_bit = remaining & -remaining
                vertex_index = lowest_bit.bit_length() - 1
                if adjacency_bitmasks[vertex_index] & vertex_mask:
                    return False
                remaining -= lowest_bit
            return True

        def signed_connected_spanning_sum_subset_dp() -> int:
            """O(n 2^n): sum of (-1)^|F| over edge sets F that span all vertices and are connected."""
            vertex_count = instance_count
            full_mask = (1 << vertex_count) - 1
            # w[mask] = ∑_{F ⊆ E(S)} (-1)^|F| = (1-1)^{|E(S)|} ∈ {0,1}; w[∅]=1.
            subset_weights: list[int] = [0] * (1 << vertex_count)
            subset_weights[0] = 1
            for vertex_mask in range(1, 1 << vertex_count):
                subset_weights[vertex_mask] = (
                    1 if induced_subgraph_has_no_edges(vertex_mask) else 0
                )
            connected_signed: list[int] = [0] * (1 << vertex_count)
            for vertex_mask in range(1, 1 << vertex_count):
                if vertex_mask & (vertex_mask - 1) == 0:
                    connected_signed[vertex_mask] = 1
                    continue
                least_bit = vertex_mask & -vertex_mask
                rest_vertices = vertex_mask ^ least_bit
                total = subset_weights[vertex_mask]
                submask = rest_vertices
                while True:
                    subset_with_root = least_bit | submask
                    if subset_with_root != vertex_mask:
                        complement = vertex_mask ^ subset_with_root
                        total -= (
                            connected_signed[subset_with_root]
                            * subset_weights[complement]
                        )
                    if submask == 0:
                        break
                    submask = (submask - 1) & rest_vertices
                connected_signed[vertex_mask] = total
            return int(connected_signed[full_mask])

        # Edge brute force costs Θ(2^{|E|}) and can hang on dense graphs (|E|=Θ(n²)).
        # Subset DP costs Θ(n 2^n) in the number of interaction vertices (multiset size).
        max_vertices_for_subset_dp = 22
        if instance_count > max_vertices_for_subset_dp:
            raise NotImplementedError(
                f"Ursell φ via subset DP limited to n≤{max_vertices_for_subset_dp} "
                f"vertices (got n={instance_count})."
            )
        signed_total = signed_connected_spanning_sum_subset_dp()
        return float(signed_total) / multiset_factorial_product

    def build_interaction_graph(
        loop_vertex_sets: list[frozenset[Any]],
        same_cluster_ids: list[int],
    ) -> networkx.Graph:
        """
        loop_vertex_sets: list of frozenset of graph vertices for each loop instance
        same_cluster_ids: list of int loop-type id (identical loops share id)
        """
        loop_instance_count = len(loop_vertex_sets)
        interaction_graph = networkx.Graph()
        interaction_graph.add_nodes_from(range(loop_instance_count))
        for first_index in range(loop_instance_count):
            for second_index in range(first_index + 1, loop_instance_count):
                if same_cluster_ids[first_index] == same_cluster_ids[second_index]:
                    interaction_graph.add_edge(first_index, second_index)
                elif (
                    loop_vertex_sets[first_index] & loop_vertex_sets[second_index]
                ):
                    interaction_graph.add_edge(first_index, second_index)
        return interaction_graph

    def cluster_factorial(multiplicities: Iterable[int]) -> float:
        import math

        factorial_product = 1.0
        for multiplicity in multiplicities:
            factorial_product *= math.factorial(int(multiplicity))
        return factorial_product

    return build_interaction_graph, phi_cluster_from_graph


@app.cell
def _(
    Any,
    Counter,
    build_interaction_graph,
    combinations_with_replacement,
    math,
    networkx,
    phi_cluster_from_graph,
):
    def cluster_expansion_sum(
        loop_edge_lists: list[list[tuple[Any, Any]]],
        loop_correction_values: list[float],
        max_total_edge_weight: int,
        max_multiset_size: int = 4,
    ) -> float:
        """Σ_{connected W} φ(W) Z_W (Lemma III.1); Z_ℓ evaluated on T̃."""
        vertex_sets_per_loop: list[frozenset[Any]] = []
        for edge_list in loop_edge_lists:
            vertices_on_loop: set[Any] = set()
            for tail_vertex, head_vertex in edge_list:
                vertices_on_loop.add(tail_vertex)
                vertices_on_loop.add(head_vertex)
            vertex_sets_per_loop.append(frozenset(vertices_on_loop))
        loop_type_count = len(loop_edge_lists)
        cluster_free_energy_sum = 0.0
        for multiset_size in range(1, max_multiset_size + 1):
            for loop_index_tuple in combinations_with_replacement(
                range(loop_type_count), multiset_size
            ):
                if (
                    sum(len(loop_edge_lists[loop_index]) for loop_index in loop_index_tuple)
                    > max_total_edge_weight
                ):
                    continue
                multiplicity_counter = Counter(loop_index_tuple)
                interaction_graph = build_interaction_graph(
                    [vertex_sets_per_loop[loop_index] for loop_index in loop_index_tuple],
                    list(loop_index_tuple),
                )
                if not networkx.is_connected(interaction_graph):
                    continue
                symmetry_factorial = math.prod(
                    math.factorial(int(multiplicity))
                    for multiplicity in multiplicity_counter.values()
                )
                ursell_weight = phi_cluster_from_graph(
                    interaction_graph, symmetry_factorial
                )
                loop_product = 1.0
                for loop_index in loop_index_tuple:
                    loop_product *= loop_correction_values[loop_index]
                cluster_free_energy_sum += ursell_weight * loop_product
        return cluster_free_energy_sum

    return (cluster_expansion_sum,)


@app.cell
def _(
    Any,
    TensorNetwork,
    cluster_expansion_sum,
    compute_loop_tensor,
    enumerate_connected_paper_loops,
    enumerate_simple_cycles_as_edges,
    networkx,
    normalize_tensors,
    numpy,
):
    def collect_loops(
        interaction_graph: networkx.Graph,
        max_total_edge_weight: int,
        use_simple_cycle_basis_only: bool = False,
        max_loop_count: int = 8000,
    ) -> list[list[tuple[Any, Any]]]:
        if (
            use_simple_cycle_basis_only
            or interaction_graph.number_of_edges() > 20
        ):
            return enumerate_simple_cycles_as_edges(
                interaction_graph, maximum_cycle_length=max_total_edge_weight
            )
        return enumerate_connected_paper_loops(
            interaction_graph,
            max_edge_weight=max_total_edge_weight,
            max_loop_count=max_loop_count,
        )

    def log_partition_cluster(
        tensor_network: TensorNetwork,
        messages: dict[tuple[Any, Any], numpy.ndarray],
        max_total_edge_weight: int,
        max_multiset_size: int = 4,
        use_simple_cycle_basis_only: bool = False,
    ) -> tuple[float, list[list[tuple[Any, Any]]], list[float]]:
        """log Z ≈ ℱ(T̃) + Σ_v ln Z^(v) with ℱ from cluster sum (Lemma III.1, Eq. (13))."""
        loop_edge_lists = collect_loops(
            tensor_network.graph,
            max_total_edge_weight,
            use_simple_cycle_basis_only=use_simple_cycle_basis_only,
        )
        normalized_network, log_normalization_offset, _ = normalize_tensors(
            tensor_network, messages
        )
        loop_correction_values = [
            compute_loop_tensor(normalized_network, messages, edge_list)
            for edge_list in loop_edge_lists
        ]
        cluster_free_energy = cluster_expansion_sum(
            loop_edge_lists,
            loop_correction_values,
            max_total_edge_weight,
            max_multiset_size,
        )
        return (
            cluster_free_energy + log_normalization_offset,
            loop_edge_lists,
            loop_correction_values,
        )

    return (log_partition_cluster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### III.3 Toy example (ring; normalized network)

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
    matplotlib_pyplot,
    mo,
    normalize_tensors,
    numpy,
):
    ring_site_count = 5
    _ring_section_inverse_temperature = 0.3
    ring_tensor_network = make_ising_ring_tn(
        ring_site_count, _ring_section_inverse_temperature
    )
    log_partition_unnormalized = float(
        numpy.log(float(ring_tensor_network.contract_exact()))
    )
    _ring_section_belief_propagation_messages, belief_propagation_converged, _ = (
        belief_propagation(
            ring_tensor_network,
            max_iterations=4000,
            convergence_tolerance=1e-12,
            damping_factor=0.5,
            message_noise_standard_deviation=0.0,
        )
    )
    _ring_section_normalized_network, log_vertex_offset, _ = normalize_tensors(
        ring_tensor_network, _ring_section_belief_propagation_messages
    )
    normalized_partition = float(
        _ring_section_normalized_network.contract_exact()
    )
    log_normalized_partition = float(numpy.log(normalized_partition))

    canonical_ring_edges: list[tuple[int, int]] = []
    for site_index in range(ring_site_count):
        left_vertex = site_index
        right_vertex = (site_index + 1) % ring_site_count
        canonical_ring_edges.append(
            (left_vertex, right_vertex)
            if left_vertex < right_vertex
            else (right_vertex, left_vertex)
        )
    single_loop_correction = compute_loop_tensor(
        _ring_section_normalized_network,
        _ring_section_belief_propagation_messages,
        canonical_ring_edges,
    )
    one_plus_loop_correction = 1.0 + single_loop_correction
    log_one_plus_loop_correction = float(
        numpy.log(one_plus_loop_correction)
    )

    single_loop_edge_lists = [canonical_ring_edges]
    single_loop_correction_values = [single_loop_correction]
    max_truncation_order = 14
    truncation_orders = list(range(1, max_truncation_order + 1))
    cluster_truncation_errors: list[float] = []
    taylor_truncation_errors: list[float] = []
    for max_multiset_size in truncation_orders:
        cluster_free_energy_at_order = cluster_expansion_sum(
            single_loop_edge_lists,
            single_loop_correction_values,
            max_total_edge_weight=9999,
            max_multiset_size=max_multiset_size,
        )
        cluster_truncation_errors.append(
            abs(cluster_free_energy_at_order - log_normalized_partition)
        )
        taylor_partial_sum = sum(
            ((-1) ** (series_index + 1))
            * (single_loop_correction**series_index)
            / series_index
            for series_index in range(1, max_multiset_size + 1)
        )
        taylor_truncation_errors.append(
            abs(taylor_partial_sum - log_normalized_partition)
        )

    _ring_k_truncation_figure, _ring_k_truncation_axes_pair = (
        matplotlib_pyplot.subplots(1, 2, figsize=(10, 3.8))
    )
    _ring_k_truncation_axes_pair[0].semilogy(
        truncation_orders,
        cluster_truncation_errors,
        "o-",
        ms=4,
        label=r"$|\tilde{\mathcal{F}}_K - \ln\tilde{Z}|$",
    )
    _ring_k_truncation_axes_pair[0].axhline(
        abs(log_one_plus_loop_correction - log_normalized_partition),
        color="gray",
        ls=":",
        lw=1.2,
        label=r"$|\ln(1+Z_\ell)-\ln\tilde{Z}|$",
    )
    _ring_k_truncation_axes_pair[0].set_xlabel(r"max multiset size $K$")
    _ring_k_truncation_axes_pair[0].set_ylabel("absolute error")
    _ring_k_truncation_axes_pair[0].set_title("Cluster expansion vs truncation")
    _ring_k_truncation_axes_pair[0].legend(fontsize=8)
    _ring_k_truncation_axes_pair[0].grid(True, alpha=0.3)

    _ring_k_truncation_axes_pair[1].semilogy(
        truncation_orders,
        taylor_truncation_errors,
        "s-",
        ms=4,
        color="darkgreen",
        label=r"$|T_K - \ln\tilde{Z}|$",
    )
    _ring_k_truncation_axes_pair[1].axhline(
        abs(log_one_plus_loop_correction - log_normalized_partition),
        color="gray",
        ls=":",
        lw=1.2,
        label=r"$|\ln(1+Z_\ell)-\ln\tilde{Z}|$",
    )
    _ring_k_truncation_axes_pair[1].set_xlabel(r"Taylor order $K$")
    _ring_k_truncation_axes_pair[1].set_ylabel("absolute error")
    _ring_k_truncation_axes_pair[1].set_title(
        r"Eq. (23): $T_K=\sum_{k=1}^K (-1)^{k+1} Z_\ell^k/k$"
    )
    _ring_k_truncation_axes_pair[1].legend(fontsize=8)
    _ring_k_truncation_axes_pair[1].grid(True, alpha=0.3)
    matplotlib_pyplot.suptitle(
        rf"III.3 Toy example — $C_{ring_site_count}$ ring at "
        rf"$\beta={_ring_section_inverse_temperature}$: convergence in $K$",
        fontsize=11,
        y=1.02,
    )
    matplotlib_pyplot.tight_layout()

    _ring_summary_table = f"""
    ### Toy ring numerics ($C_{ring_site_count}$, $\\beta={_ring_section_inverse_temperature}$)

    | | |
    |---|---|
    | $\\ln Z$ (unnormalized TN) | {log_partition_unnormalized:.12f} |
    | $\\ln\\tilde{{Z}}$ (normalized TN, exact) | {log_normalized_partition:.12f} |
    | $\\sum_v \\ln Z^{{(v)}}$ (offset) | {log_vertex_offset:.12f} |
    | BP converged | `{belief_propagation_converged}` |
    | $Z_\\ell$ (single ring loop, $\\tilde{{T}}$) | {single_loop_correction:.6e} |
    | $1+Z_\\ell$ | {one_plus_loop_correction:.10f} |
    | $\\ln(1+Z_\\ell)$ | {log_one_plus_loop_correction:.12f} |
    | $\\|\\tilde{{Z}}-(1+Z_\\ell)\\|\\,/\\,\\tilde{{Z}}$ | {abs(normalized_partition - one_plus_loop_correction) / max(abs(normalized_partition), 1e-30):.2e} |
    | $\\|\\tilde{{\\mathcal{{F}}}}_{{{max_truncation_order}}} - \\ln\\tilde{{Z}}\\|$ | {cluster_truncation_errors[-1]:.2e} |
    | $\\|T_{{{max_truncation_order}}} - \\ln\\tilde{{Z}}\\|$ | {taylor_truncation_errors[-1]:.2e} |
    """
    _ring_summary_tail = """
    Both curves decay with $K$ when $|Z_\\ell|\\ll 1$: the cluster sum and the Taylor partial sums approach the same limit near $\\ln\\tilde{Z}$. The horizontal gray line is the gap between that limit and the one-line linked-cluster value $\\ln(1+Z_\\ell)$ when $\\tilde{Z}\\neq 1+Z_\\ell$ at finite $\\chi$ and BP messages.
    """
    summary_markdown = mo.md(_ring_summary_tail)

    mo.vstack([summary_markdown, _ring_k_truncation_figure])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## IV Algorithm (Fig. 3)

    ### IV.1 Cluster enumeration

    1. **Enumerate** connected loops and (once per graph) connected clusters up to weight $m$
       (Appendix II; complexity $O(n\,e^{O(m)})$ per Lemma III.2).

    ### IV.2 Message passing and normalization

    2. Run **BP** with damping, random init, optional noise; enforce small edge residual (Eq. (25)).
    3. **Normalize** $\tilde T_v=T_v/Z^{(v)}$ and store $\sum_v \ln Z^{(v)}$.

    ### IV.3 Computing cluster contributions

    4. For each cluster $\mathbf{W}$, compute $Z_{\mathbf{W}}$ and $\phi(\mathbf{W})$ from $G_{\mathbf{W}}$
       (Eq. (16)); **sum** to $\tilde F_m$ (Eq. (17)); output $\tilde F_m+\sum_v\ln Z^{(v)}$.
    5. Contributions parallelize over clusters; for PEPS-scale problems many $\phi$ are $1,-1,-\tfrac12$
       at modest $m$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## V Benchmark: 2D Ising model

    Section **V** compares BP, the cluster expansion, and the naïve loop series to **exact** finite
    lattices and the **Onsager** thermodynamic limit for the square-lattice Ising model (free energy
    density and loop weights as in Figs. 4–5).
    """)
    return


@app.cell
def _(numpy, scipy):
    def onsager_log_partition_per_site(
        inverse_temperature: float, coupling_strength: float = 1.0
    ) -> float:
        """Thermodynamic limit ln 𝒵/N for the 2D square-lattice Ising (Onsager)."""

        def onsager_integrand(
            angle_horizontal: float, angle_vertical: float
        ) -> numpy.floating:
            return numpy.log(
                numpy.cosh(2 * inverse_temperature * coupling_strength) ** 2
                - numpy.sinh(2 * inverse_temperature * coupling_strength)
                * (
                    numpy.cos(angle_horizontal)
                    + numpy.cos(angle_vertical)
                )
            )

        integral_value, _ = scipy.integrate.dblquad(
            onsager_integrand, 0, numpy.pi, 0, numpy.pi
        )
        return float(
            numpy.log(2) + integral_value / (2 * numpy.pi**2)
        )

    def onsager_free_energy(
        inverse_temperature: float, coupling_strength: float = 1.0
    ) -> float:
        """Sec. V: f = -β⁻¹ ln 𝒵/N (same as -onsager_log_partition_per_site / β)."""
        return float(
            -onsager_log_partition_per_site(
                inverse_temperature, coupling_strength
            )
            / inverse_temperature
        )

    return (onsager_free_energy,)


@app.cell
def _(
    belief_propagation,
    compute_bp_partition,
    make_ising_tn,
    matplotlib_pyplot,
    mo,
    numpy,
    onsager_free_energy,
):
    _bp_onsager_critical_beta = 0.5 * numpy.log(1 + numpy.sqrt(2))
    _bp_bethe_critical_beta = numpy.log(2.0) / 2.0
    _bp_onsager_inverse_temperature_grid = numpy.linspace(0.1, 0.6, 40)
    _bp_onsager_lattice_linear_size = 5
    _bp_onsager_site_count = _bp_onsager_lattice_linear_size**2
    belief_propagation_minus_log_partition_per_site: list[float] = []
    # βf = -ln Z / N (decreasing in β); Helmholtz f = -β⁻¹ ln Z / N is the increasing curve.
    onsager_beta_f_curve = [
        _bp_onsager_beta_sample
        * onsager_free_energy(_bp_onsager_beta_sample)
        for _bp_onsager_beta_sample in _bp_onsager_inverse_temperature_grid
    ]
    for _bp_onsager_beta_sample in _bp_onsager_inverse_temperature_grid:
        _bp_onsager_tensor_network = make_ising_tn(
            _bp_onsager_lattice_linear_size,
            _bp_onsager_lattice_linear_size,
            _bp_onsager_beta_sample,
            periodic_boundary=False,
        )
        _bp_onsager_messages, _, _ = belief_propagation(
            _bp_onsager_tensor_network,
            max_iterations=400,
            damping_factor=0.25,
        )
        _bp_onsager_log_partition = compute_bp_partition(
            _bp_onsager_tensor_network, _bp_onsager_messages
        )
        belief_propagation_minus_log_partition_per_site.append(
            -_bp_onsager_log_partition / _bp_onsager_site_count
        )
    _bp_onsager_figure, _bp_onsager_axis = matplotlib_pyplot.subplots(
        figsize=(7, 4)
    )
    _bp_onsager_axis.plot(
        _bp_onsager_inverse_temperature_grid,
        onsager_beta_f_curve,
        "k-",
        lw=2,
        label="Onsager (L→∞)",
    )
    _bp_onsager_axis.plot(
        _bp_onsager_inverse_temperature_grid,
        belief_propagation_minus_log_partition_per_site,
        "r--",
        lw=1.5,
        label=f"BP vacuum (L={_bp_onsager_lattice_linear_size})",
    )
    _bp_onsager_axis.axvline(
        _bp_onsager_critical_beta,
        color="gray",
        ls=":",
        label=f"βc (Onsager) ≈ {_bp_onsager_critical_beta:.3f}",
    )
    _bp_onsager_axis.axvline(
        _bp_bethe_critical_beta,
        color="blue",
        ls=":",
        label=f"β_BP (Bethe z=4) ≈ {_bp_bethe_critical_beta:.3f}",
    )
    _bp_onsager_axis.set_xlabel("β")
    _bp_onsager_axis.set_ylabel(r"$-\ln\mathcal{Z}/N = \beta f$ (Sec. V)")
    _bp_onsager_axis.set_title(
        r"Fig. 4(a): $\beta f$ vs $\beta$ (BP vs Onsager)"
    )
    _bp_onsager_axis.legend(fontsize=8)
    _bp_onsager_axis.grid(True, alpha=0.3)
    matplotlib_pyplot.tight_layout()
    _bp_onsager_header = mo.md(
        "### V.1 BP vacuum\n\n"
        "Paper Fig. 4(a) compares BP to Onsager using **$\\beta f=-\\ln\\mathcal{Z}/N$** "
        "(the usual **decreasing** curve in $\\beta$). "
        "The Helmholtz density is $f=-\\beta^{-1}\\ln\\mathcal{Z}/N$."
    )
    mo.vstack([_bp_onsager_header, _bp_onsager_figure])
    return


@app.cell
def _(
    belief_propagation,
    compute_bp_partition,
    ising_partition_brute_force,
    log_partition_cluster,
    make_ising_tn,
    matplotlib_pyplot,
    mo,
    numpy,
):
    _fig4bc_L = 4
    _fig4bc_N = _fig4bc_L**2
    _fig4b_beta_grid = numpy.linspace(0.25, 0.45, 25)
    _fig4c_beta_grid = numpy.linspace(0.1, 0.85, 28)
    _fig4bc_weights = [4, 6, 8, 10]
    _fig4bc_beta_c_onsager = 0.5 * numpy.log(1 + numpy.sqrt(2))
    _fig4bc_beta_c_bethe = numpy.log(2.0) / 2.0
    _fig4bc_use_simple_cycles_only = True

    def _fig4bc_beta_f_total_log_z(total_log_z: float, site_count: int) -> float:
        """βf = -ln Z / N (same y-axis as Fig. 4(a))."""
        return -float(total_log_z) / site_count

    def _fig4bc_multiset_cap(edge_weight_limit: int) -> int:
        return min(8, max(4, edge_weight_limit // 2 + 2))

    _fig4bc_figure, _fig4bc_axes = matplotlib_pyplot.subplots(1, 2, figsize=(11, 4))

    _fig4b_exact_f: list[float] = []
    _fig4b_bp_f: list[float] = []
    for _fig4b_beta in _fig4b_beta_grid:
        _fig4b_tn = make_ising_tn(
            _fig4bc_L,
            _fig4bc_L,
            _fig4b_beta,
            periodic_boundary=False,
        )
        _fig4b_Z_exact = ising_partition_brute_force(
            _fig4b_beta,
            _fig4bc_L,
            _fig4bc_L,
            periodic_boundary=False,
        )
        _fig4b_exact_f.append(
            _fig4bc_beta_f_total_log_z(numpy.log(_fig4b_Z_exact), _fig4bc_N)
        )
        _fig4b_msgs, _, _ = belief_propagation(
            _fig4b_tn, max_iterations=450, damping_factor=0.25
        )
        _fig4b_bp_f.append(
            _fig4bc_beta_f_total_log_z(
                compute_bp_partition(_fig4b_tn, _fig4b_msgs), _fig4bc_N
            )
        )
    _fig4bc_axes[0].plot(
        _fig4b_beta_grid, _fig4b_exact_f, "k--", lw=2, label="exact"
    )
    _fig4bc_axes[0].plot(
        _fig4b_beta_grid, _fig4b_bp_f, "g:", lw=1.5, label="BP vacuum"
    )
    for _fig4b_w in _fig4bc_weights:
        _fig4b_cluster_f: list[float] = []
        for _fig4b_beta in _fig4b_beta_grid:
            _fig4b_tn = make_ising_tn(
                _fig4bc_L,
                _fig4bc_L,
                _fig4b_beta,
                periodic_boundary=False,
            )
            _fig4b_msgs, _, _ = belief_propagation(
                _fig4b_tn, max_iterations=450, damping_factor=0.25
            )
            _fig4b_logz, _, _ = log_partition_cluster(
                _fig4b_tn,
                _fig4b_msgs,
                max_total_edge_weight=_fig4b_w,
                max_multiset_size=_fig4bc_multiset_cap(_fig4b_w),
                use_simple_cycle_basis_only=_fig4bc_use_simple_cycles_only,
            )
            _fig4b_cluster_f.append(
                _fig4bc_beta_f_total_log_z(_fig4b_logz, _fig4bc_N)
            )
        _fig4bc_axes[0].plot(
            _fig4b_beta_grid,
            _fig4b_cluster_f,
            "-",
            label=f"cluster w≤{_fig4b_w}",
        )
    _fig4bc_axes[0].axvline(
        _fig4bc_beta_c_onsager, color="gray", ls=":", alpha=0.7
    )
    _fig4bc_axes[0].axvline(
        _fig4bc_beta_c_bethe, color="blue", ls=":", alpha=0.7
    )
    _fig4bc_axes[0].set_xlabel("β")
    _fig4bc_axes[0].set_ylabel(r"$-\ln\mathcal{Z}/N = \beta f$")
    _fig4bc_axes[0].set_title(
        r"Fig. 4(b): $\beta f(\beta)$ with cluster corrections ($\beta\in[0.25,0.45]$)"
    )
    _fig4bc_axes[0].legend(fontsize=7, ncol=2)
    _fig4bc_axes[0].grid(True, alpha=0.3)

    _fig4c_bp_abs_delta: list[float] = []
    for _fig4c_beta in _fig4c_beta_grid:
        _fig4c_tn = make_ising_tn(
            _fig4bc_L,
            _fig4bc_L,
            _fig4c_beta,
            periodic_boundary=False,
        )
        _fig4c_Z_exact = ising_partition_brute_force(
            _fig4c_beta,
            _fig4bc_L,
            _fig4bc_L,
            periodic_boundary=False,
        )
        _fig4c_bf_exact = _fig4bc_beta_f_total_log_z(
            numpy.log(_fig4c_Z_exact), _fig4bc_N
        )
        _fig4c_msgs, _, _ = belief_propagation(
            _fig4c_tn, max_iterations=450, damping_factor=0.25
        )
        _fig4c_f_bp = _fig4bc_beta_f_total_log_z(
            compute_bp_partition(_fig4c_tn, _fig4c_msgs), _fig4bc_N
        )
        _fig4c_bp_abs_delta.append(abs(_fig4c_f_bp - _fig4c_bf_exact))
    _fig4bc_axes[1].semilogy(
        _fig4c_beta_grid,
        _fig4c_bp_abs_delta,
        "k--",
        lw=1.5,
        label="BP vacuum",
    )
    for _fig4c_w in _fig4bc_weights:
        _fig4c_err: list[float] = []
        for _fig4c_beta in _fig4c_beta_grid:
            _fig4c_tn = make_ising_tn(
                _fig4bc_L,
                _fig4bc_L,
                _fig4c_beta,
                periodic_boundary=False,
            )
            _fig4c_Z_exact = ising_partition_brute_force(
                _fig4c_beta,
                _fig4bc_L,
                _fig4bc_L,
                periodic_boundary=False,
            )
            _fig4c_bf_exact = _fig4bc_beta_f_total_log_z(
                numpy.log(_fig4c_Z_exact), _fig4bc_N
            )
            _fig4c_msgs, _, _ = belief_propagation(
                _fig4c_tn, max_iterations=450, damping_factor=0.25
            )
            _fig4c_logz, _, _ = log_partition_cluster(
                _fig4c_tn,
                _fig4c_msgs,
                max_total_edge_weight=_fig4c_w,
                max_multiset_size=_fig4bc_multiset_cap(_fig4c_w),
                use_simple_cycle_basis_only=_fig4bc_use_simple_cycles_only,
            )
            _fig4c_bf_cl = _fig4bc_beta_f_total_log_z(_fig4c_logz, _fig4bc_N)
            _fig4c_err.append(abs(_fig4c_bf_cl - _fig4c_bf_exact))
        _fig4bc_axes[1].semilogy(
            _fig4c_beta_grid,
            _fig4c_err,
            "-",
            label=f"cluster w≤{_fig4c_w}",
        )
    _fig4bc_axes[1].axvline(
        _fig4bc_beta_c_onsager, color="gray", ls=":", alpha=0.7
    )
    _fig4bc_axes[1].axvline(
        _fig4bc_beta_c_bethe, color="blue", ls=":", alpha=0.7
    )
    _fig4bc_axes[1].set_xlabel("β")
    _fig4bc_axes[1].set_ylabel(
        r"$|\delta(\beta f)| = |(\beta f)_{\rm approx} - (\beta f)_{\rm exact}|$"
    )
    _fig4bc_axes[1].set_title(r"Fig. 4(c): $|\delta(\beta f)(\beta)|$ (BP dashed)")
    _fig4bc_axes[1].legend(fontsize=7, ncol=2)
    _fig4bc_axes[1].grid(True, alpha=0.3)
    matplotlib_pyplot.tight_layout()
    _fig4bc_header = mo.md(
        "### V.2 Cluster expansion\n\n"
        "**Fig. 4(b–c):** $4\\times4$ open, exact by brute force; **$y$ axis is $\\beta f=-\\ln\\mathcal{Z}/N$** "
        "(Fig. 4(a)). Cluster loops use the **simple-cycle** basis (`collect_loops` when $|E|>20$). "
        "Weights $w\\in\\{4,6,8,10\\}$ as in the paper."
    )
    mo.vstack([_fig4bc_header, _fig4bc_figure])
    return


@app.cell
def _(
    belief_propagation,
    compute_bp_partition,
    compute_loop_tensor,
    enumerate_paper_loops,
    ising_partition_brute_force,
    log_partition_cluster,
    make_ising_tn,
    matplotlib_pyplot,
    mo,
    numpy,
):
    _fig4d_beta_bp = numpy.log(2.0) / 2.0
    _fig4d_weights = [4, 6, 8, 10]

    def _fig4d_multiset_cap(edge_weight_limit: int) -> int:
        return min(8, max(4, edge_weight_limit // 2 + 2))

    def _fig4d_beta_f_from_total_log_z(total_log_z: float, site_count: int) -> float:
        return -float(total_log_z) / site_count

    _fig4d_figure, _fig4d_axis = matplotlib_pyplot.subplots(figsize=(6.5, 4))

    _fig4d_L_loop = 3
    _fig4d_N_loop = _fig4d_L_loop**2
    _fig4d_tn_loop = make_ising_tn(
        _fig4d_L_loop,
        _fig4d_L_loop,
        _fig4d_beta_bp,
        periodic_boundary=False,
    )
    _fig4d_Z_ex_loop = ising_partition_brute_force(
        _fig4d_beta_bp,
        _fig4d_L_loop,
        _fig4d_L_loop,
        periodic_boundary=False,
    )
    _fig4d_beta_f_ex_loop = _fig4d_beta_f_from_total_log_z(
        numpy.log(_fig4d_Z_ex_loop), _fig4d_N_loop
    )
    _fig4d_msgs_loop, _, _ = belief_propagation(
        _fig4d_tn_loop, max_iterations=500, damping_factor=0.25
    )
    _fig4d_Z0 = numpy.exp(
        compute_bp_partition(_fig4d_tn_loop, _fig4d_msgs_loop)
    )
    _fig4d_loop_abs_delta: list[float] = []
    for _fig4d_w in _fig4d_weights:
        _fig4d_loops = enumerate_paper_loops(
            _fig4d_tn_loop.graph,
            max_edge_weight=_fig4d_w,
            max_loop_count=80000,
        )
        _fig4d_sum_l = sum(
            compute_loop_tensor(_fig4d_tn_loop, _fig4d_msgs_loop, _fig4d_lp)
            for _fig4d_lp in _fig4d_loops
        )
        _fig4d_Z_rec = _fig4d_Z0 + _fig4d_sum_l
        if _fig4d_Z_rec <= 0 or not numpy.isfinite(_fig4d_Z_rec):
            _fig4d_loop_abs_delta.append(float("nan"))
        else:
            _fig4d_beta_f_loop = _fig4d_beta_f_from_total_log_z(
                numpy.log(_fig4d_Z_rec), _fig4d_N_loop
            )
            _fig4d_loop_abs_delta.append(
                abs(_fig4d_beta_f_loop - _fig4d_beta_f_ex_loop)
            )
    _fig4d_axis.semilogy(
        _fig4d_weights,
        _fig4d_loop_abs_delta,
        "o:",
        color="crimson",
        mfc="none",
        label="loop (L=3, Lemma II.2)",
    )

    for _fig4d_Lc in (3, 4):
        _fig4d_n = _fig4d_Lc**2
        _fig4d_tn_c = make_ising_tn(
            _fig4d_Lc,
            _fig4d_Lc,
            _fig4d_beta_bp,
            periodic_boundary=False,
        )
        _fig4d_Z_ex_c = ising_partition_brute_force(
            _fig4d_beta_bp,
            _fig4d_Lc,
            _fig4d_Lc,
            periodic_boundary=False,
        )
        _fig4d_beta_f_ex_c = _fig4d_beta_f_from_total_log_z(
            numpy.log(_fig4d_Z_ex_c), _fig4d_n
        )
        _fig4d_msgs_c, _, _ = belief_propagation(
            _fig4d_tn_c, max_iterations=500, damping_factor=0.25
        )
        _fig4d_cycle_only = _fig4d_tn_c.graph.number_of_edges() > 20
        _fig4d_cluster_abs_delta: list[float] = []
        for _fig4d_wc in _fig4d_weights:
            _fig4d_logz_c, _, _ = log_partition_cluster(
                _fig4d_tn_c,
                _fig4d_msgs_c,
                max_total_edge_weight=_fig4d_wc,
                max_multiset_size=_fig4d_multiset_cap(_fig4d_wc),
                use_simple_cycle_basis_only=_fig4d_cycle_only,
            )
            _fig4d_beta_f_c = _fig4d_beta_f_from_total_log_z(
                _fig4d_logz_c, _fig4d_n
            )
            _fig4d_cluster_abs_delta.append(
                abs(_fig4d_beta_f_c - _fig4d_beta_f_ex_c)
            )
        _fig4d_axis.semilogy(
            _fig4d_weights,
            _fig4d_cluster_abs_delta,
            "s--",
            label=f"cluster (L={_fig4d_Lc})",
        )

    _fig4d_axis.set_xticks(_fig4d_weights)
    _fig4d_axis.set_xlabel(r"truncation weight $w$")
    _fig4d_axis.set_ylabel(
        r"$|\delta(\beta f)|$ at $\beta_{\rm BP}=\ln 2/2$"
    )
    _fig4d_axis.set_title(
        "Fig. 4(d): error vs $w$ (small $L$; paper uses $L\\in\\{10,20,30\\}$)"
    )
    _fig4d_axis.legend(fontsize=8)
    _fig4d_axis.grid(True, alpha=0.3)
    matplotlib_pyplot.tight_layout()
    mo.md(
        "### V.3 Convergence: clusters vs. loops\n\n"
        r"**Fig. 4(d):** $|\delta(\beta f)(\beta_{\rm BP})|$ vs $w$; **loop** = $Z_0+\sum_{\ell\in\mathcal{L}_G}Z_\ell$ "
        r"on $3\times3$ (full generalized loops, $|\ell|\le w$); **cluster** = Sec. III expansion. "
        r"Eq. (27)–(28) in the paper explains why naïve loop densities vanish as $L\to\infty$ while cluster "
        r"corrections can remain $O(1)$ in $f$ (here errors use $\beta f=-\ln\mathcal{Z}/N$ like Fig. 4(a))."
    )
    _fig4d_figure
    return


@app.cell
def _(
    belief_propagation,
    compute_loop_tensor,
    enumerate_simple_cycles_as_edges,
    make_ising_tn,
    matplotlib_pyplot,
    mo,
    normalize_tensors,
    numpy,
):
    _loop_magnitude_lattice_linear_size = 4
    _loop_magnitude_bethe_beta = numpy.log(2.0) / 2.0
    _loop_magnitude_onsager_beta = 0.5 * numpy.log(1 + numpy.sqrt(2))
    _loop_magnitude_periodic_boundary = True
    _loop_magnitude_inverse_temperature_grid = numpy.linspace(
        0.2, 0.6, 40
    )
    _loop_magnitude_target_weights = [4, 6, 8, 10]
    _loop_magnitude_figure, _loop_magnitude_axes_pair = (
        matplotlib_pyplot.subplots(1, 2, figsize=(10, 4))
    )
    for target_loop_weight in _loop_magnitude_target_weights:
        mean_loop_magnitude_curve: list[float] = []
        for _loop_magnitude_beta_sample in _loop_magnitude_inverse_temperature_grid:
            _loop_magnitude_tensor_network = make_ising_tn(
                _loop_magnitude_lattice_linear_size,
                _loop_magnitude_lattice_linear_size,
                _loop_magnitude_beta_sample,
                periodic_boundary=_loop_magnitude_periodic_boundary,
            )
            _loop_magnitude_messages, _, _ = belief_propagation(
                _loop_magnitude_tensor_network,
                max_iterations=800,
                damping_factor=0.25,
                message_noise_standard_deviation=0.0,
            )
            _loop_magnitude_normalized_network, _, _ = normalize_tensors(
                _loop_magnitude_tensor_network, _loop_magnitude_messages
            )
            matching_cycles = [
                cycle_edges
                for cycle_edges in enumerate_simple_cycles_as_edges(
                    _loop_magnitude_tensor_network.graph, target_loop_weight
                )
                if len(cycle_edges) == target_loop_weight
            ]
            if not matching_cycles:
                mean_loop_magnitude_curve.append(float("nan"))
                continue
            loop_tensor_magnitudes = [
                abs(
                    compute_loop_tensor(
                        _loop_magnitude_normalized_network,
                        _loop_magnitude_messages,
                        cycle_edges,
                    )
                )
                for cycle_edges in matching_cycles
            ]
            mean_loop_magnitude_curve.append(
                float(numpy.mean(loop_tensor_magnitudes))
            )
        _loop_magnitude_axes_pair[0].plot(
            _loop_magnitude_inverse_temperature_grid,
            mean_loop_magnitude_curve,
            "-",
            label=f"|ℓ|={target_loop_weight}",
        )
    _loop_magnitude_axes_pair[0].axvline(
        _loop_magnitude_bethe_beta, color="blue", ls=":", label="β_BP (Bethe z=4)"
    )
    _loop_magnitude_axes_pair[0].axvline(
        _loop_magnitude_onsager_beta,
        color="gray",
        ls=":",
        alpha=0.6,
        label="β_c (Onsager)",
    )
    _loop_magnitude_axes_pair[0].set_xlabel("β")
    _loop_magnitude_axes_pair[0].set_ylabel(r"mean $|Z_\ell|$ (normalized TN)")
    _loop_magnitude_axes_pair[0].set_title(
        f"Fig. 5(a): L={_loop_magnitude_lattice_linear_size} torus (periodic BC)"
    )
    _loop_magnitude_axes_pair[0].legend(fontsize=8)
    _loop_magnitude_axes_pair[0].grid(True, alpha=0.3)
    beta_label_pairs: list[tuple[float, str]] = [
        (0.2, "β=0.2"),
        (_loop_magnitude_bethe_beta, "β_BP"),
        (0.5, "β=0.5"),
    ]
    for reference_beta, curve_label in beta_label_pairs:
        _loop_magnitude_tensor_network = make_ising_tn(
            _loop_magnitude_lattice_linear_size,
            _loop_magnitude_lattice_linear_size,
            reference_beta,
            periodic_boundary=_loop_magnitude_periodic_boundary,
        )
        _loop_magnitude_messages, _, _ = belief_propagation(
            _loop_magnitude_tensor_network,
            max_iterations=800,
            damping_factor=0.25,
            message_noise_standard_deviation=0.0,
        )
        _loop_magnitude_normalized_network, _, _ = normalize_tensors(
            _loop_magnitude_tensor_network, _loop_magnitude_messages
        )
        mean_magnitude_by_loop_weight: dict[int, float] = {}
        _fig5b_max_loop_weight = min(
            18,
            _loop_magnitude_tensor_network.graph.number_of_nodes(),
        )
        for loop_weight in range(4, _fig5b_max_loop_weight + 1):
            cycles_at_weight = [
                cycle_edges
                for cycle_edges in enumerate_simple_cycles_as_edges(
                    _loop_magnitude_tensor_network.graph, loop_weight
                )
                if len(cycle_edges) == loop_weight
            ]
            if not cycles_at_weight:
                continue
            mean_magnitude_by_loop_weight[loop_weight] = float(
                numpy.mean(
                    [
                        abs(
                            compute_loop_tensor(
                                _loop_magnitude_normalized_network,
                                _loop_magnitude_messages,
                                cycle_edges,
                            )
                        )
                        for cycle_edges in cycles_at_weight
                    ]
                )
            )
        _fig5b_w_sorted = sorted(mean_magnitude_by_loop_weight.keys())
        _fig5b_y_sorted = [
            mean_magnitude_by_loop_weight[w] for w in _fig5b_w_sorted
        ]
        _loop_magnitude_axes_pair[1].plot(
            _fig5b_w_sorted,
            _fig5b_y_sorted,
            "o-",
            markersize=5,
            label=curve_label,
        )
    _loop_magnitude_axes_pair[1].set_xlabel(r"loop weight $w$")
    _loop_magnitude_axes_pair[1].set_ylabel(r"mean $|Z_\ell|$ (normalized TN)")
    _loop_magnitude_axes_pair[1].set_title(
        f"Fig. 5(b): decay vs $w$ (linear $y$; L={_loop_magnitude_lattice_linear_size} torus)"
    )
    _loop_magnitude_axes_pair[1].legend(fontsize=8)
    _loop_magnitude_axes_pair[1].grid(True, alpha=0.3)
    matplotlib_pyplot.tight_layout()
    _loop_magnitude_header = mo.md(
        "### V.4 Loop contribution analysis\n\n"
        "Sec. V benchmarks use **periodic** BC on the square lattice; **open** boundaries break "
        "bulk $z=4$ homogeneity so loop magnitudes tend to track **Onsager** $\\beta_c\\approx 0.441$ "
        "instead of peaking at **Bethe** $\\beta_{\\rm BP}=\\ln 2/2$. Here we match the paper with a torus. "
        "**Fig. 5(b)** uses a **linear** $y$-axis like the paper; a log $y$-scale would make "
        "roughly exponential decay in $w$ look like straight lines."
    )
    mo.vstack([_loop_magnitude_header, _loop_magnitude_figure])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## VI Discussion

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

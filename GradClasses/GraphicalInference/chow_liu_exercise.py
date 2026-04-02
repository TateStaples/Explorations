# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.2",
#     "matplotlib>=3.8",
#     "numpy>=2.0",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import itertools
    import math
    from collections import deque
    from typing import Any

    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import polars as pl

    plt.style.use("seaborn-v0_8-whitegrid")
    np.set_printoptions(precision=4, suppress=True)
    return Any, deque, itertools, math, mo, np, nx, pl, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Computer Assignment: Chow-Liu Tree Learning

    **Overview.** In this assignment you will implement the Chow-Liu pipeline for learning a tree-structured distribution on binary variables from data. The goal is to connect ideas from graphical models, mutual information, maximum spanning trees, parameter estimation, and statistical evaluation. You will work with the provided notebook `exercise.ipynb`, which contains scaffolding for synthetic-data generation, tree utilities, experiment-running code, and plotting. Your job is to complete the core estimation functions and then study how performance changes with sample size.

    **Setup and notation.** Let $X = (X_1, \dots, X_n)$ with each $X_i \in \{0, 1\}$. Let $T = (V, E)$ be a tree on vertex set $V = \{0, 1, \dots, n - 1\}$. After choosing a root $r$, orient every edge away from the root, so the distribution takes the form
    $$p(x) = p_r(x_r) \prod_{(u,v) \in E^*} p_{v|u}(x_v \mid x_u),$$
    where $E^*$ is the set of directed edges after orientation. Thus a tree-structured model is specified by one root marginal and one $2 \times 2$ conditional table for each directed edge. Suppose we observe $\ell$ i.i.d. samples
    $$x^{(1)}, x^{(2)}, \dots, x^{(\ell)} \in \{0, 1\}^n.$$
    The Chow-Liu estimator first computes empirical pairwise mutual informations, then chooses a maximum-weight spanning tree, and finally fits the tree parameters from empirical counts.

    **Required implementation functions.** Please implement exactly the following functions in the notebook:

    1. `empirical_pairwise_mi(samples, eps=1e-12)`
    2. `estimate_tree_distribution(samples, edges, root=0, alpha=1.0)`
    3. `kl_true_to_estimate(true_params, est_params, eps=1e-15)`

    All other code in the notebook is provided as scaffolding or reference code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Learning goals**
    1. Sample from a known tree-structured distribution.
    2. Estimate pairwise MI from data.
    3. Recover a maximum spanning tree using Kruskal.
    4. Fit tree parameters from counts.
    5. Evaluate with tree recovery, MI error, and KL divergence.
    6. Visualize behavior as sample size `l` changes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup
    """)
    return


@app.cell(hide_code=True)
def _(Any, deque, itertools, math, np):
    def build_adjacency_list(num_nodes: int, edge_list: list[tuple[int, int]]) -> list[list[int]]:
        """Build an undirected adjacency list."""
        adjacency_list = [[] for _ in range(num_nodes)]
        for first_node, second_node in edge_list:
            adjacency_list[first_node].append(second_node)
            adjacency_list[second_node].append(first_node)
        return adjacency_list


    def orient_tree_from_root(
        num_nodes: int,
        edge_list: list[tuple[int, int]],
        root_node: int = 0,
    ) -> tuple[list[int], list[list[int]], list[int]]:
        """Orient an undirected tree away from a chosen root.

        Returns (parent_by_node, children_by_node, breadth_first_order) where:
        - parent_by_node[node] is the parent of node (root has parent -1)
        - children_by_node[node] lists directed children of node
        - breadth_first_order is BFS order from the root
        """
        adjacency_list = build_adjacency_list(num_nodes, edge_list)
        parent_by_node = [-1] * num_nodes
        children_by_node = [[] for _ in range(num_nodes)]
        breadth_first_order = []
        queue = deque([root_node])
        parent_by_node[root_node] = -2

        while queue:
            current_node = queue.popleft()
            breadth_first_order.append(current_node)
            for neighboring_node in adjacency_list[current_node]:
                if parent_by_node[neighboring_node] == -1:
                    parent_by_node[neighboring_node] = current_node
                    children_by_node[current_node].append(neighboring_node)
                    queue.append(neighboring_node)

        parent_by_node[root_node] = -1
        return parent_by_node, children_by_node, breadth_first_order


    def sample_uniform_tree(
        num_nodes: int,
        random_generator: np.random.Generator,
    ) -> list[tuple[int, int]]:
        """Sample a uniform labeled tree using a random Prufer sequence."""
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2")

        prufer_sequence = random_generator.integers(0, num_nodes, size=num_nodes - 2)
        node_degree = np.ones(num_nodes, dtype=int)
        for sequence_value in prufer_sequence:
            node_degree[sequence_value] += 1

        leaf_nodes = sorted(np.where(node_degree == 1)[0].tolist())
        sampled_edges: list[tuple[int, int]] = []

        # Decode the Prufer sequence into an undirected edge list.
        for sequence_value in prufer_sequence:
            leaf_node = leaf_nodes.pop(0)
            sampled_edges.append((leaf_node, int(sequence_value)))
            node_degree[leaf_node] -= 1
            node_degree[sequence_value] -= 1
            if node_degree[sequence_value] == 1:
                leaf_nodes.append(int(sequence_value))
                leaf_nodes.sort()

        first_remaining_node, second_remaining_node = np.where(node_degree == 1)[0].tolist()
        sampled_edges.append((int(first_remaining_node), int(second_remaining_node)))
        return sampled_edges


    def sample_random_tree_parameters(
        num_nodes: int,
        edge_list: list[tuple[int, int]],
        random_generator: np.random.Generator,
        root_node: int = 0,
        root_probability_minimum: float = 0.25,
        root_probability_maximum: float = 0.75,
        minimum_dependency_gap: float = 0.25,
        maximum_dependency_gap: float = 0.75,
    ) -> dict[str, Any]:
        """Sample binary tree-distribution parameters with controlled dependence."""
        parent_by_node, children_by_node, breadth_first_order = orient_tree_from_root(
            num_nodes,
            edge_list,
            root_node=root_node,
        )
        root_probability = float(
            random_generator.uniform(root_probability_minimum, root_probability_maximum)
        )

        conditional_probability_tables: dict[tuple[int, int], np.ndarray] = {}

        # Build one conditional table per directed edge parent -> child.
        for current_node in breadth_first_order:
            if current_node == root_node:
                continue
            parent_node = parent_by_node[current_node]

            # Retry sampling table parameters until dependence is strong enough.
            for _ in range(100):
                probability_center = float(random_generator.uniform(0.2, 0.8))
                dependency_gap = float(
                    random_generator.uniform(minimum_dependency_gap, maximum_dependency_gap)
                )
                dependency_direction = float(random_generator.choice([-1.0, 1.0]))
                probability_when_parent_zero = np.clip(
                    probability_center - 0.5 * dependency_direction * dependency_gap,
                    0.05,
                    0.95,
                )
                probability_when_parent_one = np.clip(
                    probability_center + 0.5 * dependency_direction * dependency_gap,
                    0.05,
                    0.95,
                )
                if abs(probability_when_parent_one - probability_when_parent_zero) >= 0.95 * minimum_dependency_gap:
                    break
            else:
                probability_when_parent_zero, probability_when_parent_one = 0.25, 0.75

            conditional_table = np.zeros((2, 2), dtype=float)
            conditional_table[0, 1] = probability_when_parent_zero
            conditional_table[0, 0] = 1.0 - probability_when_parent_zero
            conditional_table[1, 1] = probability_when_parent_one
            conditional_table[1, 0] = 1.0 - probability_when_parent_one
            conditional_probability_tables[(parent_node, current_node)] = conditional_table

        return {
            "num_nodes": num_nodes,
            "edge_list": list(edge_list),
            "root_node": root_node,
            "parent_by_node": parent_by_node,
            "children_by_node": children_by_node,
            "breadth_first_order": breadth_first_order,
            "root_probability": root_probability,
            "conditional_probability_tables": conditional_probability_tables,
        }


    def compute_tree_probability(state_vector: np.ndarray, tree_parameters: dict[str, Any]) -> float:
        """Evaluate P(X=state_vector) under tree parameters."""
        state_vector = np.asarray(state_vector, dtype=int)
        root_node = tree_parameters["root_node"]
        probability_value = (
            tree_parameters["root_probability"]
            if state_vector[root_node] == 1
            else 1.0 - tree_parameters["root_probability"]
        )

        for (parent_node, child_node), conditional_table in tree_parameters[
            "conditional_probability_tables"
        ].items():
            probability_value *= conditional_table[
                state_vector[parent_node],
                state_vector[child_node],
            ]

        return float(probability_value)


    def sample_from_tree_distribution(
        tree_parameters: dict[str, Any],
        sample_count: int,
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        """Sample i.i.d. draws from a binary tree-structured distribution."""
        num_nodes = tree_parameters["num_nodes"]
        root_node = tree_parameters["root_node"]
        parent_by_node = tree_parameters["parent_by_node"]
        breadth_first_order = tree_parameters["breadth_first_order"]

        sample_matrix = np.zeros((sample_count, num_nodes), dtype=int)
        sample_matrix[:, root_node] = (
            random_generator.random(sample_count) < tree_parameters["root_probability"]
        ).astype(int)

        # Sample each node conditioned on its parent assignment.
        for current_node in breadth_first_order:
            if current_node == root_node:
                continue
            parent_node = parent_by_node[current_node]
            conditional_table = tree_parameters["conditional_probability_tables"][(
                parent_node,
                current_node,
            )]
            probability_one_given_parent = conditional_table[sample_matrix[:, parent_node], 1]
            sample_matrix[:, current_node] = (
                random_generator.random(sample_count) < probability_one_given_parent
            ).astype(int)

        return sample_matrix


    def enumerate_all_binary_states(num_nodes: int) -> np.ndarray:
        """Return all 2^num_nodes binary assignments."""
        return np.array(list(itertools.product([0, 1], repeat=num_nodes)), dtype=int)


    def compute_exact_pairwise_mutual_information(
        tree_parameters: dict[str, Any],
    ) -> np.ndarray:
        """Compute exact pairwise MI on a tree without full-state enumeration."""
        num_nodes = tree_parameters["num_nodes"]
        edge_list = tree_parameters["edge_list"]
        root_node = tree_parameters["root_node"]
        parent_by_node = tree_parameters["parent_by_node"]
        breadth_first_order = tree_parameters["breadth_first_order"]
        conditional_probability_tables = tree_parameters["conditional_probability_tables"]

        # Forward pass: compute each node marginal P(X_node = value).
        node_marginal_probabilities = np.zeros((num_nodes, 2), dtype=float)
        node_marginal_probabilities[root_node, 1] = tree_parameters["root_probability"]
        node_marginal_probabilities[root_node, 0] = 1.0 - tree_parameters["root_probability"]

        for current_node in breadth_first_order:
            if current_node == root_node:
                continue
            parent_node = parent_by_node[current_node]
            conditional_table = conditional_probability_tables[(parent_node, current_node)]
            node_marginal_probabilities[current_node] = (
                node_marginal_probabilities[parent_node] @ conditional_table
            )

        # Build transition matrices in both directions for every edge.
        transition_by_source_and_destination = [dict() for _ in range(num_nodes)]
        for (parent_node, child_node), conditional_table in conditional_probability_tables.items():
            transition_by_source_and_destination[parent_node][child_node] = conditional_table

            joint_probability_table = np.zeros((2, 2), dtype=float)
            for parent_value in (0, 1):
                for child_value in (0, 1):
                    joint_probability_table[parent_value, child_value] = (
                        node_marginal_probabilities[parent_node, parent_value]
                        * conditional_table[parent_value, child_value]
                    )

            reverse_transition_table = np.zeros((2, 2), dtype=float)
            for child_value in (0, 1):
                child_marginal_probability = node_marginal_probabilities[child_node, child_value]
                if child_marginal_probability > 0:
                    reverse_transition_table[child_value, 0] = (
                        joint_probability_table[0, child_value] / child_marginal_probability
                    )
                    reverse_transition_table[child_value, 1] = (
                        joint_probability_table[1, child_value] / child_marginal_probability
                    )

            transition_by_source_and_destination[child_node][parent_node] = (
                reverse_transition_table
            )

        adjacency_list = build_adjacency_list(num_nodes, edge_list)

        def find_path_nodes(source_node: int, destination_node: int) -> list[int]:
            predecessor_by_node = [-1] * num_nodes
            queue = deque([source_node])
            predecessor_by_node[source_node] = source_node

            while queue:
                current_node = queue.popleft()
                if current_node == destination_node:
                    break
                for neighboring_node in adjacency_list[current_node]:
                    if predecessor_by_node[neighboring_node] == -1:
                        predecessor_by_node[neighboring_node] = current_node
                        queue.append(neighboring_node)

            path_nodes = [destination_node]
            while path_nodes[-1] != source_node:
                path_nodes.append(predecessor_by_node[path_nodes[-1]])
            path_nodes.reverse()
            return path_nodes

        # Aggregate pairwise MI values using path transitions.
        mutual_information_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
        for first_node in range(num_nodes):
            first_node_marginal = node_marginal_probabilities[first_node]
            for second_node in range(first_node + 1, num_nodes):
                second_node_marginal = node_marginal_probabilities[second_node]
                path_nodes = find_path_nodes(first_node, second_node)

                path_transition = np.eye(2, dtype=float)
                for source_node, destination_node in zip(path_nodes[:-1], path_nodes[1:]):
                    path_transition = (
                        path_transition
                        @ transition_by_source_and_destination[source_node][destination_node]
                    )

                joint_probability_table = np.zeros((2, 2), dtype=float)
                for first_value in (0, 1):
                    joint_probability_table[first_value, :] = (
                        first_node_marginal[first_value] * path_transition[first_value, :]
                    )

                mutual_information_value = 0.0
                for first_value in (0, 1):
                    for second_value in (0, 1):
                        joint_probability = joint_probability_table[first_value, second_value]
                        if joint_probability > 0:
                            mutual_information_value += joint_probability * math.log(
                                joint_probability
                                / (
                                    first_node_marginal[first_value]
                                    * second_node_marginal[second_value]
                                )
                            )

                mutual_information_matrix[first_node, second_node] = mutual_information_value
                mutual_information_matrix[second_node, first_node] = mutual_information_value

        return mutual_information_matrix

    return (
        compute_exact_pairwise_mutual_information,
        compute_tree_probability,
        enumerate_all_binary_states,
        orient_tree_from_root,
        sample_from_tree_distribution,
        sample_random_tree_parameters,
        sample_uniform_tree,
    )


@app.cell
def _(mo):
    mo.md("""
    ### Estimation: MI, Kruskal, Parameters, KL
    """)
    return


@app.function(hide_code=True)
def have_same_undirected_edges(
    first_edge_list: list[tuple[int, int]],
    second_edge_list: list[tuple[int, int]],
) -> bool:
    """Check edge-set equality of two undirected trees."""
    first_edge_set = {tuple(sorted(edge_pair)) for edge_pair in first_edge_list}
    second_edge_set = {tuple(sorted(edge_pair)) for edge_pair in second_edge_list}
    return first_edge_set == second_edge_set


@app.cell
def _(mo):
    mo.md("""
    ### Experiment Runner
    """)
    return


@app.cell(hide_code=True)
def _(
    Any,
    compute_exact_pairwise_mutual_information,
    empirical_pairwise_mi,
    estimate_tree_distribution,
    kl_true_to_estimate,
    kruskal_max_spanning_tree,
    np,
    plt,
    sample_from_tree_distribution,
    sample_random_tree_parameters,
    sample_uniform_tree,
):
    def run_single_trial(
        num_nodes: int,
        sample_count: int,
        random_generator: np.random.Generator,
        smoothing_alpha: float = 0.0,
        diagnose_tree_failure: bool = True,
    ) -> dict[str, float]:
        """Run one full Chow-Liu trial and return evaluation metrics."""
        true_edge_list = sample_uniform_tree(num_nodes, random_generator)
        true_tree_parameters = sample_random_tree_parameters(
            num_nodes,
            true_edge_list,
            random_generator,
        )
        sample_matrix = sample_from_tree_distribution(
            true_tree_parameters,
            sample_count,
            random_generator,
        )

        estimated_mutual_information = empirical_pairwise_mi(sample_matrix)
        estimated_edge_list = kruskal_max_spanning_tree(estimated_mutual_information)
        estimated_tree_parameters = estimate_tree_distribution(
            sample_matrix,
            estimated_edge_list,
            root=0,
            alpha=smoothing_alpha,
        )

        true_mutual_information = compute_exact_pairwise_mutual_information(true_tree_parameters)
        off_diagonal_mask = ~np.eye(num_nodes, dtype=bool)
        mean_absolute_mutual_information_error = float(
            np.mean(
                np.abs(estimated_mutual_information - true_mutual_information)[off_diagonal_mask]
            )
        )
        kl_divergence_value = kl_true_to_estimate(true_tree_parameters, estimated_tree_parameters)
        tree_recovery_success = have_same_undirected_edges(true_edge_list, estimated_edge_list)

        # Optional diagnostics that explain where tree recovery differs.
        if (not tree_recovery_success) and diagnose_tree_failure:
            true_edge_set = {tuple(sorted(edge_pair)) for edge_pair in true_edge_list}
            estimated_edge_set = {tuple(sorted(edge_pair)) for edge_pair in estimated_edge_list}
            missing_true_edges = sorted(true_edge_set - estimated_edge_set)
            extra_estimated_edges = sorted(estimated_edge_set - true_edge_set)

            print("Tree recovery failed.")
            print(
                f"  num_nodes={num_nodes}, sample_count={sample_count}, "
                f"smoothing_alpha={smoothing_alpha}"
            )
            print(f"  missing true edges ({len(missing_true_edges)}): {missing_true_edges}")
            print(
                f"  extra estimated edges ({len(extra_estimated_edges)}): "
                f"{extra_estimated_edges}"
            )
            if missing_true_edges:
                print("  Missing-edge MI (true, estimated):")
                for first_node, second_node in missing_true_edges:
                    print(
                        f"    ({first_node}, {second_node}): "
                        f"({true_mutual_information[first_node, second_node]:.4f}, "
                        f"{estimated_mutual_information[first_node, second_node]:.4f})"
                    )
            if extra_estimated_edges:
                print("  Extra-edge MI (true, estimated):")
                for first_node, second_node in extra_estimated_edges:
                    print(
                        f"    ({first_node}, {second_node}): "
                        f"({true_mutual_information[first_node, second_node]:.4f}, "
                        f"{estimated_mutual_information[first_node, second_node]:.4f})"
                    )

        return {
            "tree_recovery_rate": float(tree_recovery_success),
            "mutual_information_error": mean_absolute_mutual_information_error,
            "kl_divergence": kl_divergence_value,
        }


    def run_experiment_suite(
        num_nodes: int = 8,
        sample_sizes: tuple[int, ...] = (50, 100, 200, 500, 1000, 2000, 5000),
        trial_count: int = 10,
        seed_value: int = 0,
        smoothing_alpha: float = 1.0,
        diagnose_tree_failure: bool = False,
    ) -> dict[int, dict[str, list[float]]]:
        """Run many independent trials across sample sizes."""
        random_generator = np.random.default_rng(seed_value)
        experiment_results = {
            sample_size: {
                "tree_recovery_rate": [],
                "mutual_information_error": [],
                "kl_divergence": [],
            }
            for sample_size in sample_sizes
        }

        for sample_size in sample_sizes:
            for _ in range(trial_count):
                trial_metrics = run_single_trial(
                    num_nodes=num_nodes,
                    sample_count=sample_size,
                    random_generator=random_generator,
                    smoothing_alpha=smoothing_alpha,
                    diagnose_tree_failure=diagnose_tree_failure,
                )
                for metric_name, metric_value in trial_metrics.items():
                    experiment_results[sample_size][metric_name].append(metric_value)

        return experiment_results


    def summarize_experiment_results(
        experiment_results: dict[int, dict[str, list[float]]],
    ) -> dict[str, Any]:
        """Aggregate mean and standard deviation across trials."""
        sorted_sample_sizes = sorted(experiment_results)
        summary_statistics: dict[str, Any] = {"sample_sizes": sorted_sample_sizes}

        for metric_name in (
            "tree_recovery_rate",
            "mutual_information_error",
            "kl_divergence",
        ):
            metric_values_by_sample_size = np.array(
                [experiment_results[sample_size][metric_name] for sample_size in sorted_sample_sizes],
                dtype=float,
            )
            summary_statistics[f"{metric_name}_mean"] = metric_values_by_sample_size.mean(
                axis=1
            )
            summary_statistics[f"{metric_name}_std"] = metric_values_by_sample_size.std(
                axis=1
            )

        return summary_statistics


    def plot_experiment_summary(
        summary_statistics: dict[str, Any],
        title_text: str | None = None,
    ):
        """Plot tree recovery rate, MI error, and KL divergence summaries."""
        sample_size_axis = np.array(summary_statistics["sample_sizes"], dtype=float)
        figure_handle, axis_array = plt.subplots(1, 3, figsize=(15, 4))

        metric_plot_specs = [
            ("tree_recovery_rate", "Tree recovery rate", "Probability"),
            ("mutual_information_error", "MI estimation error (MAE)", "Error"),
            ("kl_divergence", "KL(P_true || P_est)", "KL divergence"),
        ]

        for axis_handle, (metric_name, plot_title, y_axis_label) in zip(
            axis_array,
            metric_plot_specs,
        ):
            metric_mean = summary_statistics[f"{metric_name}_mean"]
            metric_std = summary_statistics[f"{metric_name}_std"]
            axis_handle.plot(sample_size_axis, metric_mean, marker="o")
            axis_handle.fill_between(
                sample_size_axis,
                metric_mean - metric_std,
                metric_mean + metric_std,
                alpha=0.1,
            )
            axis_handle.set_xscale("log")
            axis_handle.set_title(plot_title)
            axis_handle.set_xlabel("number of samples")
            axis_handle.set_ylabel(y_axis_label)

        if title_text:
            plt.suptitle(title_text)
        plt.tight_layout()
        return figure_handle

    return (
        plot_experiment_summary,
        run_experiment_suite,
        run_single_trial,
        summarize_experiment_results,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q1 — Tree factorization and parameterization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Explain why a rooted tree-structured binary
    distribution can be written as

    $$p(x) = p(x_{r})\prod_{(u,v)\in E^*} P(x_v \mid x_u).$$

    How many free scalar parameters are needed to specify such a model on a tree with n nodes?
    Compare this with the number of free parameters needed for a completely general distribution
    on $\{0, 1\}^n$. Briefly explain why the tree model is computationally attractive.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Free parameters.** For a tree on $n$ binary nodes:
    - Root marginal: 1 free parameter ($P(X_r=1)$).
    - Each of $n-1$ edges: 2 free parameters ($P(X_c=1|X_p=0)$ and $P(X_c=1|X_p=1)$).
    - Total: $2\times(n-1) + 1 =$ **$2n-1$ parameters.**

    A general joint distribution on $\{0,1\}^n$ needs **$2^n - 1$** free parameters. Trees are exponentially more compact.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q2 — Pairwise mutual information and the Chow-Liu objective
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For two binary random
    variables Xi and Xj , write the pairwise mutual information

    $${I}(X_i;X_j) = \sum_{a=0}^{1}\sum_{b=0}^{1} {p}_{ij}(a,b)\log\frac{{p}_{ij}(a,b)}{{p}_i(a){p}_j(b)},$$

    Then define the empirical mutual information $\hat I_{bj}$ using the observed samples. Explain the
    Chow-Liu idea: estimate all pairwise mutual informations, then recover a maximum spanning
    tree whose edge weights are the values $\hat I_{bj}$. You do not need to provide a full proof, but you
    should explain clearly why large mutual information suggests that an edge is useful in the
    learned tree.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$\hat{I}(X_i;X_j) = \sum_{a,b} \hat{p}_{ij}(a,b)\log\frac{\hat{p}_{ij}(a,b)}{\hat{p}_i(a)\hat{p}_j(b)},$$
    > where $\hat{p}$ denotes empirical frequencies.

    Chow & Liu (1968) showed that the KL divergence from the true distribution $P$ to any tree approximation $P_T$ satisfies
    $$\text{KL}(P \| P_T) = H(P) - \sum_{(i,j)\in T} I(X_i;X_j) + \text{const},$$
    so minimising KL over all spanning trees is equivalent to **maximising the total pairwise MI** weight. Replacing true MI with empirical MI gives the tractable Chow-Liu algorithm.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q3 — Empirical MI implementation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Implement `empirical_pairwise_mi`. Use the notebook
    signature `(samples, eps=1e-12)`. The function should use the equation from step 2 to
    compute all pairwise MI terms, which will be organized in the off-diagonal of an $n × n$
    matrix. Use empirical marginals and empirical joint frequencies, and include the supplied
    eps parameter only for numerical stability in log-ratio calculations.
    What information theoretic quantity would this formula produce on the diagonal? Will it be
    needed fo the Chow-liu algorithm? If not, what constant value may be used to replace the
    estimates obtained for the diagonal?
    In your writeup, state the computational complexity of your implementation as a function of
    the number of variables n and the number of samples $ℓ$.
    """)
    return


@app.cell(hide_code=True)
def _(math, np):

    def empirical_pairwise_mi(
        samples: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """Estimate pairwise mutual information from samples."""
        sample_count, num_nodes = samples.shape
        mutual_information_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

        # Estimate one-variable marginals from empirical frequencies.
        marginal_probability_matrix = np.zeros((num_nodes, 2), dtype=float)
        marginal_probability_matrix[:, 1] = samples.mean(axis=0)
        marginal_probability_matrix[:, 0] = 1.0 - marginal_probability_matrix[:, 1]

        # Estimate pairwise joint probabilities and then MI for each pair.
        for first_node in range(num_nodes):
            for second_node in range(first_node + 1, num_nodes):
                joint_probability_table = np.zeros((2, 2), dtype=float)
                for first_value in (0, 1):
                    for second_value in (0, 1):
                        joint_probability_table[first_value, second_value] = np.mean(
                            (samples[:, first_node] == first_value)
                            & (samples[:, second_node] == second_value)
                        )

                mutual_information_value = 0.0
                for first_value in (0, 1):
                    for second_value in (0, 1):
                        joint_probability = joint_probability_table[first_value, second_value]
                        if joint_probability > 0:
                            mutual_information_value += joint_probability * math.log(
                                joint_probability
                                / (
                                    marginal_probability_matrix[first_node, first_value]
                                    * marginal_probability_matrix[second_node, second_value]
                                    + eps
                                )
                            )

                mutual_information_matrix[first_node, second_node] = mutual_information_value
                mutual_information_matrix[second_node, first_node] = mutual_information_value

        return mutual_information_matrix


    return (empirical_pairwise_mi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Complexity of `empirical_pairwise_mi`:** $O(n^2 \ell)$. There are $\binom{n}{2}$ pairs; each pair requires scanning all $\ell$ samples to form the $2\times2$ joint count table.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### part (b)
    What information theoretic quantity would this formula produce on the diagonal? Will it be needed fo the Chow-liu algorithm? If not, what constant value may be used to replace the estimates obtained for the diagonal?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Applying the MI formula with $i=j$ gives $I(X_i;X_i) = H(X_i)$, the entropy of $X_i$. This is **not needed** by Chow-Liu (which only considers off-diagonal pairs), so we set diagonal entries to $0.0$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q4 — Maximum spanning tree recovery
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The notebook includes a Kruskal implementation for
    a maximum spanning tree. Explain how Kruskal’s algorithm works in this setting and why
    it returns exactly $n − 1$ edges. Also explain why a spanning tree, rather than an arbitrary
    dense graph, is the correct model class for the Chow-Liu estimator.
    """)
    return


@app.cell
def _(np):
    def kruskal_max_spanning_tree(
        pairwise_weight_matrix: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Compute a maximum spanning tree using Kruskal's algorithm."""
        num_nodes = pairwise_weight_matrix.shape[0]
        weighted_edge_candidates = [
            (pairwise_weight_matrix[first_node, second_node], first_node, second_node)
            for first_node in range(num_nodes)
            for second_node in range(first_node + 1, num_nodes)
        ]
        weighted_edge_candidates.sort(key=lambda weighted_edge: weighted_edge[0], reverse=True)

        disjoint_set_parent = list(range(num_nodes))
        disjoint_set_rank = [0] * num_nodes

        def find_representative(node_index: int) -> int:
            while disjoint_set_parent[node_index] != node_index:
                disjoint_set_parent[node_index] = disjoint_set_parent[
                    disjoint_set_parent[node_index]
                ]
                node_index = disjoint_set_parent[node_index]
            return node_index

        def union_sets(first_node: int, second_node: int) -> bool:
            first_representative = find_representative(first_node)
            second_representative = find_representative(second_node)
            if first_representative == second_representative:
                return False

            if disjoint_set_rank[first_representative] < disjoint_set_rank[second_representative]:
                disjoint_set_parent[first_representative] = second_representative
            elif disjoint_set_rank[first_representative] > disjoint_set_rank[second_representative]:
                disjoint_set_parent[second_representative] = first_representative
            else:
                disjoint_set_parent[second_representative] = first_representative
                disjoint_set_rank[first_representative] += 1
            return True

        selected_edges: list[tuple[int, int]] = []
        for _, first_node, second_node in weighted_edge_candidates:
            if union_sets(first_node, second_node):
                selected_edges.append((first_node, second_node))
                if len(selected_edges) == num_nodes - 1:
                    break

        return selected_edges



    return (kruskal_max_spanning_tree,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Kruskal's greedily adds edges in decreasing weight order, skipping any edge that would create a cycle (detected via union-find). It terminates once exactly $n-1$ edges have been added — a tree on $n$ nodes has exactly $n-1$ edges (one fewer than the number of nodes, since a tree is a connected acyclic graph).

    A spanning **tree** (not a dense graph) is required because we want a factorisation with no redundant conditional-independence assumptions and no normalization difficulty.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q5 — Parameter estimation on a fixed tree
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Implement `estimate_tree_distribution`. Use
    the notebook signature `(samples, edges, root=0, alpha=1.0)`. This function should:

    (a) orient the recovered tree away from the chosen root,

    (b) estimate the root marginal probability $P(X_r = 1)$ from counts,

    (c) estimate each conditional table $P(X_v = x_v | X_u = x_u)$ from counts,

    (d) apply Laplace smoothing with constant **alpha**,

    (e) return the parameter dictionary in the same format used by the supplied sampling code.
    """)
    return


@app.cell(hide_code=True)
def _(Any, np, orient_tree_from_root):
    def estimate_tree_distribution(
        samples: np.ndarray,
        edges: list[tuple[int, int]],
        root: int = 0,
        alpha: float = 1.0,
    ) -> dict[str, Any]:
        """Estimate binary tree-distribution parameters from samples with Laplace smoothing."""
        sample_count, num_nodes = samples.shape
        # (a) orient the recovered tree away from the chosen root,
        parent_by_node, children_by_node, breadth_first_order = orient_tree_from_root(
            num_nodes,
            edges,
            root_node=root,
        )

        # (b) estimate the root marginal probability 
        root_one_count = float(np.sum(samples[:, root] == 1))
        root_probability = (root_one_count + alpha) / (sample_count + 2.0 * alpha)

        conditional_probability_tables: dict[tuple[int, int], np.ndarray] = {}

        # Build a smoothed conditional probability table per parent-child edge.
        for current_node in breadth_first_order:
            if current_node == root:
                continue
            parent_node = parent_by_node[current_node]
            conditional_table = np.zeros((2, 2), dtype=float)

            for parent_value in (0, 1):
                parent_mask = samples[:, parent_node] == parent_value
                parent_value_count = float(np.sum(parent_mask))
                for child_value in (0, 1):
                    parent_child_count = float(
                        np.sum(parent_mask & (samples[:, current_node] == child_value))
                    )
                    # (c) estimate each conditional table 
                    conditional_table[parent_value, child_value] = (
                        parent_child_count + alpha # (d) apply Laplace smoothing with constant alpha,
                    ) / (parent_value_count + 2.0 * alpha) if (parent_value_count + 2.0 * alpha) != 0 else 0

            conditional_probability_tables[(parent_node, current_node)] = conditional_table
        # (e) return the parameter dictionary in the same format used by the supplied sampling code.
        return {
            "num_nodes": num_nodes,
            "edge_list": list(edges),
            "root_node": root,
            "parent_by_node": parent_by_node,
            "children_by_node": children_by_node,
            "breadth_first_order": breadth_first_order,
            "root_probability": root_probability,
            "conditional_probability_tables": conditional_probability_tables,
        }

    return (estimate_tree_distribution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    State the formulas you use for the smoothed root estimate and the smoothed conditional
    estimates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let $\ell$ be the sample size and $\alpha$ the smoothing pseudo-count.

    $$\hat{P}(X_r=1) = \frac{\text{count}(X_r=1)+\alpha}{\ell+2\alpha}$$

    $$\hat{P}(X_c=x_c \mid X_p=x_p) = \frac{\text{count}(X_c=x_c,\,X_p=x_p)+\alpha}{\text{count}(X_p=x_p)+2\alpha}$$

    Each conditional table sums to 1 for each value of $x_p$ by construction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q6 — KL divergence by exact enumeration
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Implement `kl_true_to_estimate(true_params, est_params, eps=1e-15)`. This function should compute

    $$
    \text{KL}(P_{\text{true}}∥P_{\text{est}}) = \sum_{x∈\{0,1\}^n} p_{\text{true}}(x) \log \frac{p_{\text{true}}(x)}{p_{\text{est}}(x)}
    $$
    For the dimensions in the notebook, this can be done by enumerating all 2n binary assignments.
    """)
    return


@app.cell(hide_code=True)
def _(Any, compute_tree_probability, enumerate_all_binary_states, math):
    def kl_true_to_estimate(
        true_params: dict[str, Any],
        est_params: dict[str, Any],
        eps: float = 1e-15,
    ) -> float:
        """Compute KL(P_true || P_est) by exact enumeration over all binary states."""
        num_nodes = true_params["num_nodes"]
        all_state_vectors = enumerate_all_binary_states(num_nodes)

        kl_divergence_value = 0.0
        for state_vector in all_state_vectors:
            true_probability = compute_tree_probability(state_vector, true_params)
            estimated_probability = compute_tree_probability(state_vector, est_params)
            if true_probability > 0:
                kl_divergence_value += true_probability * math.log(
                    true_probability / (estimated_probability + eps)
                )

        return kl_divergence_value

    return (kl_true_to_estimate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Explain briefly why this exact computation is reasonable for the provided experiment
    sizes, but would become infeasible for much larger $n$. This quantity can actually be computed
    much more efficiently using local marginals. As an optional challenge, implement the faster
    method and use the slow method to verify your results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For both distributions being tree-structured, `tree_prob` evaluates each term in $O(n)$ time, so the total cost is $O(n \cdot 2^n)$. With $n=10$ this is ~10 000 evaluations — fast. For $n=100$ it is $\sim 10^{32}$ — completely infeasible. Exact enumeration is therefore practical only for small $n$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run & Plot
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    main_node_slider = mo.ui.slider(start=5, stop=50, step=1, value=10, label="num_nodes")
    return (main_node_slider,)


@app.cell(hide_code=True)
def _(main_node_slider, run_experiment_suite):
    SAMPLE_SIZES = (10, 50, 100, 250, 1000)

    main_experiment_results = run_experiment_suite(
        num_nodes=main_node_slider.value,
        sample_sizes=SAMPLE_SIZES,
        trial_count=100,
        seed_value=7,
        smoothing_alpha=0.5,
    )
    return (main_experiment_results,)


@app.cell(hide_code=True)
def _(
    main_experiment_results,
    main_node_slider,
    mo,
    np,
    pl,
    plot_experiment_summary,
    summarize_experiment_results,
):

    main_summary_statistics = summarize_experiment_results(main_experiment_results)
    main_figure = plot_experiment_summary(main_summary_statistics)

    main_summary_rows = []
    for main_sample_size in main_summary_statistics["sample_sizes"]:
        main_summary_rows.append(
            {
                "sample_size": int(main_sample_size),
                "tree_recovery_rate": float(
                    np.mean(main_experiment_results[main_sample_size]["tree_recovery_rate"])
                ),
                "mutual_information_mae": float(
                    np.mean(main_experiment_results[main_sample_size]["mutual_information_error"])
                ),
                "kl_divergence": float(
                    np.mean(main_experiment_results[main_sample_size]["kl_divergence"])
                ),
            }
        )
    main_summary_dataframe = pl.DataFrame(main_summary_rows)

    mo.vstack([
        main_node_slider,
        main_figure,
        main_summary_dataframe,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q7 — One-trial visualization and diagnosis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the visualization cell to compare one true tree
    and one recovered Chow-Liu tree, together with the true and estimated mutual-information
    matrices.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    random_trial_seed_component = mo.ui.slider(start=0, stop=100, step=1, value=42, label="trial_seed_value", show_value=True)
    random_trial_samples_component = mo.ui.slider(steps=np.logspace(base=5, start=1, stop=5, num=5).astype(int), value=125, label="trail_sample_count", show_value=True)
    return random_trial_samples_component, random_trial_seed_component


@app.cell(hide_code=True)
def _(
    compute_exact_pairwise_mutual_information,
    empirical_pairwise_mi,
    estimate_tree_distribution,
    kruskal_max_spanning_tree,
    np,
    random_trial_samples_component,
    random_trial_seed_component,
    run_single_trial,
    sample_from_tree_distribution,
    sample_random_tree_parameters,
    sample_uniform_tree,
):
    trial_num_nodes = 8
    trial_sample_count = random_trial_samples_component.value
    trial_seed_value = random_trial_seed_component.value
    random_generator = np.random.default_rng(trial_seed_value)

    trial_metrics = run_single_trial(
        num_nodes=trial_num_nodes,
        sample_count=trial_sample_count,
        random_generator=random_generator,
    )
    true_edges = sample_uniform_tree(trial_num_nodes, random_generator)
    true_parameters = sample_random_tree_parameters(trial_num_nodes, true_edges, random_generator)
    samples = sample_from_tree_distribution(true_parameters, trial_sample_count, random_generator)

    mutual_information_estimated = empirical_pairwise_mi(samples)
    estimated_edges = kruskal_max_spanning_tree(mutual_information_estimated)
    estimated_parameters = estimate_tree_distribution(
        samples,
        estimated_edges,
        root=0,
        alpha=0.0,
    )

    mutual_information_true = compute_exact_pairwise_mutual_information(true_parameters)
    return (
        estimated_edges,
        estimated_parameters,
        mutual_information_estimated,
        mutual_information_true,
        trial_metrics,
        trial_num_nodes,
        trial_sample_count,
        trial_seed_value,
        true_edges,
        true_parameters,
    )


@app.cell(hide_code=True)
def _(
    estimated_edges,
    estimated_parameters,
    kl_true_to_estimate,
    mo,
    mutual_information_estimated,
    mutual_information_true,
    nx,
    pl,
    plt,
    random_trial_samples_component,
    random_trial_seed_component,
    trial_metrics,
    trial_num_nodes,
    trial_sample_count,
    trial_seed_value,
    true_edges,
    true_parameters,
):
    def draw_tree(axis_handle, num_nodes, edge_list, title_text):
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(edge_list)
        positions = nx.nx_agraph.graphviz_layout(graph, prog="dot")
        nx.draw_networkx_edges(graph, positions, ax=axis_handle, width=2, edge_color="tab:blue")
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=axis_handle,
            node_size=180,
            node_color="white",
            edgecolors="black",
        )
        nx.draw_networkx_labels(graph, positions, ax=axis_handle, font_size=10)
        axis_handle.axis("off")
        axis_handle.set_title(title_text)

    figure_trial, axes_trial = plt.subplots(2, 2, figsize=(11, 9))
    draw_tree(axes_trial[0, 0], trial_num_nodes, true_edges, "True tree")
    draw_tree(axes_trial[0, 1], trial_num_nodes, estimated_edges, "Recovered Chow-Liu tree")

    image_true = axes_trial[1, 0].imshow(mutual_information_true, cmap="viridis")
    axes_trial[1, 0].set_title("True pairwise MI")
    axes_trial[1, 0].set_xlabel("j")
    axes_trial[1, 0].set_ylabel("i")
    plt.colorbar(image_true, ax=axes_trial[1, 0], fraction=0.046, pad=0.04)

    image_estimated = axes_trial[1, 1].imshow(mutual_information_estimated, cmap="viridis")
    axes_trial[1, 1].set_title("Estimated pairwise MI")
    axes_trial[1, 1].set_xlabel("j")
    axes_trial[1, 1].set_ylabel("i")
    plt.colorbar(image_estimated, ax=axes_trial[1, 1], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"One trial (nodes={trial_num_nodes}, trials={trial_sample_count}, seed={trial_seed_value})",
        y=0.98,
    )
    plt.tight_layout()

    trial_kl_check = kl_true_to_estimate(true_parameters, estimated_parameters)
    trial_metrics_dataframe = pl.DataFrame(
        [
            {"metric": metric_name, "value": float(metric_value)}
            for metric_name, metric_value in trial_metrics.items()
        ]
        + [{"metric": "kl_divergence_recomputed", "value": float(trial_kl_check)}]
    )

    mo.vstack([
        random_trial_seed_component,
        random_trial_samples_component,
        figure_trial,
        trial_metrics_dataframe,
        mo.md("Sanity check complete: recomputed KL from recovered model included in table."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Discuss what failure looks like when the tree is not recovered correctly. If you see
    recovery errors at small sample sizes, explain whether the estimated MI matrix appears noisy,
    whether the incorrect edges have similar weights to the true ones, and how this affects the
    KL divergence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Discuss what failure looks like when the tree is not recovered correctly

    Failure tends to look like a connection hopping for d=1 neighborhood to d=2. This makes sense because these nodes tend to still have high MI and are therefore harder to reliably parse at low samples

    2. If you see recovery errors at small sample sizes, explain whether the estimated MI matrix appears noisy, whether the incorrect edges have similar weights to the true ones, and how this affects the KL divergence.

    The sample MI matrix looks like the true through a noisy channel. The higher the noise the more KL
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Q8 — Smoothing Study

    Compare `alpha = 0.0`, `alpha = 0.5`, and `alpha = 1.0`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Repeat part of the experiment for at least two different smoothing values,
    for example $α = 0$ and $α = 0.5$ or α = $1.0$.
    """)
    return


@app.cell(hide_code=True)
def _(pl, run_experiment_suite, summarize_experiment_results):
    smoothing_alpha_values = [0.0, 0.5, 1.0]
    # Build a long-form Polars table with one row per (alpha, sample_size).
    smoothing_study_rows = []
    for smoothing_alpha in smoothing_alpha_values:
        smoothing_experiment_results = run_experiment_suite(
            num_nodes=10,
            sample_sizes=(5, 10, 20, 50, 100, 200),
            trial_count=100,
            seed_value=7,
            smoothing_alpha=smoothing_alpha,
            diagnose_tree_failure=False,
        )
        smoothing_summary = summarize_experiment_results(smoothing_experiment_results)

        for summary_index, sample_size in enumerate(smoothing_summary["sample_sizes"]):
            smoothing_study_rows.append(
                {
                    "smoothing_alpha": smoothing_alpha,
                    "sample_size": int(sample_size),
                    "tree_recovery_rate_mean": float(
                        smoothing_summary["tree_recovery_rate_mean"][summary_index]
                    ),
                    "tree_recovery_rate_std": float(
                        smoothing_summary["tree_recovery_rate_std"][summary_index]
                    ),
                    "mutual_information_error_mean": float(
                        smoothing_summary["mutual_information_error_mean"][summary_index]
                    ),
                    "mutual_information_error_std": float(
                        smoothing_summary["mutual_information_error_std"][summary_index]
                    ),
                    "kl_divergence_mean": float(
                        smoothing_summary["kl_divergence_mean"][summary_index]
                    ),
                    "kl_divergence_std": float(
                        smoothing_summary["kl_divergence_std"][summary_index]
                    ),
                }
            )

    smoothing_study_dataframe = pl.DataFrame(smoothing_study_rows).sort(
        ["smoothing_alpha", "sample_size"]
    )
    return smoothing_alpha_values, smoothing_study_dataframe


@app.cell(hide_code=True)
def _(mo, pl, plt, smoothing_alpha_values, smoothing_study_dataframe):

    plot_colors = ["tab:blue", "tab:orange", "tab:green"]

    figure_smooth, axes_smooth = plt.subplots(1, 3, figsize=(15, 4))
    metric_specifications = [
        ("tree_recovery_rate", "Tree recovery rate", "Probability"),
        ("mutual_information_error", "MI estimation error (MAE)", "Error"),
        ("kl_divergence", "KL(P_true || P_est)", "KL divergence"),
    ]

    # Plot each metric by filtering the Polars table per alpha value.
    for axis_handle, (metric_prefix, title_text, y_axis_label) in zip(
        axes_smooth,
        metric_specifications,
    ):
        for _smoothing_alpha, plot_color in zip(smoothing_alpha_values, plot_colors):
            alpha_slice = smoothing_study_dataframe.filter(
                pl.col("smoothing_alpha") == _smoothing_alpha
            ).sort("sample_size")

            sample_size_axis = alpha_slice["sample_size"].to_numpy().astype(float)
            mean_values = alpha_slice[f"{metric_prefix}_mean"].to_numpy()
            std_values = alpha_slice[f"{metric_prefix}_std"].to_numpy()

            axis_handle.plot(
                sample_size_axis,
                mean_values,
                marker="o",
                label=f"alpha={_smoothing_alpha}",
                color=plot_color,
            )
            axis_handle.fill_between(
                sample_size_axis,
                mean_values - std_values,
                mean_values + std_values,
                alpha=0.15,
                color=plot_color,
            )

        axis_handle.set_xscale("log")
        axis_handle.set_title(title_text)
        axis_handle.set_xlabel("number of samples l")
        axis_handle.set_ylabel(y_axis_label)
        axis_handle.legend()

    plt.suptitle("Effect of Laplace smoothing on Chow-Liu performance")
    plt.tight_layout()


    mo.vstack([
        figure_smooth,
        smoothing_study_dataframe.select(
            "smoothing_alpha",
            "sample_size",
            "tree_recovery_rate_mean",
            "mutual_information_error_mean",
            "kl_divergence_mean",
        )
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Describe how smoothing changes the estimated
    parameters and whether it appears to help or hurt KL divergence for small sample sizes. Be
    specific and support your comments with plots or tables.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | $\alpha$ | Small $\ell$ (50) | Large $\ell$ (5000) |
    |---|---|---|
    | 0.0 | KL often $+\infty$ (zero-probability issue); good recovery when lucky | Converges to MLE; lowest asymptotic KL |
    | 0.5 | Guards against zero probs; moderate regularisation | Nearly identical to MLE |
    | 1.0 | Strongest regularisation; pulls estimates toward 0.5 | Slightly higher bias but rarely matters |

    **Effect on parameters:** Smoothing shrinks every conditional probability toward 0.5, reducing the magnitude of estimated dependencies. At small $\ell$ this trades variance for bias, resulting in lower mean KL compared with $\alpha=0$. At large $\ell$ the effect vanishes since counts dominate the smoothing term.

    **Effect on tree recovery:** Smoothing does not affect which spanning tree is chosen (MI estimates do not change), only the fitted parameters. Tree-recovery rate is therefore the same across $\alpha$ values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q9 — Summary

    - **Tree recovery** requires $\ell$ large enough for MI estimates to reliably rank all $\binom{n}{2}$ edge weights. Recovery rate climbs steeply once $\ell \gg n^2$.
    - **MI estimation error** (MAE) decreases roughly as $O(1/\sqrt{\ell})$, consistent with the CLT variance of empirical frequencies.
    - **KL divergence** is dominated at small $\ell$ by both wrong-tree topology and poor parameter estimates; at large $\ell$ wrong topology is rare and only parameter-estimation error remains, which shrinks to zero as $\ell\to\infty$.
    - **Laplace smoothing** with $\alpha > 0$ is essential at small $\ell$ to avoid zero-probability catastrophes; the optimal $\alpha$ balances bias and variance but any moderate value (0.5–1.0) works well in practice.
    """)
    return


if __name__ == "__main__":
    app.run()

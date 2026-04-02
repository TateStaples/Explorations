import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from typing import TypeAlias
    import time
    import os
    return TypeAlias, mo, np, os, time


@app.cell
def _(mo):
    mo.md(r"""
    # Computer Assignment: Belief-Propagation for Sudoku

    **Overview.** In this assignment you will model Sudoku as a constraint satisfaction problem (CSP)
    using a factor graph and implement a test solver with sum-product (belief propagation).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Main Homework
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Setup and notation.** Let $x_{r,c} \in \{1, . . . , 9\}$ denote the value in row $r$ and column $c$ for
    $r, c \in \{1, . . . , 9\}$. Let $x$ denote the vector of all 81 variables. A factor graph represents a nonnegative
    function $f(x) = \prod_{a \in F} f_a(x_{\partial a})$, and defines a distribution
    $$
    \mu(x) = \frac{1}{Z} f(x), \quad Z = \sum_x f(x).
    $$

    Observed clues $\{y_{r,c}\}_{(r,c) \in O}$, where $O$ is the set of observed locations, are encoded as unary factors.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### **1. Sudoku as a CSP with observations.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let $f_\sigma(x_1, . . . , x_9)$ be a function that equals 1 if
    $(x_1, . . . , x_9)$ is a permutation of $\{1, 2, 3, 4, 5, 6, 7, 8, 9\}$ and zero otherwise. Describe Sudoku
    as a CSP with three groups of constraints: rows, columns, and $3 \times 3$ subblocks. Show how
    observed clues are defined by unary factors $f_{r,c}(x_{r,c})$ which are 1 for the observed value and
    0 otherwise. Write the full factorization of $f(x)$ using permutation factors and unary factors.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Sudoku can be formulated as a CSP with three types of constraints:
    - **Row constraints:** For each row $r$, the variables $\{x_{r,1}, \ldots, x_{r,9}\}$ must form a permutation of $\{1, \ldots, 9\}$.
    - **Column constraints:** For each column $c$, the variables $\{x_{1,c}, \ldots, x_{9,c}\}$ must form a permutation of $\{1, \ldots, 9\}$.
    - **Block constraints:** For each $3 \times 3$ block $b$, the variables in that block must form a permutation of $\{1, \ldots, 9\}$.

    Observed clues are encoded as unary factors:
    $$
    f_{r,c}(x_{r,c}) =
    \begin{cases}
    1 & \text{if } x_{r,c} = y_{r,c} \text{ (observed)} \\
    1 & \text{if } (r,c) \text{ not observed} \\
    0 & \text{otherwise}
    \end{cases}
    $$

    The full factorization is:
    $$
    f(x) = \prod_{r=1}^9 f_\sigma(x_{r,1}, \ldots, x_{r,9})
        \prod_{c=1}^9 f_\sigma(x_{1,c}, \ldots, x_{9,c})
        \prod_{b=1}^9 f_\sigma(x_{\text{block}_b})
        \prod_{(r,c) \in O} f_{r,c}(x_{r,c})
    $$
    where $O$ is the set of observed cells and $x_{\text{block}_b}$ are the variables in block $b$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### **2. All-different constraints and pairwise factorization.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Each row/column/subgrid constraint $f_\sigma : \mathcal{X}^9 \to \{0, 1\}$ is a permutation constraint on 9 variables taking values in $\{1, 2, . . . , 9\}$
    (i.e., it evaluates to 1 if pattern is a permutation of the alphabet and 0 otherwise). Since the
    number of variables equals the alphabet size, this is equivalent to an all-different constraint
    $f_{\neq} : \mathcal{X}^9 \to \{0, 1\}$ on a set of 9 variables (i.e., it evaluates to 1 if all variables are different from
    each other and 0 otherwise).

    Explain how an all-different constraint $f_{\neq}$ on 9 variables can be factorized into 36 pairwise
    not-equal constraints $f_{\neq} : \mathcal{X}^2 \to \{0, 1\}$ by using the complete graph on those 9 variables. Give
    an explicit formula for $f_\sigma(x_1, . . . , x_9)$ in terms of the pairwise not-equal factors.
    What is the total number of pairwise not-equal constraints required for Sudoku?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    f_\sigma(x_1, \dots, x_9) = \prod_{i=1}^8 \prod_{j=i+1}^9 f_{\neq}(x_i, x_j)
    $$

    We needed 27 $f_{\sigma}$ terms to model the constraints on the 9 rows, 9 columns, and 9 blocks. Since each $f_{\sigma}$ is a product of 36 terms, we had $27 \times 36 = 972$ terms in total.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### **3. Low-complexity BP update for not-equal factors.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    For alphabet size $q$ with pairwise
    not-equal factors $f_{ij}(x_i, x_j) = f_{\neq}(x_i, x_j)$ enforcing $x_i \neq x_j$, use sum-product in the probability
    domain, derive the factor-to-variable update
    $$
    \hat{m}_{ij \to i}(x_i) = \sum_{x_j=1}^q f_{\neq}(x_i, x_j) m_{j \to ij}(x_j).
    $$

    Show this can be computed for all $x_i$ in $O(q)$ time by precomputing $S = \sum_{x_j=1}^q m_{j \to \psi}(x_j)$
    and using
    $$
    \hat{m}_{ij \to i}(x_i) = S - m_{j \to \psi}(x_i).
    $$
    Then write the variable-to-factor update for a Sudoku variable, including its unary factor and
    all incoming pairwise messages.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    \begin{align*}
    \text{ne}(f_{ij}) &= \set{x_{i}, x_{j}}\\
    \hat{m}_{f \to i}(x_i) &= \sum_{\mathbf{x}_{\text{ne}(f) \setminus \{i\}}} \left( f(\mathbf{x}_{\text{ne}(f)}) \prod_{k \in \text{ne}(f) \setminus \{i\}} m_{k \to f}(x_k) \right)\\
    &= \sum_{x_{j}} f_{ij}(x_i, x_j) m_{j \to k}(x_j)\\
    &= \sum_{x_j=1}^q f_{ij}(x_i, x_j) m_{j \to ij}(x_j)
    \end{align*}
    $$

    $$
    \hat{m}_{ij \to i}(x_i) = \underbrace{\left( 0 \cdot m_{j \to ij}(x_i) \right)}_{\text{Case } x_j = x_i} + \underbrace{\sum_{x_j \neq x_i} \left( 1 \cdot m_{j \to ij}(x_j) \right)}_{\text{Case } x_j \neq x_i}
    $$

    We can define S as all the messages
    $$
    S = \sum_{x_j=1}^q m_{j \to ij}(x_j)
    $$

    Now the message is just the all the messages minus the one for $x_i$
    $$
    \hat{m}_{ij \to i}(x_i) = S - m_{j \to ij}(x_i)
    $$

    Computing $S$ takes $O(q)$. Computing $(S - m)$ for each of the $q$ states takes $O(1)$. Thus, the total complexity for updating the message vector is **$O(q)$**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    For a Sudoku variable $x_i$ (a cell in the grid), let:
    * $\phi_i(x_i)$ be the unary factor (representing the initial digit clue).
    * $\mathcal{N}(i)$ be the set of all factor nodes connected to $x_i$ (pairwise constraints from row, column, and block).

    The generic variable-to-factor message from $x_i$ to a specific factor $f_{a}$ (where $f_{a} \in \mathcal{N}(i)$) is the product of all *other* incoming information:

    $$
    m_{i \to a}(x_i) = \phi_i(x_i) \prod_{b \in F(i) \setminus a} \hat{m}_{b \to i}(x_i)
    $$

    In the context of Sudoku, where $f_a$ is a specific pairwise not-equal constraint between $x_i$ and a neighbor $x_j$ (denoted $f_{ij}$), the update is:

    $$
    m_{i \to ij}(x_i) = \phi_i(x_i) \prod_{f \in \mathcal{N}(i) \setminus \{f_{ij}\}} \hat{m}_{f \to i}(x_i)
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### **4. Detecting a satisfying assignment and early termination.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Describe a procedure to
    detect a satisfying assignment during BP iterations. Your factor update procedure can check
    if the most likely symbol for $x_i$ is different from the most-likely symbol for $x_j$ for all $i, j$. Also
    discuss what to do if any belief becomes identically zero (e.g., indicating that a contradiction
    has arised).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The procedure can detect a satisfying assignment by confirming the the most likely symbol for each variable is different than the most likely symbol for all other variables in the same row, column, and block. If this condition is met for all variables, then a satisfying assignment has been found.

    If any belief becomes identically zero, it indicates that a contradiction has arisen. In this case, the algorithm should terminate early and report that no valid solution exists for the given Sudoku puzzle.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### **5. Implementation details.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Please implement the basic BP update loop running 50 iterations
    with damping factor $\alpha$ set to 0.5. Use damping to stabilize updates by combining the standard
    message update $m'_{i \to ij}(x)$ with the previous message $m^{(t)}_{i \to ij}(x)$:
    $$
    m^{(t+1)}_{i \to ij}(x) = (1 - \alpha)m^{(t)}_{i \to ij} + \alpha m'_{i \to ij}(x), \quad x \in \mathcal{X}.
    $$

    After each update, one can normalize messages so they sum to one over the alphabet. Note
    that one can compute the reciprocal and multiply to reduce the number of divisions by $q$.
    This keeps numbers bounded and ensures comparability of beliefs and also forces the value of
    $S$ to 1 in the shortcut above. Describe any other details in the algorithm that you implement.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### **6. Test set**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Use the provided test set of Sudoku puzzles which contains 50 easyish puzzles. My
    BP solver is successful on 27/50 of these examples when I use 400 iterations and 26/50 when
    I use 40 iterations. Each puzzle is represented by 9 lines of 9 digits with 0 used to denote
    blanks. The following test example should decode successfully with BP:
    """)
    return


@app.cell
def _(os):
    # Example puzzles
    EXAMPLE_PUZZLES = {
        "Easy (Test)": """003020600
    900305001
    001806400
    008102900
    700000008
    006708200
    002609500
    800203009
    005010300""",
        "Medium": """530070000
    600195000
    098000060
    800060003
    400803001
    700020006
    060000280
    000419005
    000080079""",
        "Hard": """800000000
    003600000
    070090200
    050007000
    000045700
    000100030
    001000068
    008500010
    090000400""",
    }

    # Load internet puzzles
    _internet_path = None
    _candidate_paths = [
        "files/internet_sudoku.txt",
        "GraphicalInference/files/internet_sudoku.txt",
    ]
    for p in _candidate_paths:
        if os.path.exists(p):
            _internet_path = p
            break

    if _internet_path:
        with open(_internet_path, "r") as f:
            lines = f.readlines()

        current_grid_name = None
        current_grid_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Grid"):
                if current_grid_name and current_grid_lines:
                    EXAMPLE_PUZZLES[current_grid_name] = "\n".join(current_grid_lines)
                current_grid_name = line
                current_grid_lines = []
            else:
                current_grid_lines.append(line)

        # Add the last one
        if current_grid_name and current_grid_lines:
            EXAMPLE_PUZZLES[current_grid_name] = "\n".join(current_grid_lines)
    else:
        print(f"Warning: {_internet_path} not found.")
    return (EXAMPLE_PUZZLES,)


@app.cell
def _(np):
    # Decode sudoku string to numpy array
    def decode_sudoku(s: str) -> np.ndarray:
        digits = [int(c) for c in s if c.isdigit()]
        if len(digits) != 81:
            # Fallback for empty or partial lines, preserve 0s if padded?
            # Assuming standard 81 digit sudoku
            if len(digits) > 81:
                digits = digits[:81]
            elif len(digits) < 81:
                digits = digits + [0] * (81 - len(digits))

        return np.array(digits).reshape((9, 9))
    return (decode_sudoku,)


@app.cell
def _(mo):
    mo.md(r"""
    ### **7. Bells and whistles**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    You will submit the (written completely by hand) BP solver (say for the hard coded example
    above) as part of your assignment along with details requested above. If the code doesn’t
    fully work, that is not preferred but it is ok.
    After that, you are allowed to use generative AI coding tools to do things like: debugging,
    adding command line arguments, having the solver read the sudoku.txt file and execute on
    all examples, report the success rate and timing, add visualization of the decoding process,
    and add guided decimation and backtracking.
    You will also submit the final GenAI version along with a few paragraphs describing excatly
    what features it has and how easy (or hard) it was to implement these with GenAI.
    """)
    return


@app.cell
def _(decode_sudoku, np):
    from collections import defaultdict

    class SudokuBP:
        MAX_ITERATIONS = 40
        GRID_SIZE = 9
        DAMPING = 0.5

        def __init__(self, grid: np.ndarray):
            self.grid = grid.copy()
            self.variables = [
                (r, c) for r in range(self.GRID_SIZE) for c in range(self.GRID_SIZE)
            ]

            # Initialize unary factors
            self.initial_beliefs: np.ndarray = np.zeros(
                (self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE), dtype=float
            )
            for row in range(self.GRID_SIZE):
                for col in range(self.GRID_SIZE):
                    value: int = self.grid[row, col]
                    if value != 0:
                        self.initial_beliefs[row, col, value - 1] = 1.0
                    else:
                        # Unknown cells have uniform prior
                        self.initial_beliefs[row, col, :] = 1.0 / SudokuBP.GRID_SIZE

            # Messages: m_{i -> j} represents the message from variable i to variable j, saying what they think they are
            self.messages = {}
            for row, col in self.variables:
                for neighbor in self.neighbors(row, col):
                    assert self.initial_beliefs[row, col, :].sum() > 0
                    initial_msg = 1 - self.initial_beliefs[row, col, :].copy()
                    initial_msg /= initial_msg.sum()
                    self.messages[((row, col), neighbor)] = initial_msg

        @staticmethod
        def from_string(s: str) -> "SudokuBP":
            grid = decode_sudoku(s)
            return SudokuBP(grid)

        @staticmethod
        def from_filepath(filepath: str) -> "SudokuBP":
            with open(filepath, "r") as f:
                s = f.read()
            return SudokuBP.from_string(s)

        @staticmethod
        def neighbors(row: int, col: int) -> list[tuple[int, int]]:
            neighs = set()

            # Same row
            for c in range(SudokuBP.GRID_SIZE):
                if c != col:
                    neighs.add((row, c))
            # Same column
            for r in range(SudokuBP.GRID_SIZE):
                if r != row:
                    neighs.add((r, col))
            # Same box
            box_row_start = (row // 3) * 3
            box_col_start = (col // 3) * 3
            for r in range(box_row_start, box_row_start + 3):
                for c in range(box_col_start, box_col_start + 3):
                    if (r, c) != (row, col):
                        neighs.add((r, c))
            return list(neighs)

        @staticmethod
        def check_board(grid: np.ndarray) -> bool:
            for row in range(SudokuBP.GRID_SIZE):
                for col in range(SudokuBP.GRID_SIZE):
                    value = grid[row, col]
                    if value == 0:
                        raise ValueError("Board is not completely filled.")
                    for neigh_row, neigh_col in SudokuBP.neighbors(row, col):
                        if grid[neigh_row, neigh_col] == value:
                            return False
            return True

        def _belief_update(self):
            new_messages = {}

            for r, c in self.variables:
                neighbors = self.neighbors(r, c)

                # For each neighbor j, compute message from (r,c) to j
                for neighbor in neighbors:
                    # Step 1: Compute variable-to-factor message m_{i -> f_ij}, telling what it thinks it is
                    # ∆xi | f(xi, xj) = 1
                    var_to_factor = self.initial_beliefs[r, c, :].copy()
                    # print(f"Cell ({r}, {c}) initial belief: {var_to_factor}")
                    for other in neighbors:
                        if other != neighbor:
                            # Incoming message from other -> (r,c)
                            # print(
                            #     f"Cell ({r}, {c}) receiving message from {other}: {self.messages[(other, (r, c))]}"
                            # )
                            var_to_factor *= self.messages[
                                (other, (r, c))
                            ]  # update beliefs with what your neighbors are telling you

                    # Normalize to prevent numerical issues
                    if var_to_factor.sum() > 0:
                        var_to_factor /= var_to_factor.sum()
                    else:
                        raise ValueError(
                            f"Zero message encountered at cell ({r}, {c}) to neighbor {neighbor}."
                        )
                        # var_to_factor = np.ones(self.GRID_SIZE) / self.GRID_SIZE

                    # Step 2: Apply factor f_≠ using S-m trick
                    # Factor-to-Variable: what values you should avoid according to your neighbor
                    S = var_to_factor.sum()  # Should be 1 after normalization
                    factor_to_var = (
                        S - var_to_factor
                    )  # Subtract i's belief from what it thinks j should be

                    # Normalize outgoing message
                    if factor_to_var.sum() > 0:
                        factor_to_var /= factor_to_var.sum()
                    else:
                        raise ValueError(
                            f"Zero factor-to-variable message at cell ({r}, {c}) to neighbor {neighbor}."
                        )
                        # factor_to_var = np.ones(self.GRID_SIZE) / self.GRID_SIZE

                    # Step 3: Apply damping
                    old_msg = self.messages[((r, c), neighbor)]
                    new_msg = (
                        self.DAMPING * old_msg + (1 - self.DAMPING) * factor_to_var
                    )
                    new_msg /= new_msg.sum()  # Re-normalize after damping

                    new_messages[((r, c), neighbor)] = new_msg

            self.messages = new_messages

        def _most_likely_assignment(self) -> np.ndarray:
            assignment = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
            for r, c in self.variables:
                # Compute belief: unary factor * all incoming messages
                belief = self.initial_beliefs[r, c, :].copy()
                for neighbor in self.neighbors(r, c):
                    belief *= self.messages[(neighbor, (r, c))]
                assignment[r, c] = np.argmax(belief) + 1
            return assignment

        def run(self) -> np.ndarray:
            for iteration in range(self.MAX_ITERATIONS):
                self._belief_update()
                if self.check_board(self._most_likely_assignment()):
                    break
            assignment = self._most_likely_assignment()
            if not self.check_board(assignment):
                print(assignment)
                raise ValueError("No valid solution found.")
            print(f"Solution found {iteration} iters:")
            return assignment
    return (SudokuBP,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Interactive Solver Dashboard

    Use the controls below to select a puzzle and run different solvers. The notebook uses marimo's reactive execution - changing any input will automatically update dependent outputs.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Fancy BP
    """)
    return


@app.cell
def _(SudokuBP, decode_sudoku, np, time):
    import copy
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    class SudokuBPWithExtras(SudokuBP):
        """Extended BP solver with guided decimation, backtracking, visualization, and batch testing."""

        def __init__(self, grid: np.ndarray, verbose: bool = False):
            super().__init__(grid)
            self.verbose = verbose
            self.iteration_history = []

        @staticmethod
        def from_string(s: str, verbose: bool = False) -> "SudokuBPWithExtras":
            grid = decode_sudoku(s)
            return SudokuBPWithExtras(grid, verbose)

        def _get_belief(self, r: int, c: int) -> np.ndarray:
            """Compute the marginal belief for a cell."""
            belief = self.initial_beliefs[r, c, :].copy()
            for neighbor in self.neighbors(r, c):
                belief *= self.messages[(neighbor, (r, c))]
            if belief.sum() > 0:
                belief /= belief.sum()
            return belief

        def _get_entropy(self, belief: np.ndarray) -> float:
            """Compute entropy of a belief distribution."""
            belief = belief + 1e-10
            return -np.sum(belief * np.log(belief))

        def _get_confidence(self, belief: np.ndarray) -> float:
            """Get confidence score (max belief value)."""
            return np.max(belief)

        def _belief_update(self) -> bool:
            """Update messages using Sum-Product algorithm. Returns True if successful, False if conflict."""
            new_messages = {}

            for r, c in self.variables:
                neighbors = self.neighbors(r, c)
                for neighbor in neighbors:
                    # Step 1: Variable-to-Factor message (m_i->j)
                    # ∆xi | f(xi, xj) = 1
                    var_to_factor = self.initial_beliefs[r, c, :].copy()
                    for other in neighbors:
                        if other != neighbor:
                            # Incoming message from other -> (r,c)
                            var_to_factor *= self.messages[(other, (r, c))]

                    # Normalize to prevent numerical issues
                    if var_to_factor.sum() > 0:
                        var_to_factor /= var_to_factor.sum()
                    else:
                        # BP Conflict detected
                        return False

                    # Step 2: Apply factor f_≠ using S-m trick
                    # Factor-to-Variable: what values you should avoid according to your neighbor
                    S = var_to_factor.sum()  # Should be 1
                    factor_to_var = S - var_to_factor

                    # Normalize outgoing message
                    if factor_to_var.sum() > 0:
                        factor_to_var /= factor_to_var.sum()
                    else:
                        # BP Conflict detected
                        return False

                    # Step 3: Apply damping
                    old_msg = self.messages[((r, c), neighbor)]
                    new_msg = (
                        self.DAMPING * old_msg + (1 - self.DAMPING) * factor_to_var
                    )
                    if new_msg.sum() > 0:
                        new_msg /= new_msg.sum()
                    else:
                        return False

                    new_messages[((r, c), neighbor)] = new_msg

            self.messages = new_messages
            return True

        def _record_state(self):
            """Record current state for visualization."""
            beliefs = np.zeros((self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))
            for r, c in self.variables:
                beliefs[r, c, :] = self._get_belief(r, c)
            self.iteration_history.append(beliefs.copy())

        def run_with_tracking(
            self, max_iterations: int = None, reset_history: bool = True
        ) -> tuple[np.ndarray | None, int]:
            """Run BP and return (solution, iterations_used). Returns None if no solution."""
            if max_iterations is None:
                max_iterations = self.MAX_ITERATIONS

            if reset_history:
                self.iteration_history = []

            if not self.iteration_history:
                self._record_state()

            for iteration in range(max_iterations):
                if not self._belief_update():
                    if self.verbose:
                        print(f"BP Conflict detected at iter {iteration}")
                    return None, iteration + 1

                self._record_state()

                assignment = self._most_likely_assignment()
                if self.check_board(assignment):
                    if self.verbose:
                        print(f"Converged after {iteration + 1} iterations")
                    return assignment, iteration + 1

            assignment = self._most_likely_assignment()
            if self.check_board(assignment):
                return assignment, max_iterations
            return None, max_iterations

        def run_with_decimation(
            self, bp_iterations: int = 40, max_decimations: int = 81
        ) -> np.ndarray | None:
            """Run BP with guided decimation when BP alone doesn't converge."""
            self.iteration_history = []
            decimation_count = 0

            while decimation_count < max_decimations:
                result, iters = self.run_with_tracking(
                    bp_iterations, reset_history=False
                )
                if result is not None:
                    if self.verbose:
                        print(f"Solved with {decimation_count} decimations")
                    return result

                best_confidence = -1
                best_cell = None
                best_value = None

                for r, c in self.variables:
                    if self.grid[r, c] != 0:
                        continue

                    belief = self._get_belief(r, c)
                    confidence = self._get_confidence(belief)

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_cell = (r, c)
                        best_value = np.argmax(belief) + 1

                if best_cell is None:
                    break

                r, c = best_cell
                if self.verbose:
                    print(
                        f"Decimating cell ({r}, {c}) to {best_value} (confidence: {best_confidence:.3f})"
                    )

                self.grid[r, c] = best_value
                self.initial_beliefs[r, c, :] = 0.0
                self.initial_beliefs[r, c, best_value - 1] = 1.0

                for neighbor in self.neighbors(r, c):
                    belief = self.initial_beliefs[r, c, :] / self.initial_beliefs[r, c, :].sum()
                    initial_message = 1 - belief
                    initial_message /= initial_message.sum()
                    self.messages[((r, c), neighbor)] = initial_message

                self._record_state()
                decimation_count += 1

            return None

        def run_with_backtracking(
            self, bp_iterations: int = 40, max_depth: int = 10
        ) -> np.ndarray | None:
            """Run BP with decimation and backtracking on conflicts."""
            self.iteration_history = []
            return self._backtrack_solve(bp_iterations, max_depth, 0)

        def _backtrack_solve(
            self, bp_iterations: int, max_depth: int, depth: int
        ) -> np.ndarray | None:
            if depth > max_depth:
                return None

            saved_grid = self.grid.copy()
            saved_beliefs = self.initial_beliefs.copy()
            saved_messages = {k: v.copy() for k, v in self.messages.items()}

            result, _ = self.run_with_tracking(bp_iterations, reset_history=False)
            if result is not None:
                return result

            candidates = []
            for r, c in self.variables:
                if self.grid[r, c] != 0:
                    continue
                belief = self._get_belief(r, c)
                confidence = self._get_confidence(belief)
                candidates.append((confidence, r, c, belief))

            if not candidates:
                return None

            candidates.sort(reverse=True)
            _, r, c, belief = candidates[0]

            value_order = np.argsort(belief)[::-1]

            for value in value_order:
                value = int(value) + 1
                if belief[value - 1] < 0.01:
                    continue

                if self.verbose:
                    print(f"{'  ' * depth}Trying cell ({r}, {c}) = {value}")

                self.grid[r, c] = value
                self.initial_beliefs[r, c, :] = 0.0
                self.initial_beliefs[r, c, value - 1] = 1.0
                for neighbor in self.neighbors(r, c):
                    belief = self.initial_beliefs[r, c, :] / self.initial_beliefs[r, c, :].sum()
                    initial_message = 1 - belief
                    initial_message /= initial_message.sum()
                    self.messages[((r, c), neighbor)] = initial_message

                self._record_state()
                result = self._backtrack_solve(bp_iterations, max_depth, depth + 1)
                if result is not None:
                    return result

                self.grid = saved_grid.copy()
                self.initial_beliefs = saved_beliefs.copy()
                self.messages = {k: v.copy() for k, v in saved_messages.items()}

            return None

        def visualize_convergence(self, interval: int = 200):
            """Create a matplotlib animation showing belief convergence."""
            if not self.iteration_history:
                print("No iteration history. Run with tracking first.")
                return None

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

            def draw_frame(frame_idx):
                ax.clear()
                beliefs = self.iteration_history[frame_idx]

                ax.set_xlim(0, 9)
                ax.set_ylim(0, 9)
                ax.set_aspect("equal")
                ax.axis("off")
                ax.set_title(
                    f"BP Iteration {frame_idx}/{len(self.iteration_history) - 1}",
                    fontsize=14,
                )

                for i in range(10):
                    lw = 2.5 if i % 3 == 0 else 0.5
                    ax.axhline(i, color="black", linewidth=lw)
                    ax.axvline(i, color="black", linewidth=lw)

                for r in range(self.GRID_SIZE):
                    for c in range(self.GRID_SIZE):
                        x, y = c, 8 - r

                        belief = beliefs[r, c]
                        best_value = np.argmax(belief) + 1
                        confidence = np.max(belief)

                        if self.grid[r, c] != 0:
                            rect = patches.Rectangle(
                                (x, y), 1, 1, facecolor="lightgray", edgecolor="none"
                            )
                            ax.add_patch(rect)
                            ax.text(
                                x + 0.5,
                                y + 0.5,
                                str(self.grid[r, c]),
                                ha="center",
                                va="center",
                                fontsize=16,
                                fontweight="bold",
                            )
                        else:
                            color = plt.cm.RdYlGn(confidence)
                            rect = patches.Rectangle(
                                (x, y),
                                1,
                                1,
                                facecolor=color,
                                edgecolor="none",
                                alpha=0.6,
                            )
                            ax.add_patch(rect)

                            if confidence > 0.1:
                                fontsize = 10 + int(confidence * 6)
                                ax.text(
                                    x + 0.5,
                                    y + 0.5,
                                    str(best_value),
                                    ha="center",
                                    va="center",
                                    fontsize=fontsize,
                                    alpha=0.3 + 0.7 * confidence,
                                )

            anim = FuncAnimation(
                fig,
                draw_frame,
                frames=len(self.iteration_history),
                interval=interval,
                repeat=True,
            )
            plt.close(fig)
            return HTML(anim.to_jshtml())

        def plot_final_beliefs(self):
            """Plot the final belief distribution as a heatmap."""
            if not self.iteration_history:
                print("No iteration history. Run with tracking first.")
                return

            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            final_beliefs = self.iteration_history[-1]

            for digit in range(1, 10):
                ax = axes[(digit - 1) // 3, (digit - 1) % 3]
                belief_map = final_beliefs[:, :, digit - 1]

                im = ax.imshow(belief_map, cmap="YlOrRd", vmin=0, vmax=1)
                ax.set_title(f"Digit {digit}", fontsize=12)
                ax.set_xticks(range(9))
                ax.set_yticks(range(9))

                for i in [2.5, 5.5]:
                    ax.axhline(i, color="black", linewidth=2)
                    ax.axvline(i, color="black", linewidth=2)

            fig.colorbar(im, ax=axes, shrink=0.6, label="Belief")
            plt.suptitle("Final Belief Distribution by Digit", fontsize=14)
            plt.tight_layout()
            return fig

        def plot_entropy_over_time(self):
            """Plot average entropy over iterations."""
            if not self.iteration_history:
                print("No iteration history. Run with tracking first.")
                return

            entropies = []
            for beliefs in self.iteration_history:
                total_entropy = 0
                count = 0
                for r in range(self.GRID_SIZE):
                    for c in range(self.GRID_SIZE):
                        if self.grid[r, c] == 0:
                            total_entropy += self._get_entropy(beliefs[r, c])
                            count += 1
                entropies.append(total_entropy / max(count, 1))

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(entropies, "b-", linewidth=2)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Average Entropy")
            ax.set_title("Belief Entropy Over BP Iterations (lower = more confident)")
            ax.grid(True, alpha=0.3)
            return fig

        @staticmethod
        def run_benchmark(
            filepath: str,
            method: str = "bp",
            bp_iterations: int = 40,
            verbose: bool = False,
        ):
            """
            Run solver on all puzzles in a file and report statistics.

            Args:
                filepath: Path to file with puzzles (9 lines per puzzle)
                method: "bp", "decimation", or "backtrack"
                bp_iterations: Number of BP iterations per attempt
                verbose: Print progress
            """
            with open(filepath, "r") as f:
                content = f.read()

            lines = [l for l in content.strip().split("\n") if l.strip()]
            puzzles = []
            for i in range(0, len(lines), 9):
                if i + 9 <= len(lines):
                    puzzle_str = "\n".join(lines[i : i + 9])
                    puzzles.append(puzzle_str)

            print(f"Found {len(puzzles)} puzzles")
            print(f"Method: {method}, BP iterations: {bp_iterations}")
            print("-" * 50)

            solved = 0
            total_time = 0
            times = []

            for i, puzzle_str in enumerate(puzzles):
                solver = SudokuBPWithExtras.from_string(puzzle_str, verbose=False)

                start = time.time()

                if method == "bp":
                    result, _ = solver.run_with_tracking(bp_iterations)
                elif method == "decimation":
                    result = solver.run_with_decimation(bp_iterations)
                elif method == "backtrack":
                    result = solver.run_with_backtracking(bp_iterations)
                else:
                    raise ValueError(f"Unknown method: {method}")

                elapsed = time.time() - start
                times.append(elapsed)
                total_time += elapsed

                if result is not None:
                    solved += 1
                    status = "✓"
                else:
                    status = "✗"

                if verbose:
                    print(f"Puzzle {i + 1}: {status} ({elapsed:.3f}s)")

            print("-" * 50)
            print(
                f"Solved: {solved}/{len(puzzles)} ({100 * solved / len(puzzles):.1f}%)"
            )
            print(f"Total time: {total_time:.2f}s")
            print(f"Average time: {total_time / len(puzzles):.3f}s per puzzle")
            if times:
                print(f"Min/Max time: {min(times):.3f}s / {max(times):.3f}s")

            return solved, len(puzzles)
    return (SudokuBPWithExtras,)


@app.cell
def _(EXAMPLE_PUZZLES, mo):
    # --- 1. UI Inputs ---
    mo.md("### 🎮 Interactive Sudoku Solver (Single View)")

    # Puzzle Selection
    puzzle_dropdown = mo.ui.dropdown(
        options=list(EXAMPLE_PUZZLES.keys()),
        value="Easy (Test)",
        label="Select Puzzle",
    )
    use_custom = mo.ui.switch(label="Custom Input", value=False)
    puzzle_text = mo.ui.text_area(
        value=EXAMPLE_PUZZLES["Easy (Test)"],
        label="Custom Puzzle (0=blank)",
        rows=9,
    )

    # Solver Parameters
    solver_method = mo.ui.radio(
        options={
            "Pure BP": "bp",
            "With Decimation": "decimation",
            "With Backtracking": "backtrack",
        },
        value="Pure BP",
        label="Method",
        inline=True,
    )
    bp_iterations_slider = mo.ui.slider(
        start=10, stop=200, step=10, value=40, label="Max Iterations"
    )
    bp_damping_slider = mo.ui.slider(
        start=0.1, stop=0.9, step=0.1, value=0.5, label="BP Damping (α)"
    )
    return (
        bp_damping_slider,
        bp_iterations_slider,
        puzzle_dropdown,
        puzzle_text,
        solver_method,
        use_custom,
    )


@app.cell
def _(
    EXAMPLE_PUZZLES,
    SudokuBPWithExtras,
    bp_damping_slider,
    bp_iterations_slider,
    mo,
    puzzle_dropdown,
    puzzle_text,
    solver_method,
    time,
    use_custom,
):
    # --- 2. Execution Logic ---

    # Layout for Controls (re-displayed here for unity)
    controls_ui = mo.vstack(
        [
            mo.hstack([puzzle_dropdown, use_custom], justify="start", gap=2),
            puzzle_text if use_custom.value else mo.md(""),
            mo.md("---"),
            mo.hstack(
                [solver_method, bp_iterations_slider, bp_damping_slider],
                gap=2,
                align="center",
            ),
        ]
    )

    # Extract puzzle string
    selected_puzzle = (
        puzzle_text.value
        if use_custom.value
        else EXAMPLE_PUZZLES[puzzle_dropdown.value]
    )

    # Configure and Run Solver
    solver = SudokuBPWithExtras.from_string(selected_puzzle, verbose=False)
    solver.MAX_ITERATIONS = bp_iterations_slider.value
    solver.DAMPING = bp_damping_slider.value

    # Run Solver
    _start_t = time.time()
    result = None
    iters_used = 0

    if solver_method.value == "bp":
        result, iters_used = solver.run_with_tracking(bp_iterations_slider.value)
    elif solver_method.value == "decimation":
        result = solver.run_with_decimation(bp_iterations_slider.value)
        iters_used = len(solver.iteration_history)
    else:  # backtrack
        result = solver.run_with_backtracking(bp_iterations_slider.value)
        iters_used = len(solver.iteration_history)

    elapsed_ms = (time.time() - _start_t) * 1000

    # Results Message
    if result is not None:
        status = mo.callout(
            mo.md(f"**✅ Solved in {iters_used} steps ({elapsed_ms:.1f}ms)**"),
            kind="success",
        )
    else:
        status = mo.callout(
            mo.md(
                f"**❌ Failed to converge ({elapsed_ms:.1f}ms)** - Try increasing iterations or changing method."
            ),
            kind="warn",
        )

    solver_history = solver.iteration_history
    return controls_ui, selected_puzzle, solver_history, status


@app.cell
def _(mo, solver_history):
    # --- 3. History Slider ---
    # Defined in separate cell to depend on history length
    max_frame = max(0, len(solver_history) - 1)

    iter_view_slider = mo.ui.slider(
        start=0,
        stop=max_frame,
        step=1,
        value=max_frame,
        label=f"View Step (0-{max_frame})",
    )
    return iter_view_slider, max_frame


@app.cell
def _(
    controls_ui,
    decode_sudoku,
    iter_view_slider,
    max_frame,
    mo,
    np,
    selected_puzzle,
    solver_history,
    status,
):
    # --- 4. Visualization & Assembly ---

    # Grid Render Function
    def create_interactive_grid(
        beliefs: np.ndarray, original_grid: np.ndarray
    ) -> mo.Html:
        """Create clickable/hoverable grid."""
        # CSS for custom tooltip
        css = """
        <style>
        .sudoku-cell { position: relative; cursor: pointer; }
        .sudoku-tooltip {
            visibility: hidden;
            width: 140px;
            background-color: #222;
            color: #fff;
            text-align: left;
            border-radius: 4px;
            padding: 8px;
            position: absolute;
            z-index: 10;
            bottom: 120%;
            left: 50%;
            transform: translateX(-50%);
            white-space: pre-wrap;
            font-size: 11px;
            opacity: 0;
            transition: opacity 0.2s;
            pointer-events: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .sudoku-cell:hover .sudoku-tooltip {
            visibility: visible;
            opacity: 1;
        }
        /* Arrow */
        .sudoku-tooltip::after {
            content: " ";
            position: absolute;
            top: 100%; /* At the bottom of the tooltip */
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #222 transparent transparent transparent;
        }
        </style>
        """

        html_parts = [
            css,
            '<div style="font-family: monospace; display: inline-block;">',
        ]
        html_parts.append('<table style="border-collapse: collapse; font-size: 18px;">')

        for r in range(9):
            if r % 3 == 0 and r > 0:
                html_parts.append(
                    '<tr style="height: 3px;"><td colspan="11" style="background: #333;"></td></tr>'
                )

            html_parts.append("<tr>")
            for c in range(9):
                if c % 3 == 0 and c > 0:
                    html_parts.append('<td style="width: 3px; background: #333;"></td>')

                belief = beliefs[r, c]
                best_value = int(np.argmax(belief)) + 1
                confidence = float(np.max(belief))

                # Create tooltip content (no title attribute)
                dist_lines = [f"{v + 1}: {belief[v] * 100:.1f}%" for v in range(9)]
                tooltip_text = "\n".join(dist_lines)

                if original_grid[r, c] != 0:
                    # Given clue
                    cell_style = "background: #000000; font-weight: bold; color: white;"
                    cell_value = str(original_grid[r, c])
                else:
                    # Computed
                    r_color = int(255 * (1 - confidence))
                    g_color = int(200 * confidence)
                    cell_style = f"background: rgb({r_color}, {g_color}, 100); opacity: {0.4 + 0.6 * confidence};"
                    cell_value = str(best_value) if confidence > 0.1 else "?"

                # Wrap cell content in a div/span that handles the hover?
                # Or apply class to td. But TD cannot hold absolute easily if not block?
                # TD is table-cell. Relative positioning on TD works in most browsers.
                # Tooltip span must be inside TD.

                html_parts.append(
                    f'<td class="sudoku-cell" style="width: 40px; height: 40px; text-align: center; '
                    f'vertical-align: middle; border: 1px solid #999; {cell_style}">'
                    f"{cell_value}"
                    f'<span class="sudoku-tooltip">{tooltip_text}</span>'
                    f"</td>"
                )
            html_parts.append("</tr>")

        html_parts.append("</table>")
        html_parts.append(
            '<p style="font-size: 12px; color: #666; margin-top: 8px;">💡 Hover to see belief distribution</p>'
        )
        html_parts.append("</div>")
        return mo.Html("".join(html_parts))

    # Render Current Frame
    # Use slider value safely

    frame_idx = min(int(iter_view_slider.value), max_frame)
    current_beliefs = (
        solver_history[frame_idx] if solver_history else np.zeros((9, 9, 9))
    )
    original_grid = decode_sudoku(selected_puzzle)

    grid_viz = create_interactive_grid(current_beliefs, original_grid)

    # Final Assembly - The Single View
    mo.vstack([controls_ui, status, iter_view_slider, grid_viz])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Conventional SAT Solvers
    """)
    return


@app.cell
def _(TypeAlias, np):
    VARIABLE: TypeAlias = int  # variable index
    ASSIGNMENT: TypeAlias = bool | None
    LITERAL: TypeAlias = tuple[VARIABLE, bool]  # (variable index, is_positive)
    CLAUSE: TypeAlias = list[LITERAL]  # (variable index, is_positive)
    CNF: TypeAlias = list[CLAUSE]  # list of clauses
    BOARD: TypeAlias = np.ndarray  # 9x9 numpy array representing Sudoku board

    RANK = 3
    GRID_SIZE = RANK * RANK

    def get_var_idx(row: int, col: int, value: int) -> int:
        return row * GRID_SIZE**2 + col * GRID_SIZE + (value - 1)

    def get_var_from_idx(index: int) -> tuple[int, int, int]:
        row = index // (GRID_SIZE**2)
        col = (index % (GRID_SIZE**2)) // GRID_SIZE
        value = (index % GRID_SIZE) + 1
        return (row, col, value)

    def cnf_encode_sudoku(grid: BOARD) -> tuple[CNF, list[ASSIGNMENT]]:
        # At least one value per cell
        cell_constraints: CNF = [
            [(get_var_idx(row, col, value), True) for value in range(1, GRID_SIZE + 1)]
            for row in range(GRID_SIZE)
            for col in range(GRID_SIZE)
        ]

        # At most one value per cell (if v1 then not v2)
        unique_value_constraints: CNF = [
            [(get_var_idx(row, col, v1), False), (get_var_idx(row, col, v2), False)]
            for row in range(GRID_SIZE)
            for col in range(GRID_SIZE)
            for v1 in range(1, GRID_SIZE + 1)
            for v2 in range(v1 + 1, GRID_SIZE + 1)
        ]

        # Each value appears at least once in each row
        row_constraints: CNF = [
            [(get_var_idx(row, col, value), True) for col in range(GRID_SIZE)]
            for row in range(GRID_SIZE)
            for value in range(1, GRID_SIZE + 1)
        ]

        # Each value appears at most once in each row
        row_unique: CNF = [
            [(get_var_idx(row, c1, value), False), (get_var_idx(row, c2, value), False)]
            for row in range(GRID_SIZE)
            for value in range(1, GRID_SIZE + 1)
            for c1 in range(GRID_SIZE)
            for c2 in range(c1 + 1, GRID_SIZE)
        ]

        # Each value appears at least once in each column
        col_constraints: CNF = [
            [(get_var_idx(row, col, value), True) for row in range(GRID_SIZE)]
            for col in range(GRID_SIZE)
            for value in range(1, GRID_SIZE + 1)
        ]

        # Each value appears at most once in each column
        col_unique: CNF = [
            [(get_var_idx(r1, col, value), False), (get_var_idx(r2, col, value), False)]
            for col in range(GRID_SIZE)
            for value in range(1, GRID_SIZE + 1)
            for r1 in range(GRID_SIZE)
            for r2 in range(r1 + 1, GRID_SIZE)
        ]

        # Each value appears at least once in each box
        box_constraints: CNF = [
            [
                (get_var_idx(box_row * RANK + i, box_col * RANK + j, value), True)
                for i in range(RANK)
                for j in range(RANK)
            ]
            for box_row in range(RANK)
            for box_col in range(RANK)
            for value in range(1, GRID_SIZE + 1)
        ]

        # Each value appears at most once in each box
        box_unique: CNF = []
        for box_row in range(RANK):
            for box_col in range(RANK):
                for value in range(1, GRID_SIZE + 1):
                    cells = [
                        (box_row * RANK + i, box_col * RANK + j)
                        for i in range(RANK)
                        for j in range(RANK)
                    ]
                    for idx1, (r1, c1) in enumerate(cells):
                        for r2, c2 in cells[idx1 + 1 :]:
                            box_unique.append(
                                [
                                    (get_var_idx(r1, c1, value), False),
                                    (get_var_idx(r2, c2, value), False),
                                ]
                            )

        # Clues as unit clauses
        clue_clauses: CNF = []
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                value = grid[row, col]
                if value != 0:
                    clue_clauses.append([(get_var_idx(row, col, value), True)])

        clauses: CNF = (
            cell_constraints
            + unique_value_constraints
            + row_constraints
            + row_unique
            + col_constraints
            + col_unique
            + box_constraints
            + box_unique
            + clue_clauses
        )

        # Initialize assignments
        assignments: list[ASSIGNMENT] = [None] * (GRID_SIZE**3)
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                value = grid[row, col]
                if value != 0:
                    # Set this value to True
                    var_idx = get_var_idx(row, col, value)
                    assignments[var_idx] = True
                    # Set all other values for this cell to False
                    for v in range(1, GRID_SIZE + 1):
                        if v != value:
                            assignments[get_var_idx(row, col, v)] = False

        return clauses, assignments
    return (
        ASSIGNMENT,
        BOARD,
        CLAUSE,
        GRID_SIZE,
        LITERAL,
        VARIABLE,
        cnf_encode_sudoku,
        get_var_from_idx,
        get_var_idx,
    )


@app.cell
def _(
    ASSIGNMENT: "TypeAlias",
    BOARD: "TypeAlias",
    CLAUSE: "TypeAlias",
    GRID_SIZE,
    LITERAL: "TypeAlias",
    cnf_encode_sudoku,
    decode_sudoku,
    get_var_from_idx,
    np,
):
    class DPLLSolver:
        def __init__(self, grid: BOARD):
            self.grid = grid.copy()
            self.clauses, self.assignments = cnf_encode_sudoku(self.grid)
            self.history = []
            self.assignment_types = {}  # var_idx -> 'decision' | 'propagate'

        @staticmethod
        def from_string(s: str) -> "DPLLSolver":
            return DPLLSolver(decode_sudoku(s))

        def check_literal(self, literal: LITERAL) -> bool | None:
            var_idx, is_positive = literal
            assignment: ASSIGNMENT = self.assignments[var_idx]
            if assignment is None:
                return None
            return assignment if is_positive else not assignment

        def _record_state(
            self,
            action: str,
            focus_cell: tuple[int, int] | None = None,
            details: str = "",
            is_conflict: bool = False,
        ):
            """Record current state for visualization."""
            # Snapshot of assignments (only true ones for visualization)
            current_assignments = [
                i for i, v in enumerate(self.assignments) if v is True
            ]
            self.history.append(
                {
                    "assignments": current_assignments,
                    "types": self.assignment_types.copy(),
                    "action": action,
                    "focus": focus_cell,
                    "details": details,
                    "conflict": is_conflict,
                }
            )

        def _check_clause(self, clause: CLAUSE) -> bool | None:
            """Returns True if satisfied, False if conflict, None if undetermined."""
            has_unassigned = False
            for lit in clause:
                result = self.check_literal(lit)
                if result is True:
                    return True
                if result is None:
                    has_unassigned = True
            return None if has_unassigned else False

        def _unit_propagate(self) -> list[int] | None:
            """Returns list of assigned variable indices, or None if conflict."""
            propagated = []
            changed = True

            while changed:
                changed = False
                for clause in self.clauses:
                    unassigned_lits = []
                    satisfied = False

                    for lit in clause:
                        result = self.check_literal(lit)
                        if result is True:
                            satisfied = True
                            break
                        elif result is None:
                            unassigned_lits.append(lit)

                    if satisfied:
                        continue

                    if len(unassigned_lits) == 0:
                        self._undo(propagated)
                        return None  # Conflict

                    if len(unassigned_lits) == 1:
                        var_idx, is_positive = unassigned_lits[0]
                        if self.assignments[var_idx] is None:
                            self.assignments[var_idx] = is_positive
                            self.assignment_types[var_idx] = "propagate"
                            propagated.append(var_idx)
                            changed = True

            return propagated

        def _decide(self) -> int | None:
            """Pick an unassigned variable index."""
            for var_idx, assignment in enumerate(self.assignments):
                if assignment is None:
                    return var_idx
            return None

        def _is_satisfied(self) -> bool:
            """Check if all clauses are satisfied."""
            for clause in self.clauses:
                if self._check_clause(clause) is not True:
                    return False
            return True

        def _undo(self, var_indices: list[int]):
            """Undo assignments for the given variable indices."""
            for var_idx in var_indices:
                self.assignments[var_idx] = None
                if var_idx in self.assignment_types:
                    del self.assignment_types[var_idx]

        def _extract_board(self) -> BOARD:
            """Extract Sudoku board from assignments."""
            board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
            for var_idx, assigned in enumerate(self.assignments):
                if assigned is True:
                    row, col, value = get_var_from_idx(var_idx)
                    board[row, col] = value
            return board

        def run(self, max_steps: int = 5000, max_history: int = 500) -> BOARD | None:
            self.history = []
            self.assignment_types = {}

            # Since DPLL is recursive, we cheat by using a mutable step counter or class attribute
            self._step_count = 0
            self._max_steps = max_steps
            self._max_history = max_history

            if len(self.history) < self._max_history:
                self._record_state("start", details="Initial State")
            return self._solve()

        def _solve(self) -> BOARD | None:
            self._step_count += 1
            if self._step_count > self._max_steps:
                return None  # Timeout

            # Unit propagation
            propagated = self._unit_propagate()
            if propagated is None:
                if len(self.history) < self._max_history:
                    self._record_state(
                        "conflict",
                        is_conflict=True,
                        details="Conflict during propagation",
                    )
                return None  # Conflict detected

            # Record propagation if anything changed
            if propagated:
                if len(self.history) < self._max_history:
                    # Focus on the last propagated item for visual relevance
                    last_prop = propagated[-1]
                    r, c, _ = get_var_from_idx(last_prop)
                    self._record_state(
                        "propagate",
                        # focus_cell=(r, c),
                        details=f"Propagated {len(propagated)} vars",
                    )

            # Pick a decision variable
            var_idx = self._decide()
            if var_idx is None:
                # No more variables - final check for satisfaction
                if self._is_satisfied():
                    self._record_state("solved", details="Solved!")
                    return self._extract_board()
                return None  # No more variables but not satisfied

            # Try True first
            r, c, v = get_var_from_idx(var_idx)
            self._record_state(
                "decide", focus_cell=(r, c), details=f"Decide ({r}, {c}) = {v}"
            )

            self.assignments[var_idx] = True
            self.assignment_types[var_idx] = "decision"

            result = self._solve()
            if result is not None:
                return result

            # Backtrack: undo propagations from True branch, try False
            self._record_state(
                "backtrack", focus_cell=(r, c), details=f"Backtracking ({r}, {c})"
            )
            # self._undo(propagated)

            # Undo the True assignment
            self.assignments[var_idx] = None
            if var_idx in self.assignment_types:
                del self.assignment_types[var_idx]

            # Try False
            self.assignments[var_idx] = False
            self.assignment_types[var_idx] = "decision"  # Still a decision leg

            result = self._solve()
            if result is not None:
                return result

            # Both branches failed - undo everything and backtrack
            self._undo(propagated + [var_idx])
            return None
    return (DPLLSolver,)


@app.cell
def _(
    BOARD: "TypeAlias",
    CLAUSE: "TypeAlias",
    LITERAL: "TypeAlias",
    VARIABLE: "TypeAlias",
    cnf_encode_sudoku,
    decode_sudoku,
    get_var_from_idx,
    np,
):
    class CDCLSolver:
        RANK = 3
        GRID_SIZE = RANK * RANK

        def __init__(self, grid: BOARD):
            self.grid = grid.copy()
            self.clauses, self.assignments = cnf_encode_sudoku(self.grid)
            self.decision_level = 0
            self.levels = [0 if v is not None else -1 for v in self.assignments]
            self.antecedents: list[CLAUSE | None] = [None] * len(self.assignments)
            self.trail: list[VARIABLE] = []  # Variables in assignment order
            self.history = []

            # Track initial assignments in trail
            for var_idx, val in enumerate(self.assignments):
                if val is not None:
                    self.trail.append(var_idx)
                    if self.levels[var_idx] < 0:
                        self.levels[var_idx] = 0

        @staticmethod
        def from_string(s: str) -> "CDCLSolver":
            return CDCLSolver(decode_sudoku(s))

        def _record_state(
            self,
            action: str,
            focus_cell: tuple[int, int] | None = None,
            details: str = "",
            is_conflict: bool = False,
        ):
            """Record current state for visualization."""
            current_assignments = [
                i for i, v in enumerate(self.assignments) if v is True
            ]

            # Derive types
            types = {}
            for var_idx in current_assignments:
                if self.levels[var_idx] == 0:
                    types[var_idx] = "fixed"
                elif self.antecedents[var_idx] is None:
                    types[var_idx] = "decision"
                else:
                    types[var_idx] = "propagate"

            if len(self.history) >= getattr(self, "_max_history", 500):
                if not self.history or self.history[-1].get("action") != "skip":
                    self.history.append(
                        {
                            "action": "skip",
                            "assignments": [],
                            "details": "History Limit Reached",
                        }
                    )
                return

            self.history.append(
                {
                    "assignments": current_assignments,
                    "types": types,
                    "action": action,
                    "focus": focus_cell,
                    "details": details,
                    "conflict": is_conflict,
                }
            )

        def check_literal(self, literal: LITERAL) -> bool | None:
            var_idx, is_positive = literal
            assignment = self.assignments[var_idx]
            if assignment is None:
                return None
            return assignment if is_positive else not assignment

        def _assign(self, var: VARIABLE, value: bool, antecedent: CLAUSE | None = None):
            """Assign a variable and record metadata."""
            self.assignments[var] = value
            self.levels[var] = self.decision_level
            self.antecedents[var] = antecedent
            self.trail.append(var)

        def _unassign(self, var: VARIABLE):
            """Unassign a variable."""
            self.assignments[var] = None
            self.levels[var] = -1
            self.antecedents[var] = None

        def _propagate(self) -> CLAUSE | None:
            """Unit propagation. Returns conflict clause or None."""
            changed = True
            prop_count = 0
            while changed:
                changed = False
                for clause in self.clauses:
                    unassigned = []
                    satisfied = False

                    for lit in clause:
                        result = self.check_literal(lit)
                        if result is True:
                            satisfied = True
                            break
                        elif result is None:
                            unassigned.append(lit)

                    if satisfied:
                        continue

                    if len(unassigned) == 0:
                        return clause  # Conflict

                    if len(unassigned) == 1:
                        var_idx, is_positive = unassigned[0]
                        if self.assignments[var_idx] is None:
                            self._assign(var_idx, is_positive, clause)
                            changed = True
                            prop_count += 1
            return None

        def _analyze_conflict(self, conflict_clause: CLAUSE) -> tuple[CLAUSE, int]:
            """Analyze conflict and learn a clause using 1-UIP (Unique Implication Point) logic."""
            if self.decision_level == 0:
                return ([], -1)  # UNSAT at level 0

            # 1-UIP starts with the conflict clause as the current reason
            learned = set(conflict_clause)

            # Count how many literals in the current learned clause are from the current decision level
            def count_current_level_lits():
                return sum(
                    1 for lit in learned if self.levels[lit[0]] == self.decision_level
                )

            # Trace back through the trail at the current decision level
            trail_idx = len(self.trail) - 1

            while count_current_level_lits() > 1:
                # Find the most recently assigned variable in the trail that is in the learned clause
                while trail_idx >= 0:
                    var = self.trail[trail_idx]
                    # Check if negation of this assignment is in the learned clause
                    # (assigned True implies (var, False) is in the conflict reason)
                    assignment = self.assignments[var]
                    target_lit = (var, not assignment)

                    if (
                        target_lit in learned
                        and self.levels[var] == self.decision_level
                    ):
                        # Found the literal to resolve away
                        antecedent = self.antecedents[var]
                        if antecedent:
                            # Resolution: (learned \ {target_lit}) U (antecedent \ {(var, assignment)})
                            learned.remove(target_lit)
                            for ant_lit in antecedent:
                                if ant_lit[0] != var:
                                    learned.add(ant_lit)
                            break
                    trail_idx -= 1

                if trail_idx < 0:
                    break

            learned_list = list(learned)
            if not learned_list:
                return [], 0

            # Backtrack level is the maximum level of any literal other than the 1-UIP
            other_levels = [
                self.levels[lit[0]]
                for lit in learned_list
                if self.levels[lit[0]] < self.decision_level
            ]
            backtrack_level = max(other_levels) if other_levels else 0

            return learned_list, backtrack_level

        def _backtrack(self, level: int):
            """Backtrack to given decision level."""
            while self.trail:
                var = self.trail[-1]
                if self.levels[var] <= level:
                    break
                self.trail.pop()
                self._unassign(var)
            self.decision_level = level

        def _decide(self) -> VARIABLE | None:
            """Pick an unassigned variable."""
            for var_idx, assignment in enumerate(self.assignments):
                if assignment is None:
                    return var_idx
            return None

        def _is_satisfied(self) -> bool:
            """Check if all clauses are satisfied."""
            for clause in self.clauses:
                satisfied = False
                for lit in clause:
                    if self.check_literal(lit) is True:
                        satisfied = True
                        break
                if not satisfied:
                    return False
            return True

        def _extract_board(self) -> BOARD:
            """Extract Sudoku board from assignments."""
            board = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
            for var_idx, assigned in enumerate(self.assignments):
                if assigned is True:
                    row, col, value = get_var_from_idx(var_idx)
                    board[row, col] = value
            return board

        def run(self, max_steps: int = 10000, max_history: int = 500) -> BOARD | None:
            self.history = []
            self._max_history = max_history
            self._record_state("start", details="Initial State")
            step_count = 0

            # Initial propagation
            conflict = self._propagate()

            if self.trail:
                self._record_state("propagate", details="Initial Propagation")

            if conflict is not None:
                self._record_state(
                    "conflict", is_conflict=True, details="Conflict at level 0"
                )
                return None  # UNSAT at level 0

            while step_count < max_steps:
                step_count += 1

                # Make a decision
                var = self._decide()
                if var is None:
                    # No more variables - check if satisfied
                    if self._is_satisfied():
                        self._record_state("solved", details="Solved!")
                        return self._extract_board()
                    return None

                self.decision_level += 1
                self._assign(var, True)

                r, c, v = get_var_from_idx(var)
                self._record_state(
                    "decide", focus_cell=(r, c), details=f"Decide ({r}, {c}) = {v}"
                )

                # Propagate and handle conflicts
                conflict = self._propagate()

                if not conflict:
                    self._record_state("propagate", details="Propagated vars")

                while conflict is not None:
                    # Increment step count for each conflict to prevent hangs
                    step_count += 1
                    if step_count >= max_steps:
                        break

                    self._record_state(
                        "conflict", is_conflict=True, details=f"Conflict!"
                    )

                    # Analyze conflict and learn
                    learned_clause, backtrack_level = self._analyze_conflict(conflict)

                    if backtrack_level < 0:
                        self._record_state(
                            "conflict", is_conflict=True, details="UNSAT"
                        )
                        return None  # UNSAT

                    # Add learned clause - this will force a different assignment
                    if learned_clause:
                        self.clauses.append(learned_clause)

                    # Backtrack
                    self._backtrack(backtrack_level)

                    self._record_state(
                        "backtrack", details=f"Backtrack to L{backtrack_level}"
                    )

                    # The learned clause should now be a unit clause that propagates
                    conflict = self._propagate()

            # Max steps reached
            self._record_state(
                "conflict", is_conflict=True, details="Max Steps Reached"
            )
            return None
    return (CDCLSolver,)


@app.cell
def _(EXAMPLE_PUZZLES, mo):
    # SAT Solver Controls
    sat_puzzle_dropdown = mo.ui.dropdown(
        options=list(EXAMPLE_PUZZLES.keys()),
        value="Easy (Test)",
        label="Select Puzzle",
    )
    sat_use_custom = mo.ui.switch(label="Custom Input", value=False)
    sat_puzzle_text = mo.ui.text_area(
        value=EXAMPLE_PUZZLES["Easy (Test)"],
        label="Custom Puzzle (0=blank)",
        rows=9,
    )

    sat_solver_choice = mo.ui.radio(
        options={"DPLL": "dpll", "CDCL": "cdcl"}, value="DPLL", label="SAT Solver"
    )

    # Layout logic moved to dependent cell to avoid lifecycle error
    return (
        sat_puzzle_dropdown,
        sat_puzzle_text,
        sat_solver_choice,
        sat_use_custom,
    )


@app.cell
def _(
    CDCLSolver,
    DPLLSolver,
    EXAMPLE_PUZZLES,
    mo,
    sat_puzzle_dropdown,
    sat_puzzle_text,
    sat_solver_choice,
    sat_use_custom,
    time,
):
    # --- SAT Logic & UI Assembly ---

    # Build UI Layout (here, where values are accessible)
    sat_controls_ui = mo.vstack(
        [
            mo.md("## 🧮 SAT-Based Solvers"),
            mo.hstack([sat_puzzle_dropdown, sat_use_custom], justify="start", gap=2),
            sat_puzzle_text if sat_use_custom.value else mo.md(""),
            mo.md("---"),
            sat_solver_choice,
        ]
    )

    # Run SAT solver
    _sat_start = time.time()

    # Extract puzzle string
    sat_selected_puzzle = (
        sat_puzzle_text.value
        if sat_use_custom.value
        else EXAMPLE_PUZZLES[sat_puzzle_dropdown.value]
    )

    if sat_solver_choice.value == "dpll":
        sat_solver = DPLLSolver.from_string(sat_selected_puzzle)
    else:
        sat_solver = CDCLSolver.from_string(sat_selected_puzzle)

    sat_result = sat_solver.run(max_steps=20000)
    sat_elapsed = time.time() - _sat_start

    sat_history = sat_solver.history

    if sat_result is not None:
        sat_res_ui = mo.callout(
            mo.md(
                f"**✅ Solved with {sat_solver_choice.value.upper()} ({sat_elapsed * 1000:.1f}ms)**"
            ),
            kind="success",
        )
    else:
        sat_res_ui = mo.callout(
            mo.md(f"**❌ {sat_solver_choice.value.upper()} returned UNSAT or failed**"),
            kind="danger",
        )
    return sat_controls_ui, sat_history, sat_res_ui, sat_selected_puzzle


@app.cell
def _(mo, sat_history):
    # SAT History Slider
    sat_max_frame = max(0, len(sat_history) - 1)

    sat_slider = mo.ui.slider(
        start=0,
        stop=sat_max_frame,
        step=1,
        value=sat_max_frame,
        label=f"SAT Step (0-{sat_max_frame})",
    )
    return (sat_slider,)


@app.cell
def _(
    decode_sudoku,
    get_var_from_idx,
    get_var_idx,
    mo,
    np,
    sat_controls_ui,
    sat_history,
    sat_res_ui,
    sat_selected_puzzle,
    sat_slider,
    sat_solver_choice,
):
    # SAT Visualization
    def create_sat_grid(frame_data, original_grid):
        assignments = frame_data["assignments"]
        types = frame_data.get("types", {})
        focus = frame_data.get("focus")
        action = frame_data.get("action")
        is_conflict = frame_data.get("conflict", False)
        details = frame_data.get("details", "")

        # Reconstruct board from assignments
        board_vals = np.zeros((9, 9), dtype=int)
        for idx in assignments:
            r, c, v = get_var_from_idx(idx)
            board_vals[r, c] = v

        html_parts = ['<div style="font-family: monospace; display: inline-block;">']

        # Status Line
        status_color = "red" if is_conflict else "#333"
        html_parts.append(
            f'<div style="margin-bottom: 5px; color: {status_color}; font-weight: bold;">{details}</div>'
        )

        html_parts.append('<table style="border-collapse: collapse; font-size: 18px;">')

        for r in range(9):
            if r % 3 == 0 and r > 0:
                html_parts.append(
                    '<tr style="height: 3px;"><td colspan="11" style="background: #333;"></td></tr>'
                )

            html_parts.append("<tr>")
            for c in range(9):
                if c % 3 == 0 and c > 0:
                    html_parts.append('<td style="width: 3px; background: #333;"></td>')

                val = board_vals[r, c]
                cell_value = str(val) if val > 0 else ""

                # Determine Style
                bg_color = "white"
                text_color = "black"
                font_weight = "normal"
                border_color = "#999"
                border_width = "1px"

                if original_grid[r, c] != 0:
                    bg_color = "#000000"
                    text_color = "white"
                    font_weight = "bold"
                elif val > 0:
                    # It's an assigned variable
                    # Find var_idx
                    # Note: Sudoku values are 1-9. get_var_idx expects value.
                    v_idx = get_var_idx(r, c, val)
                    t = types.get(v_idx, "")

                    if t == "decision":
                        bg_color = "#add8e6"  # Light Blue
                        font_weight = "bold"
                        # User asked for 'forking variable' color -> Blue
                        # If this matches focus, maybe darker?
                    elif t == "propagate":
                        bg_color = "#e6f2ff"  # Very Light Blue

                # Highlight Focus
                if focus == (r, c):
                    border_color = "blue"
                    border_width = "3px"
                    if is_conflict:
                        border_color = "red"
                        bg_color = "#ffcccc"

                cell_style = (
                    f"background: {bg_color}; color: {text_color}; "
                    f"font-weight: {font_weight}; "
                    f"width: 40px; height: 40px; text-align: center; vertical-align: middle; "
                    f"border: {border_width} solid {border_color};"
                )

                html_parts.append(f'<td style="{cell_style}">{cell_value}</td>')
            html_parts.append("</tr>")

        html_parts.append("</table>")

        # Legend
        html_parts.append(
            '<div style="font-size: 12px; color: #666; margin-top: 8px;">'
            '<span style="background: #add8e6; padding: 0 4px;">Decision</span> '
            '<span style="background: #e6f2ff; padding: 0 4px;">Propagated</span> '
            '<span style="border: 2px solid red; padding: 0 4px;">Conflict</span>'
            "</div>"
        )
        html_parts.append("</div>")
        return mo.Html("".join(html_parts))

    # Render
    sat_frame_idx = (
        min(int(sat_slider.value), len(sat_history) - 1) if sat_history else 0
    )
    current_frame = (
        sat_history[sat_frame_idx] if sat_history else {"assignments": [], "types": {}}
    )

    sat_orig_grid = decode_sudoku(sat_selected_puzzle)
    sat_viz = create_sat_grid(current_frame, sat_orig_grid)

    mo.vstack(
        [
            sat_controls_ui,
            mo.md(f"### {sat_solver_choice.value.upper()} Visualization"),
            sat_res_ui,
            sat_slider,
            sat_viz,
        ]
    )
    return


if __name__ == "__main__":
    app.run()

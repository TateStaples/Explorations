# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.2",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "pylatexenc==2.10",
#     "pyzmq",
#     "qiskit==2.3.0",
#     "qiskit-aer==0.17.2",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import qiskit as qk
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.quantum_info import Statevector
    from qiskit.visualization import plot_histogram, plot_bloch_multivector
    import numpy as np
    import pylatexenc
    from matplotlib import pyplot as plt

    return (
        ClassicalRegister,
        QuantumCircuit,
        QuantumRegister,
        Statevector,
        mo,
        np,
        plot_bloch_multivector,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Duke University Phys 627/ECE 523 (Klco)
    ## Exercise Set 2026
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## I. Warm-Up and Preliminaries
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A. Tensor Products

    Nature appears to be constructed such that every quantum particle resides in its own linear vector space. Combining these linear vector spaces in order to describe many-particle quantum systems is achieved mathematically through the tensor product operation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.1
    1. If a state is separable or unentangled, its state may be succinctly written in tensor product form, $|\psi_1\rangle \otimes |\psi_2\rangle \otimes \cdots \otimes |\psi_n\rangle$. For a set of $n$ two-dimensional Hilbert spaces, how many real numbers are needed to specify such an unentangled wavefunction? For an entangled state that cannot be broken into such pieces, how many real numbers are needed to specify its wavefunction? How does this scaling change for $d$-level quantum degrees of freedom?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > N unentangled qubits

    Each qubit can be described by two real numbers ($\Reals^2$). This can be seen as $\theta \in [0, \pi)$ and $\phi \in [0, 2\pi)$ on the Bloch sphere, or we can derive in from the Hilbert space $\mathcal{H}_\text{qubit} = \mathbb{C}^2$ with 2 constraints: $|\braket{\psi}|^2 = 1$ and global phase invariance $\braket{\psi} = \exp(i\alpha)\braket{\psi}$. $\mathbb{C}^2$ is 4 dimensional, but the constraints reduce it to 2 dimensions.

    > N entangled qubits

    Entanglement is a quantum property where the state of one qubit cannot be described independently of the state of another qubit. This means we need 2 complex numbers ($\mathbb{C}^2$) to describe the state of N entangled qubits. In total, we need $2^N$ complex numbers or $2^{N+1}$ real numbers to describe the state of N entangled qubits. We still have the same constraints as above so we need $2^{N+1} - 2$ real numbers to describe the state of N entangled qubits.

    > N unentangled d-level quantum systems

    We have $d$ complex numbers to describe and 2 constraints so we need $2d-2$ real numbers per system. For $N$ systems, we need $N(2d-2)$ real numbers.

    > N entangled d-level quantum systems

    Now we have $d^N$ total states each with a complex number. We still have the same constraints as above so we need $2d^N - 2$ real numbers to describe the state of N entangled d-level quantum systems.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.2
    2. Declaring the basis in Dirac notation ($|000\rangle, \cdots, |111\rangle$), write the vector representation of $|\psi\rangle^{\otimes 3}$ for $|\psi\rangle = \frac{|0\rangle + \sqrt{2}|1\rangle}{\sqrt{3}}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $\ket{\psi} = \sqrt{\frac{1}{3}}\ket{0} + \sqrt{\frac{2}{3}}\ket{1}$
    """)
    return


@app.cell
def _(mo, np):
    _one = np.array([0, 1])
    _zero = np.array([1, 0])
    _psi = 1 / np.sqrt(3) * (_zero + np.sqrt(2) * _one)
    _res = np.kron(_psi, np.kron(_psi, _psi)).T
    assert sum(_res.conj() * _res).round(6) == 1.0
    mo.ui.matrix(_res, label="$\\psi \\otimes \\psi \\otimes \\psi$", disabled=True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    \begin{align*}
        \ket{000} = \sqrt{\frac{1}{3}}^3 \times \sqrt{\frac{2}{3}}^0 \\
        \ket{001} = \sqrt{\frac{1}{3}}^2 \times \sqrt{\frac{2}{3}}^1 \\
        \ket{010} = \sqrt{\frac{1}{3}}^2 \times \sqrt{\frac{2}{3}}^1 \\
        \ket{011} = \sqrt{\frac{1}{3}}^1 \times \sqrt{\frac{2}{3}}^2 \\
        \ket{100} = \sqrt{\frac{1}{3}}^2 \times \sqrt{\frac{2}{3}}^1 \\
        \ket{101} = \sqrt{\frac{1}{3}}^1 \times \sqrt{\frac{2}{3}}^2 \\
        \ket{110} = \sqrt{\frac{1}{3}}^1 \times \sqrt{\frac{2}{3}}^2 \\
        \ket{111} = \sqrt{\frac{1}{3}}^1 \times \sqrt{\frac{2}{3}}^3 \\
    \end{align*}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.3
    3. Algorithmically describe how to implement the tensor product operation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You multiply every element of the first tensor by the entire second tensor.

    Given two matrices $A$ of size $m \times n$ and $B$ of size $p \times q$, the tensor product $C = A \otimes B$ results in a matrix of size $mp \times nq$:

    $$
    A \otimes B = \begin{bmatrix}
    a_{11}B & \cdots & a_{1n}B \\
    \vdots & \ddots & \vdots \\
    a_{m1}B & \cdots & a_{mn}B
    \end{bmatrix}
    $$


    1.  **Coordinate Mapping**: For input tensors $A$ and $B$, an element in the resulting tensor $C$ at indices $(i, j, k, l, \dots)$ is the product of elements from $A$ and $B$ at their respective sub-indices.
    2.  **Algorithm (Pseudo-code)**:
        * Initialize an empty structure with dimensions equal to the product of the input dimensions.
        * Iterate through each element $a_{i}$ in $A$.
        * Iterate through each element $b_{j}$ in $B$.
        * Assign $C_{idx} = a_{i} \times b_{j}$, where $idx$ is the concatenated or flattened coordinate.


    ```python
    import numpy as np

    # Tensor product
    C = np.kron(A, B)
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Discuss the relationship between binary ordering of a Hilbert space basis and the tensor product operation. If two binary-ordered Hilbert spaces are expressed jointly, $H = H_1 \otimes H_2$, what is the natural ordering of the resulting $(d_1 \times d_2)$-dimensional Hilbert space?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The binary counting system. First 0 with 0-$d_1$, then 1 with 0-$d_2$, etc.

    This makes since because sense because any $n$ qubits quantum system is in a complex superposition of $2^n$ classical bits or a superposition of all $n$ long bitstrings. The tensor product structure allows for indexing into $H$ by concatentating the index into $H_1$ with $H_2$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Use this understanding to write (or generate) the matrix form of the unitary operators (extending the SWAP operator) that reorder the Hilbert spaces in the following ways:
    """)
    return


@app.cell
def _(mo, np):
    from scipy.special import perm
    def generate_d_string(n: int, d=2) -> list[str]:
        """Generates a string representation of the basis states for n subsystems of dimension d."""
        total_dim = d**n
        basis_states = []

        for i in range(total_dim):
            # Convert index to base-d representation
            state = []
            idx = i
            for _ in range(n):
                state.append(str(idx % d))
                idx //= d
            basis_states.append(''.join(reversed(state)))

        return basis_states

    def swap_matrix(target_order: list[int], d=2) -> mo.ui.matrix:
        n = len(target_order)
        total_dim = d**n

        # 1. Create a range of all possible input indices (column indices)
        input_indices = np.arange(total_dim)

        # 2. "Unravel" these indices into their individual subsystem components (basis states)
        input_coords = np.unravel_index(input_indices, shape=(d,) * n)

        # 3. Permute the coordinates according to the target order
        output_coords = tuple(input_coords[i] for i in target_order)

        # 4. "Ravel" the permuted coordinates back into flat output indices (row indices)
        output_indices = np.ravel_multi_index(output_coords, dims=(d,) * n)

        # 5. Construct the matrix
        # P[row, col] = 1 means input 'col' maps to output 'row'
        P = np.zeros((total_dim, total_dim), dtype=int)
        P[output_indices, input_indices] = 1

        basis_states = generate_d_string(n, d)
        permuted_basis_states = [basis_states[i] for i in output_indices]

        list_of_outer_products = [
            mo.md(rf"""$\ket{{{bs}}} \bra{{{pbs}}}$""")
            for bs, pbs in zip(basis_states, permuted_basis_states)
        ]

        display = mo.ui.matrix(
            value=P, 
            disabled=True, 
            label=f"Swap Matrix for order {target_order}, d={d}",
            row_labels=permuted_basis_states,
            column_labels=basis_states
        )



        return mo.vstack([
            display,
            mo.md("Outer Product Form:"),
            *list_of_outer_products
        ])

    return (swap_matrix,)


@app.cell
def _(mo):
    mo.md(r"""
    $H_1 \otimes H_2 \otimes H_3 \to H_3 \otimes H_2 \otimes H_1$ with $d_{1,2,3} = 2$
    """)
    return


@app.cell
def _(swap_matrix):
    swap_matrix([2, 1, 0])
    return


@app.cell
def _(mo):
    mo.md(r"""
    $H_1 \otimes H_2 \otimes H_3 \otimes H_4 \to H_2 \otimes H_3 \otimes H_1 \otimes H_4$ with $d_{1,2,3,4} = 2$
    """)
    return


@app.cell
def _(swap_matrix):
    swap_matrix(target_order=[1, 2, 0, 3], d=2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $H_1 \otimes H_2 \to H_2 \otimes H_1$ with $d_{1,2} = 3$
    """)
    return


@app.cell
def _(swap_matrix):
    swap_matrix(target_order=[1, 0], d=3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Express these unitary operators in matrix form and as a list of $\prod_i d_i$ non-zero matrix elements (coefficients of outer product representation).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. $
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.4
    4. By hand, write the matrix representations of $X \otimes Y$, $\mathbb{I}_2 \otimes Y$, $Y \otimes \mathbb{I}_2$, and $Y \otimes \mathbb{I}_4$, where the Pauli matrices are
    $$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$
       and $\mathbb{I}_k$ is the $k$-dimensional Identity matrix. Implement the tensor product operator in code to produce a picture of the matrix $Z \otimes X \otimes Y \otimes Z \otimes Y + X \otimes I_2 \otimes Y \otimes Z \otimes Y$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$X \otimes Y = \begin{bmatrix}0&1\\1&0\end{bmatrix}\otimes \begin{bmatrix}0&-i\\i&0\end{bmatrix} = \begin{bmatrix} 0 & 0 & 0 & -i \\ 0 & 0 & i & 0 \\ 0 & -i & 0 & 0 \\ i & 0 & 0 & 0 \end{bmatrix}$$

    $$I_2 \otimes Y = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \otimes \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} = \begin{bmatrix} 0 & -i & 0 & 0 \\ i & 0 & 0 & 0 \\ 0 & 0 & 0 & -i \\ 0 & 0 & i & 0 \end{bmatrix}$$

    $$Y \otimes I_2 = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} \otimes \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & 0 & -i & 0 \\ 0 & 0 & 0 & -i \\ i & 0 & 0 & 0 \\ 0 & i & 0 & 0 \end{bmatrix}$$

    $$Y \otimes I_4 = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} \otimes \begin{bmatrix} 1&0&0&0 \\ 0&1&0&0 \\ 0&0&1&0 \\ 0&0&0&1 \end{bmatrix} = \begin{bmatrix} 0 & 0 & 0 & 0 & -i & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & -i & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & -i & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & -i \\ i & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & i & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & i & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & i & 0 & 0 & 0 & 0 \end{bmatrix}$$
    """)
    return


@app.cell
def _(plt):
    from qiskit.quantum_info import SparsePauliOp

    X = SparsePauliOp("X")
    Y = SparsePauliOp("Y")
    Z = SparsePauliOp("Z")
    I = SparsePauliOp("I")

    operator = (Z ^ X ^ Y ^ Z ^ Y) + (X ^ I ^ Y ^ Z ^ Y)
    # TODO: add a legend
    plt.imshow(operator.to_matrix().real, cmap="binary")
    plt.show()
    return (SparsePauliOp,)


@app.cell
def _(mo):
    mo.md(r"""
    #### A.5
    5. Due to the rapid growth in dimensionality of Hilbert spaces connected through the tensor product operation, structural computational design can have significant impact on computational resources. In some cases, the generation or manipulation of entanglement provides opportunities and guidance to improve such designs.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Show that, for the application of a tensor product operator (e.g., non-entangling time evolution), $A \otimes B$ on a state vector, $v$, one may employ the following identity
    $$[A \otimes B] v = \text{vec} [B V A^T]$$
       where $V$ is a matrix form of the vector $v$ that is reversed by the operation $\text{vec}[\cdot]$. (May be helpful to begin with a separable $v$ then employ the linearity of quantum mechanics)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let the state vector $v$ belong to the composite Hilbert space $\mathcal{H}_A \otimes \mathcal{H}_B$. We define the mapping between the vector $v$ and its matrix representation $V$ using the standard column-stacking convention.

    If the system is in a **separable state** $v = x \otimes y$ (where $x \in \mathcal{H}_A$ and $y \in \mathcal{H}_B$), the vector is:
    $$v = x \otimes y = \begin{pmatrix} x_1 y \\ x_2 y \\ \vdots \end{pmatrix}$$

    This vector corresponds to stacking the columns of the matrix $V$, defined as the outer product of the components:
    $$V = y x^T$$
    *(Note: If $x$ is dimension $d_A$ and $y$ is dimension $d_B$, then $V$ is a $d_B \times d_A$ matrix.)*


    Apply the operator $(A \otimes B)$ to the separable state $v = x \otimes y$:
    $$v' = (A \otimes B)(x \otimes y)$$

    By the definition of the Kronecker product on vectors, this applies $A$ to the first subsystem and $B$ to the second:
    $$v' = (Ax) \otimes (By)$$

    Let us define the transformed components as $x' = Ax$ and $y' = By$. We now construct the matrix form $V'$ for this new vector $v'$. Using the definition established in Step 1 ($V = yx^T$):
    $$V' = y' (x')^T$$

    Substitute the definitions of $x'$ and $y'$:
    $$V' = (By) (Ax)^T$$

    Apply the transpose identity $(CD)^T = D^T C^T$:
    $$V' = (By) (x^T A^T)$$

    By the associativity of matrix multiplication, we regroup the terms:
    $$V' = B (y x^T) A^T$$

    Since $V = y x^T$, we substitute $V$ back into the equation:
    $$V' = B V A^T$$

    Finally, taking the vectorization of both sides recovers the vector state:
    $$v' = \text{vec}(V') = \text{vec}(B V A^T)$$

    Any arbitrary quantum state $v$ can be written as a linear combination of separable states (e.g., via Singular Value Decomposition or Schmidt Decomposition):
    $$v = \sum_k c_k (x_k \otimes y_k)$$

    By linearity, the matrix form $V$ is the sum of the individual outer products:
    $$V = \sum_k c_k (y_k x_k^T)$$

    Now, apply the operator $(A \otimes B)$ to the general sum:

    $$
    \begin{aligned}
    (A \otimes B) v &= (A \otimes B) \sum_k c_k (x_k \otimes y_k) \\
    &= \sum_k c_k \left[ (A \otimes B) (x_k \otimes y_k) \right] & \text{(Linearity of Operator)} \\
    &= \sum_k c_k \text{vec}\left( B (y_k x_k^T) A^T \right) & \text{(Result from Step 2)} \\
    &= \text{vec}\left( B \left[ \sum_k c_k y_k x_k^T \right] A^T \right) & \text{(Linearity of Matrix Mult. \& vec)} \\
    &= \text{vec}(B V A^T)
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - For the case of $\text{dim}[A] = \text{dim}[B] = 10^3$ (i.e., A and B are $10^3 \times 10^3$ dimensional matrices), estimate the percent reduction in the number of floating point operations used to act the tensor-product operator on the state vector when employing this identity.
    """)
    return


@app.cell
def _():
    def calculate_flops(n):
        flops_direct = 2 * (n**2) ** 2
        flops_identity = 2 * n**3 + 2 * n**3
        return flops_direct, flops_identity

    n = 1000
    direct, identity = calculate_flops(n)
    reduction = (direct - identity) / direct * 100
    ratio = direct / identity
    print(f"Direct FLOPs: {direct:.2e}")
    print(f"Identity FLOPs: {identity:.2e}")
    print(f"Reduction: {reduction:.4f}%")
    print(f"Speedup Factor: {ratio:.2f}x")
    return


@app.cell
def _(mo):
    mo.md(r"""
    We compare the cost of applying the operator using the standard direct method versus the vectorization identity.

    **Parameters:**
    * Dimension of subsystem A: $n = 10^3$
    * Dimension of subsystem B: $n = 10^3$
    * Total Hilbert space dimension: $N = n^2 = 10^6$

    **1. Direct Method ($[A \otimes B] v$)**
    This approach involves constructing the explicit Kronecker product matrix and multiplying it by the vector $v$.

    * **Matrix Size:** The matrix $M = A \otimes B$ has dimensions $N \times N$ ($10^6 \times 10^6$).
    * **Operation:** Dense Matrix-Vector Multiplication.
    * **FLOP Count:** $\approx 2N^2 = 2(n^2)^2 = 2n^4$
    * **Calculation:**
        $$2 \times (10^6)^2 = 2 \times 10^{12} \text{ FLOPs (2 Trillion)}$$
    * **Memory Note:** Storing this matrix in double precision (8 bytes) would require $\approx 8$ Terabytes of RAM, rendering it infeasible on most hardware.

    **2. Identity Method ($\text{vec}[B V A^T]$)**
    This approach keeps the operators separated and applies them sequentially to the reshaped state matrix $V$ ($n \times n$).

    * **Step 1:** Compute $W = V A^T$. This is an $(n \times n) \times (n \times n)$ matrix multiplication.
        * Cost: $2n^3$ FLOPs.
    * **Step 2:** Compute $Z = B W$. This is another $(n \times n) \times (n \times n)$ matrix multiplication.
        * Cost: $2n^3$ FLOPs.
    * **Total FLOP Count:** $\approx 4n^3$
    * **Calculation:**
        $$4 \times (10^3)^3 = 4 \times 10^9 \text{ FLOPs (4 Billion)}$$


    **Speedup Factor:**
    $$\frac{\text{Direct Cost}}{\text{Identity Cost}} = \frac{2n^4}{4n^3} = \frac{n}{2} = \frac{1000}{2} = \mathbf{500\times}$$

    **Percent Reduction:**
    $$\text{Reduction} = \left( 1 - \frac{4 \times 10^9}{2 \times 10^{12}} \right) \times 100 = \left( 1 - \frac{1}{500} \right) \times 100 = \mathbf{99.8\%}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### B. Gram-Schmidt Orthogonalization

    The Gram-Schmidt procedure produces an orthonormal basis, $\{|v_k\rangle\}$, in a $d$-dimensional vector space from an arbitrary spanning basis, $\{|w_k\rangle\}$. Upon choosing a vector in the basis as the initial seed, $|v_1\rangle = \frac{|w_1\rangle}{\sqrt{\langle w_1 | w_1 \rangle}}$, remaining vectors are sequentially orthogonalized from the collection as
    $$|v'_k\rangle \equiv |w_k\rangle - \sum_{j=1}^{k-1} \langle v_j | w_k \rangle |v_j\rangle, \quad |v_k\rangle = \frac{|v'_k\rangle}{\sqrt{\langle v'_k | v'_k \rangle}}$$
    for $2 \leq k \leq d$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.1
    1. Show that the Gram-Schmidt process produces an orthonormal basis for the vector space.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Orthonormal basis is defined as a set of vectors that are mutually orthogonal, have unit norm, and span the entire vector space. The process of proving that the Gram-Schmidt process produces an orthonormal basis is based on showing that each step in the process maintains these properties.

    1. All vectors are normalized
    2. All vectors are orthogonal
    3. For any iteration $i$, the vectors $v_1, v_2, ..., v_i$ span an $i$ dimensional subspace

    **Initial Step**
    1. $v_1$ is normalized by dividing by its norm
    2. There are no vectors to be orthogonal to
    3. $v_1$ spans a 1 dimensional subspace

    **Inductive Step**
    1. $\ket{v_i}$ is normalized by dividing by its norm
    2. $\ket{v_i'}$ is orthogonal to all previous vectors $\ket{v_1'}, ..., \ket{v_{i-1}'}$ by removing the components of the previous vectors from $\ket{v_i}$ (and $\ket{v_i}$ is a scaled version of $\ket{v_i'}$)
    3. $\ket{v_1'}, ..., \ket{v_i'}$ span an $i$ dimensional subspace because $\ket{v_1'}, ..., \ket{v_{i-1}'}$ span an $i-1$ dimensional subspace and $\ket{v_i'}$ is not in that subspace (because it is orthogonal to all of them) & in a new dimension (since $\ket{u_1}, ..., \ket{u_i}$ are linearly independent)

    **Conclusion**
    Since all three properties are maintained throughout the process, the Gram-Schmidt process produces an orthonormal basis.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.2
    2. Determine (recommend teaching a computer how to calculate) the orthonormal basis identified through GS orthogonalization of the following $\{|w_k\rangle\}$ sets:
       - $\{|\psi_1\rangle, |\psi_2\rangle, |\psi_3\rangle, |\psi_4\rangle\}$
       - $\{|\psi_4\rangle, |\psi_1\rangle, |\psi_2\rangle, |\psi_3\rangle\}$
       - $\{|\psi_3\rangle, |\psi_4\rangle, |\psi_1\rangle, |\psi_2\rangle\}$
       - $\{|\psi_2\rangle, |\psi_3\rangle, |\psi_4\rangle, |\psi_1\rangle\}$
       with states
       $$|\psi_1\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\0\\0\\-i\end{pmatrix}, \quad |\psi_2\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\0\\0\\1\end{pmatrix}, \quad |\psi_3\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix}0\\1\\-1\\0\end{pmatrix}, \quad |\psi_4\rangle = \frac{1}{2}\begin{pmatrix}1\\1\\1\\1\end{pmatrix}$$
    """)
    return


@app.cell
def _(np):
    _psi_1 = 1 / np.sqrt(2) * np.array([1, 0, 0, -1j])
    _psi_2 = 1 / np.sqrt(2) * np.array([1, 0, 0, 1])
    _psi_3 = 1 / np.sqrt(2) * np.array([0, 1, -1, 0])
    _psi_4 = 1 / np.sqrt(2) * np.array([0, 1, 1, 0])

    def gram_schmidt(vectors: list[np.ndarray]) -> list[np.ndarray]:
        new_basis = []
        for vector in vectors:
            for basis_vector in new_basis:
                vector = vector - np.vdot(basis_vector, vector) * basis_vector
            new_basis.append(vector / np.linalg.norm(vector))
        return new_basis

    # 1234, 4123, 3412, 2341
    orderings = [
        (_psi_1, _psi_2, _psi_3, _psi_4),
        (_psi_4, _psi_1, _psi_2, _psi_3),
        (_psi_3, _psi_4, _psi_1, _psi_2),
        (_psi_2, _psi_3, _psi_4, _psi_1),
    ]

    for i, ordering in enumerate(orderings):
        new_basis = gram_schmidt(ordering)
        print(f"Ordering {i + 1}:")
        for j, vector in enumerate(new_basis):
            print(f"  {j + 1}: {list(vector)}")
        print()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### C. Hermitian and Unitary Matrices
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.1
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    1. Define the properties of Hermitian and unitary matrices. Discuss which properties make Hermitian and unitary matrices logical candidates for representing physical properties of energy spectra (as Hamiltonian) and propagation (as time evolution operators), respectively.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Hermitian Matrices ($H$)**
    * **Definition:** $H = H^\dagger$
    * **Properties:** * Real eigenvalues ($\lambda \in \mathbb{R}$).
        * Orthogonal eigenvectors (complete basis).
    * **Physical Role (Hamiltonian):**
        * **Reality:** Measurements must be real; $H$ guarantees a real energy spectrum.
        * **Basis:** Allows any state to be represented as a superposition of energy eigenstates.


    **Unitary Matrices ($U$)**
    * **Definition:** $U^\dagger U = I$
    * **Properties:**
        * Preserves inner products and vector norms.
        * Eigenvalues have modulus 1 ($|\lambda| = 1$).
    * **Physical Role (Evolution):**
        * **Probability:** Norm preservation ensures the total probability remains exactly 1 over time.
        * **Reversibility:** Unitary operators are always invertible, maintaining deterministic evolution.


    **The Link**
    $U(t) = e^{-iHt/\hbar}$
    * **Hermitian $H$** (real energy) $\rightarrow$ **Unitary $U$** (conserved probability).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""

    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.2
    2. Solve the Schrödinger equation for a time-independent Hamiltonian to determine the relation between the Hamiltonian and the time evolution operator.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The Schrödinger equation is:
    $$i\hbar \frac{d}{dt}|\psi(t)\rangle = H |\psi(t)\rangle$$

    For a time-independent Hamiltonian $H$:
    $$\frac{d}{dt}|\psi(t)\rangle = -\frac{i}{\hbar} H |\psi(t)\rangle$$

    This is a first-order linear differential equation of the form $\dot{y} = Ay$. The solution is:
    $$|\psi(t)\rangle = e^{-iHt/\hbar} |\psi(0)\rangle$$

    The time evolution operator $U(t)$ matches the state at $t=0$ to $t$:
    $$|\psi(t)\rangle = U(t) |\psi(0)\rangle$$

    Thus:
    $$U(t) = e^{-iHt/\hbar}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.3
    3. Show that: for every Hermitian matrix, $A$, the exponential $e^{iA}$ is unitary. (The series expansion may be useful)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let $U = e^{iA}$. We want to show $U^\dagger U = I$.

    Since $A$ is Hermitian, $A^\dagger = A$.
    $$U^\dagger = (e^{iA})^\dagger = e^{-iA^\dagger} = e^{-iA}$$

    Note: $(e^M)^\dagger = e^{M^\dagger}$.

    Now computing the product (since $A$ commutes with itself, $e^X e^Y = e^{X+Y}$):
    $$U^\dagger U = e^{-iA} e^{iA} = e^{-iA + iA} = e^0 = I$$

    Thus, $e^{iA}$ is unitary.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.4
    4. Show that: every unitary matrix can be written as the imaginary exponential of a Hermitian matrix, $e^{iH}$, and can thus be considered as the time evolution of some Hamiltonian. (Consider their spectral decompositions)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Any unitary matrix $U$ can be diagonalized by a unitary transformation $V$:
    $$U = V D V^\dagger$$
    where $D$ is a diagonal matrix containing the eigenvalues of $U$.
    Since $U$ is unitary, its eigenvalues lie on the unit circle in the complex plane. Thus, we can write the diagonal elements as:
    $$D_{jj} = e^{i\lambda_j}$$
    where $\lambda_j \in \mathbb{R}$.

    We can define a diagonal matrix $\Lambda$ such that $\Lambda_{jj} = \lambda_j$. Then $D = e^{i\Lambda}$.

    Now, substitute back:
    $$U = V e^{i\Lambda} V^\dagger = e^{i V \Lambda V^\dagger}$$

    Let $H = V \Lambda V^\dagger$.
    Since $\Lambda$ is real and diagonal, $\Lambda^\dagger = \Lambda$.
    $$H^\dagger = (V \Lambda V^\dagger)^\dagger = (V^\dagger)^\dagger \Lambda^\dagger V^\dagger = V \Lambda V^\dagger = H$$
    Thus, $H$ is Hermitian.

    Therefore, $U = e^{iH}$ for some Hermitian $H$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.5
    5. **Ehrenfest’s theorem**: Show that the commutation of a time-independent operator with the Hamiltonian provides a necessary and sufficient condition for its expectation value to be constant in time.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let $A$ be a time-independent operator ($\partial A / \partial t = 0$). The expectation value is $\langle A \rangle = \langle \psi(t) | A | \psi(t) \rangle$.

    Taking the time derivative:
    $$\frac{d}{dt} \langle A \rangle = \frac{d}{dt} \langle \psi | A | \psi \rangle = \langle \dot{\psi} | A | \psi \rangle + \langle \psi | A | \dot{\psi} \rangle$$

    Using Schrödinger eq: $|\dot{\psi}\rangle = \frac{-i}{\hbar} H |\psi\rangle$ and $\langle \dot{\psi} | = \frac{i}{\hbar} \langle \psi | H^\dagger = \frac{i}{\hbar} \langle \psi | H$.

    $$\frac{d}{dt} \langle A \rangle = \left( \frac{i}{\hbar} \langle \psi | H \right) A | \psi \rangle + \langle \psi | A \left( \frac{-i}{\hbar} H | \psi \rangle \right)$$
    $$= \frac{i}{\hbar} \langle \psi | (HA - AH) | \psi \rangle$$
    $$= \frac{i}{\hbar} \langle [H, A] \rangle$$

    For $\langle A \rangle$ to be constant in time for *any* state $|\psi\rangle$, we require $\frac{d}{dt} \langle A \rangle = 0$, which implies $\langle [H, A] \rangle = 0$. Since this must hold for any state, the operator itself must be zero:
    $$[H, A] = 0$$

    Thus, commutation with $H$ is necessary and sufficient.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.6
    6. Discuss when a unitary transformation of time evolution can be encoded as a unitary transformation of the associated Hamiltonian, $U e^{iH} U^\dagger = e^{i U H U^\dagger}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The identity $U e^A U^\dagger = e^{U A U^\dagger}$ holds for **all** matrix exponentials and unitary $U$.

    Proof using series expansion:
    $$U e^A U^\dagger = U \left( \sum_{k=0}^\infty \frac{A^k}{k!} \right) U^\dagger = \sum_{k=0}^\infty \frac{U A^k U^\dagger}{k!}$$

    Note that $U A^k U^\dagger = (U A U^\dagger)(U A U^\dagger)...(U A U^\dagger) = (U A U^\dagger)^k$.

    $$= \sum_{k=0}^\infty \frac{(U A U^\dagger)^k}{k!} = e^{U A U^\dagger}$$

    So this encoding is **always** possible. This means simply that changing the basis of the system rotates the Hamiltonian, and the time evolution operator rotates accordingly in the same way.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.7
    7. Given the Schrödinger equation ($\hbar = 1$) with time-independent Hamiltonian,
       $$i \frac{\partial}{\partial t} |\psi\rangle = \hat{H} |\psi\rangle$$
       find the equation of motion (time derivative) for the time-evolution of the density matrix,
       $$\rho(t) = \sum_{j=1}^n p_j |\psi_j(t)\rangle \langle \psi_j(t)|$$
       Does the density matrix evolve like a Heisenberg picture operator?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$\dot{\rho}(t) = \sum_j p_j \frac{d}{dt} \left( |\psi_j\rangle \langle \psi_j| \right)$$
    $$= \sum_j p_j \left( |\dot{\psi}_j\rangle \langle \psi_j| + |\psi_j\rangle \langle \dot{\psi}_j| \right)$$

    Substitute $|\dot{\psi}\rangle = -i H |\psi\rangle$ and $\langle \dot{\psi}| = i \langle \psi | H$:

    $$= \sum_j p_j \left( (-i H |\psi_j\rangle) \langle \psi_j| + |\psi_j\rangle (i \langle \psi_j | H) \right)$$
    $$= -i H \left( \sum_j p_j |\psi_j\rangle \langle \psi_j| \right) + i \left( \sum_j p_j |\psi_j\rangle \langle \psi_j| \right) H$$
    $$= -i H \rho + i \rho H$$
    $$= -i [H, \rho]$$

    So the equation is:
    $$\frac{\partial \rho}{\partial t} = -i [H, \rho]$$
    (or with $\hbar$: $i \hbar \dot{\rho} = [H, \rho]$).

    **Comparison to Heisenberg Picture:**
    A Heisenberg picture operator $A_H(t)$ evolves as:
    $$\frac{d A_H}{dt} = \frac{i}{\hbar} [H, A_H]$$
    (Note the sign difference: $+i$ vs $-i$).

    So **no**, the density matrix (which is a formulation of the Schrödinger picture state) evolves with the opposite sign commutator compared to Heisenberg operators. It evolves more like a "co-state" or the dual to the operators.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### D. Pauli Matrices

    Operators for spin-1/2 quantum degrees of freedom can be represented as $2 \times 2$ matrices.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.1
    1. Pauli matrices are matrix representations of Pauli operators with respect to the "Z"-basis, $|0\rangle$ and $|1\rangle$,
       $$(\sigma_0, \vec{\sigma}) = (\sigma_0, \sigma_1, \sigma_2, \sigma_3) = \left( \begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}, \begin{pmatrix}0 & 1 \\ 1 & 0\end{pmatrix}, \begin{pmatrix}0 & -i \\ i & 0\end{pmatrix}, \begin{pmatrix}1 & 0 \\ 0 & -1\end{pmatrix} \right)$$
       Express each of these Pauli matrices in outer product notation. How does each Pauli matrix transform the two orthonormal basis states, $|0\rangle$ and $|1\rangle$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Outer Product Notation:**
    *   $I = \sigma_0 = |0\rangle\langle0| + |1\rangle\langle1|$
    *   $X = \sigma_1 = |0\rangle\langle1| + |1\rangle\langle0|$
    *   $Y = \sigma_2 = -i|0\rangle\langle1| + i|1\rangle\langle0|$
    *   $Z = \sigma_3 = |0\rangle\langle0| - |1\rangle\langle1|$

    **Transformations:**
    *   **I**: $I|0\rangle = |0\rangle$, $I|1\rangle = |1\rangle$ (Identity)
    *   **X**: $X|0\rangle = |1\rangle$, $X|1\rangle = |0\rangle$ (Bit flip)
    *   **Y**: $Y|0\rangle = i|1\rangle$, $Y|1\rangle = -i|0\rangle$ (Bit flip + Phase)
    *   **Z**: $Z|0\rangle = |0\rangle$, $Z|1\rangle = -|1\rangle$ (Phase flip)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.2
    2. Using the Kronecker delta and Levi-Civita epsilon in index notation, determine succinct expressions for the commutators, $[\sigma_i, \sigma_j]$, the anticommutators, $\{\sigma_i, \sigma_j\}$, and the operator products, $\sigma_i \sigma_j$, for $i, j \in \{1, 2, 3\}$. Use this to determine the maximum number of Pauli operators that can be simultaneously represented as diagonal matrices.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Product Rule:**
    $\sigma_i \sigma_j = \delta_{ij} I + i \sum_k \epsilon_{ijk} \sigma_k$

    **Commutator:**
    $[\sigma_i, \sigma_j] = \sigma_i \sigma_j - \sigma_j \sigma_i = 2i \sum_k \epsilon_{ijk} \sigma_k$

    **Anticommutator:**
    $\{\sigma_i, \sigma_j\} = \sigma_i \sigma_j + \sigma_j \sigma_i = 2 \delta_{ij} I$

    **Simultaneous Diagonalization:**
    Commuting matrices can be simultaneously diagonalized. Since $[\sigma_i, \sigma_j] \neq 0$ for $i \neq j$, no two distinct Pauli matrices (excluding $I$) commute.
    Thus, the maximum number is **1** (plus the identity). Typically, we choose $Z$ to be diagonal.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.3
    3. Let $\vec{v}$ be any real three-dimensional unit vector and $\theta$ a real number. Show that
       $$e^{i\theta \vec{v} \cdot \vec{\sigma}} = \cos(\theta)I_2 + i \sin(\theta)\vec{v} \cdot \vec{\sigma}$$
       where $\vec{v} \cdot \vec{\sigma} \equiv \sum_{j=1}^3 v_j \sigma_j$. Generalize this to show that for any function, $f(\cdot)$, that maps complex numbers to complex numbers,
       $$f(\theta \vec{v} \cdot \vec{\sigma}) = \frac{f(\theta) + f(-\theta)}{2} I + \frac{f(\theta) - f(-\theta)}{2} \vec{v} \cdot \vec{\sigma}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let $A = \vec{v} \cdot \vec{\sigma}$.
    Property: $A^2 = (\sum v_i \sigma_i)(\sum v_j \sigma_j) = \sum_{ij} v_i v_j \sigma_i \sigma_j = \sum_{ij} v_i v_j (\delta_{ij} I + i \epsilon_{ijk} \sigma_k) = (\sum v_i^2) I + 0 = |\vec{v}|^2 I = I$.
    So $A^2 = I$.

    **Exponential:**
    $e^{i\theta A} = \sum_{n=0}^\infty \frac{(i\theta)^n A^n}{n!} = \sum_{even} \frac{(i\theta)^n I}{n!} + \sum_{odd} \frac{(i\theta)^n A}{n!} $
    $= \cos(\theta) I + i \sin(\theta) A$

    **General Function (via Spectral Decomposition):**
    Since $A^2=I$, eigenvalues of $A$ are $\pm 1$. The eigenvalues of $\theta A$ are $\pm \theta$.
    $f(\theta A)$ will have eigenvalues $f(\theta)$ and $f(-\theta)$.
    We can write $f(\theta A) = c_0 I + c_1 A$.
    For eigenvalue $+1$ (of A): $f(\theta) = c_0 + c_1$.
    For eigenvalue $-1$ (of A): $f(-\theta) = c_0 - c_1$.
    Solving for coefficients:
    $c_0 = \frac{f(\theta) + f(-\theta)}{2}$
    $c_1 = \frac{f(\theta) - f(-\theta)}{2}$

    Thus:
    $$f(\theta A) = \frac{f(\theta) + f(-\theta)}{2} I + \frac{f(\theta) - f(-\theta)}{2} A $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.4
    4. Pauli decompositions are often utilized in quantum computing to break computations into pieces that are manageable to implement on quantum devices. Convince yourself that a complete basis for representing generic spin-1/2 Hermitian operators is spanned by the Pauli matrices, i.e., all single-qubit operators can be expressed as a vector of Pauli coefficients. Utilizing the trace properties of the Pauli operators, devise a systematic way of determining the Pauli decomposition of an arbitrary single-qubit operator, $H_2$. Extend the technique to arbitrary $n$-qubit operators, $H_{2^n}$. Implement your scheme to determine the multi-qubit Pauli decomposition of the four-qubit operators:
       $$\hat{\phi}(JLP) = \frac{1}{15} \text{diag}(-15, -13, \dots, 15)$$
       and
       $$\hat{\phi}(HO) = \frac{a+a^\dagger}{\sqrt{2}}$$
       where $\hat{\phi}(HO)$ is defined by matrix elements $\hat{\phi}_{j,j+1} = \hat{\phi}_{j+1,j} = \sqrt{j+1} \forall j \in \{0, 14\}$. How does the number of terms in these two decompositions scale with the number of qubits spanned by the operator? For an arbitrary operator, what is the worst-case scaling of the number of terms in its Pauli decomposition?
    """)
    return


@app.cell
def _(np):
    def get_pauli_label(i: int, n_qubits: int) -> str:
        """Convert integer index to Pauli string (e.g. 0 -> II...I, 1 -> II...X)"""
        labels = ["I", "X", "Y", "Z"]
        s = ""
        for _ in range(n_qubits):
            s = labels[i % 4] + s
            i //= 4
        return s

    def pauli_decomposition(matrix: np.ndarray) -> dict[str, float]:
        """
        Decompose an n-qubit matrix into Pauli basis.
        Returns a dictionary of {pauli_string: coefficient} for non-zero coeffs.
        Uses the trace trick: c_P = (1/2^n) * Tr(Matrix @ P)
        """
        from qiskit.quantum_info import Pauli

        n = int(np.log2(matrix.shape[0]))
        dim = 2**n
        coeffs = {}

        # We can iterate through all 4^n Paulis.
        # For n=4, 4^4 = 256, which is small enough.

        for i in range(4**n):
            label = get_pauli_label(i, n)
            P = Pauli(label).to_matrix()

            # c = (1/d) * Tr(M * P)
            # Since P is Hermitian, Tr(M P) is checking overlap.
            # Using Frobenius inner product concepts.
            # P matrix from Qiskit is normalized such that eigenvalues are +/- 1.
            # Tr(P_i P_j) = 2^n * delta_ij

            trace_val = np.trace(matrix @ P)
            c = trace_val / dim

            if abs(c) > 1e-10:
                coeffs[label] = c

        return coeffs

    # Define operators
    N = 4
    dim = 2**N

    # 1. JLP Operator: Diagonal with linear spacing
    diag_vals = np.linspace(-15, 15, 16) / 15.0  # -1 to 1 basically
    # Actually the problem says: 1/15 * diag(-15, -13, ..., 15)
    # This generates -1, -13/15, ..., 1
    # Check spacing: -15, -13, -11, ..., 15 is range(start=-15, stop=17, step=2)
    diag_vals_exact = np.arange(-15, 17, 2) / 15.0
    phi_jlp = np.diag(diag_vals_exact)

    jlp_decomp = pauli_decomposition(phi_jlp)

    # 2. HO Operator: (a + a_dag)/sqrt(2) ~ x
    # phi_{j, j+1} = sqrt(j+1)
    phi_ho = np.zeros((dim, dim))
    for _j in range(dim - 1):
        val = np.sqrt(_j + 1)
        phi_ho[_j, _j + 1] = val
        phi_ho[_j + 1, _j] = val
    # Normalize? "defined by matrix elements ..." - doesn't explicitly say /sqrt(2) in element def,
    # but the formula says (a+a_dag)/sqrt(2).
    # The matrix elements given are for 'a' usually: a|n> = sqrt(n)|n-1>. <n-1|a|n>=sqrt(n).
    # so <j|a|j+1> = sqrt(j+1).
    # phi_ho matrix constructed above corresponds to (a + a_dag).
    # We should divide by sqrt(2) to match the formula.
    phi_ho /= np.sqrt(2)

    ho_decomp = pauli_decomposition(phi_ho)

    print(f"JLP Decomposition ({len(jlp_decomp)} terms):")
    for k, v in list(jlp_decomp.items())[:5]:  # Show first 5
        print(f"  {k}: {v:.4f}")

    print(f"\nHO Decomposition ({len(ho_decomp)} terms):")
    for k, v in list(ho_decomp.items())[:5]:
        print(f"  {k}: {v:.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Scaling Analysis:**
    *   **Worst Case:** An arbitrary $2^n \times 2^n$ Hermitian matrix has $4^n$ degrees of freedom. Thus, in the worst case, the Pauli decomposition requires **$4^n$ terms**.
    *   **$\phi(JLP)$ (Diagonal):** This operator is diagonal in the computational basis. It only involves $I$ and $Z$ terms. The number of terms scales as the number of diagonal matrices, which is $2^n$ (all combinations of Z and I). Is it efficient? $16$ terms for $n=4$.
    *   **$\phi(HO)$ (Tridiagonal):** The harmonic oscillator position operator couples nearest neighbors. It involves bit flips ($X$) and phase alignment ($X$ and $Y$). The scaling is generally polynomial in $n$ for physical operators like this (often $O(n)$ or $O(n^2)$ weight-1 and weight-2 Paulis dominant, but exact representation might require more).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### E. Connecting Finite and Infinite Dimensions

    The quantum harmonic oscillator Hamiltonian, $\hat{H} = \frac{\hat{p}^2}{2m} + \frac{m\omega^2}{2} \hat{x}^2$ may be expressed in terms of dimensionless variables, $\bar{p} = \hat{p}/\sqrt{\hbar m \omega}$ and $\bar{x} = \hat{x} \sqrt{m\omega/\hbar}$, as
    $$\hat{H} = \frac{\hbar\omega}{2} (\bar{p}^2 + \bar{x}^2)$$
    In this exercise, you will calculate (confirm) the harmonic oscillator ground state energy in four complementary ways.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### E.1
    1. **Ladder operators**:
       Considering the operators $\hat{a} = \frac{\bar{x} + i\bar{p}}{\sqrt{2}}$ and $\hat{a}^\dagger = \frac{\bar{x} - i\bar{p}}{\sqrt{2}}$, propagate the position-momentum canonical commutation relation to determine $\hat{a}, \hat{a}^\dagger$ and $[\bar{x}, \bar{p}]$. By writing the Hamiltonian in terms of these ladder operators, calculate the ground state energy expectation value as $\langle 0|\hat{H}|0\rangle$ where $|0\rangle$ is the ground state.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Given $[\hat{x}, \hat{p}] = i\hbar$.
    Dimensionless variables: $\bar{x} = \sqrt{\frac{m\omega}{\hbar}} \hat{x}$ and $\bar{p} = \frac{1}{\sqrt{\hbar m \omega}} \hat{p}$.
    Commutator: $[\bar{x}, \bar{p}] = \frac{1}{\hbar} [\hat{x}, \hat{p}] = i$.

    **Ladder Commutator:**
    $[\hat{a}, \hat{a}^\dagger] = \frac{1}{2} [\bar{x} + i\bar{p}, \bar{x} - i\bar{p}] = \frac{1}{2} ([\bar{x}, -i\bar{p}] + [i\bar{p}, \bar{x}]) = \frac{1}{2} (-i[\bar{x}, \bar{p}] - i[\bar{p}, \bar{x}])$
    Using $[\bar{x}, \bar{p}] = i$ and $[\bar{p}, \bar{x}] = -i$:
    $= \frac{1}{2} (-i(i) - i(-i)) = \frac{1}{2} (1 + 1) = 1$.
    So $[\hat{a}, \hat{a}^\dagger] = 1$.

    **Hamiltonian:**
    $\bar{x} = \frac{\hat{a} + \hat{a}^\dagger}{\sqrt{2}}$, $\bar{p} = \frac{\hat{a} - \hat{a}^\dagger}{i\sqrt{2}}$.
    $\bar{x}^2 + \bar{p}^2 = \frac{1}{2} (\hat{a} + \hat{a}^\dagger)^2 - \frac{1}{2} (\hat{a} - \hat{a}^\dagger)^2 = \frac{1}{2} ( (\hat{a}^2 + \hat{a}\hat{a}^\dagger + \hat{a}^\dagger\hat{a} + (\hat{a}^\dagger)^2) - (\hat{a}^2 - \hat{a}\hat{a}^\dagger - \hat{a}^\dagger\hat{a} + (\hat{a}^\dagger)^2) )$
    $= \hat{a}\hat{a}^\dagger + \hat{a}^\dagger\hat{a}$.
    Using $\hat{a}\hat{a}^\dagger = 1 + \hat{a}^\dagger\hat{a} = 1 + N$:
    $= (1 + N) + N = 2N + 1$.

    $H = \frac{\hbar\omega}{2} (\bar{x}^2 + \bar{p}^2) = \hbar\omega (N + \frac{1}{2})$.

    **Ground State Energy:**
    For the ground state $|0\rangle$, $N|0\rangle = 0$.
    $\langle 0 | H | 0 \rangle = \hbar\omega \langle 0 | (N + \frac{1}{2}) | 0 \rangle = \frac{\hbar\omega}{2}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### E.2
    2. **Finite truncation in number basis (qudit)**:
       In the basis formed by eigenstates of the number operator, $\hat{N} = \hat{a}^\dagger \hat{a}$, write the matrix representation (or a pattern-extendable low-energy subspace) for the operators $\hat{N}, \hat{a}$, and $\hat{a}^\dagger$. In the ordering of your basis, what is the vector form of $|0\rangle$? Truncating these matrices at increasing dimensionalities $\Lambda$ to systematically include higher energy states, show the convergence of $\langle 0|\hat{H}|0\rangle$ as a function of $\Lambda$. Discuss the form of the commutator in these finite truncations, and connect your observation to the Stone-von Neumann theorem through the trace of the canonical commutation relation.
    """)
    return


@app.cell
def _(np, plt):
    def run_number_basis_truncation(max_dim=10):
        energies = []
        dims = range(2, max_dim + 1)

        for dim in dims:
            # Creation (a_dag) and Annihilation (a) operators
            # a|n> = sqrt(n)|n-1>
            # a_dag|n> = sqrt(n+1)|n+1>
            a = np.zeros((dim, dim))
            for n in range(1, dim):
                a[n - 1, n] = np.sqrt(n)

            a_dag = a.T

            # Number operator N = a_dag * a
            # In the limited basis, this will be diagonal(0, 1, ..., dim-1)
            # except potentially at the boundary depending on truncation.
            N = a_dag @ a

            # Hamiltonian H = hbar*omega * (N + 1/2) (Setting hbar*omega = 1)
            H = N + 0.5 * np.eye(dim)

            # Ground state |0> is [1, 0, ..., 0]
            psi_0 = np.zeros(dim)
            psi_0[0] = 1.0

            energy = psi_0.T @ H @ psi_0
            energies.append(energy)

            # Check commutator [a, a_dag] in finite basis
            comm = a @ a_dag - a_dag @ a
            # Ideally = Identity
            # Trace of commutator of finite matrices is always 0.
            # Trace of Identity is dim.
            # Thus [A, B] = I is impossible in finite dimensions.
            if dim == 5:
                print(
                    f"Commutator trace at dim={dim}: {np.trace(comm):.2f} (Expected 0)"
                )
                print(
                    f"Top-left of commutator at dim={dim}: {comm[0, 0]:.2f} (Expected 1)"
                )

        return dims, energies

    number_dims, number_energies = run_number_basis_truncation(20)

    plt.figure(figsize=(6, 4))
    plt.plot(number_dims, number_energies, "o-", label="Number Basis")
    plt.axhline(0.5, color="r", linestyle="--", label="Exact (0.5)")
    plt.xlabel("Truncation Dimension")
    plt.ylabel("Ground State Energy")
    plt.title("Convergence in Number Basis (Exact)")
    plt.legend()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Observation:**
    In the Number basis, the matrix representation of the Hamiltonian is diagonal (essentially by definition). Truncation simply removes higher energy states. Since the ground state $|0\rangle$ is the first basis vector, the calculated energy $\langle 0 | H | 0 \rangle$ is exactly $0.5$ regardless of the truncation size (as long as dim $\ge 1$).

    **Commutator and Finite Dimensions:**
    In any finite dimensional space, $Tr([A, B]) = Tr(AB) - Tr(BA) = 0$. However, $Tr(I) = d$. Thus, $[x, p] = i\hbar I$ (or $[a, a^\dagger] = I$) cannot be satisfied exactly in finite dimensions. This is why the commutator usually fails at the edges of the truncated matrix (the "boundary term"). This relates to the Stone-von Neumann theorem which implies that the canonical commutation relations can only be represented unitarily on an infinite-dimensional Hilbert space.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### E.3
    3. **Continuous wavefunction**:
       You saw above that the ground state of the harmonic oscillator is a computational basis state when expressed in the number operator basis. Fill in the steps to Sakurai textbook Eq. (2.3.30) to show that the ground state expressed in the continuous position basis has a Gaussian wavefunction, $\psi(x) = \langle x|0\rangle \propto e^{-\bar{x}^2/2}$. Again, calculate the ground state energy expectation value, this time expressed as an integral in position space.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Start with $\hat{a}|0\rangle = 0$.
    Express in position basis: $\langle x | \hat{a} | 0 \rangle = 0$.
    Use $\hat{a} = \frac{1}{\sqrt{2}} (\bar{x} + i\bar{p}) = \frac{1}{\sqrt{2}} (\bar{x} + \frac{\partial}{\partial \bar{x}})$.

    $$\frac{1}{\sqrt{2}} \left( \bar{x} + \frac{d}{d\bar{x}} \right) \psi_0(\bar{x}) = 0$$
    $$\frac{d\psi_0}{d\bar{x}} = -\bar{x} \psi_0$$
    $$\frac{d\psi_0}{\psi_0} = -\bar{x} d\bar{x}$$
    $$\ln \psi_0 = -\frac{\bar{x}^2}{2} + C $$
    $$\psi_0(\bar{x}) = A e^{-\bar{x}^2/2}$$

    **Energy Expectation:**
    $H = \frac{1}{2} (-\frac{d^2}{d\bar{x}^2} + \bar{x}^2)$.
    Calculate $\langle \psi_0 | H | \psi_0 \rangle$.
    Using Gaussian integrals $\int e^{-x^2} dx = \sqrt{\pi}$, $\int x^2 e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$.
    Kinetic term $-\frac{d^2}{dx^2} e^{-x^2/2} = (1 - x^2) e^{-x^2/2}$.
    Potential term $x^2 e^{-x^2/2}$.
    Sum: $(1 - x^2 + x^2) = 1$.
    $H \psi_0 = \frac{1}{2} \psi_0$.
    Thus $E_0 = 1/2$ (in units of $\hbar\omega$).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### E.4
    4. **Basis of position eigenstates**:
       Consider a basis of position eigenstates organized into a grid characterized by symmetric upper/lower bounds $x_{max}$ and lattice spacing $\delta x$. First, show that an approximate representation of the squared momentum operator in this basis is $\bar{p}^2 = -\frac{\partial^2}{\partial \bar{x}^2} \approx \frac{1}{\delta x^2} A$, where $A$ is a Toeplitz matrix with $a_0 = 2$ and $a_1 = a_{-1} = -1$. The matrix representation of $\bar{x}$ in this basis is diagonal with matrix elements spanning $\pm x_{max}$. With a relationship of $\delta x = \frac{2x_{max}}{n_{states}-1}$, show that the ground state energy of this Hamiltonian approaches its continuum value as the number of states increases. Plot the associated ground state wavefunctions, are they as expected? have you chosen an $x_{max}$ that captures its spatial extent?
    """)
    return


@app.cell
def _(np, plt):
    def solve_position_basis(n_states=100, x_max=5.0):
        # Grid parameters
        x = np.linspace(-x_max, x_max, n_states)
        dx = x[1] - x[0]

        # Potential Energy Matrix: V = x^2 / 2
        # Diagonal matrix with x^2/2 on the diagonal
        V_diag = 0.5 * x**2
        V = np.diag(V_diag)

        # Kinetic Energy Matrix: T = p^2 / 2 = -0.5 * d^2/dx^2
        # Finite difference: f''(x) ~ (f(x+dx) - 2f(x) + f(x-dx)) / dx^2
        # Matrix A has 2 on diagonal, -1 on off-diagonals.
        # T = (1 / (2 * dx^2)) * A

        main_diag = 2.0 * np.ones(n_states)
        off_diag = -1.0 * np.ones(n_states - 1)

        A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        T = (1.0 / (2.0 * dx**2)) * A

        # Total Hamiltonian
        H = T + V

        # Diagonalize
        evals, evecs = np.linalg.eigh(H)

        # Ground state
        E0 = evals[0]
        psi0 = evecs[:, 0]

        # Normalize (Sum |psi|^2 dx = 1) -> Sum |psi|^2 = 1/dx
        # Standard linalg.eigh returns normalized vectors such that sum |v|^2 = 1.
        # To match continuum wavefunction, we want integral |psi(x)|^2 dx = 1.
        # So we scale by 1/sqrt(dx).
        psi0_continuum = psi0 / np.sqrt(dx)

        # Ensure positive phase for plotting
        if np.mean(psi0_continuum) < 0:
            psi0_continuum *= -1

        return x, E0, psi0_continuum

    # Convergence Study
    n_values = [10, 20, 50, 100, 200]
    energies = []

    plt.figure(figsize=(12, 5))

    # Plot Wavefunctions
    plt.subplot(1, 2, 1)

    # Exact Gaussian for comparison: psi(x) = (1/pi^1/4) * exp(-x^2/2)
    x_exact = np.linspace(-5, 5, 200)
    psi_exact = (1.0 / (np.pi**0.25)) * np.exp(-(x_exact**2) / 2.0)
    plt.plot(x_exact, psi_exact, "k--", label="Exact Gaussian", linewidth=2)

    for _n in n_values:
        x, E0, psi = solve_position_basis(n_states=_n, x_max=5.0)
        energies.append(E0)
        if _n in [20, 50, 100]:
            plt.plot(x, psi, label=f"N={_n}, E={E0:.4f}")

    plt.title("Ground State Wavefunction")
    plt.xlabel("x")
    plt.ylabel("psi(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Energy Convergence
    plt.subplot(1, 2, 2)
    plt.plot(n_values, energies, "o-")
    plt.axhline(0.5, color="r", linestyle="--", label="Exact (0.5)")
    plt.xlabel("Number of Grid Points")
    plt.ylabel("Ground State Energy")
    plt.title("Energy Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### E.5
    5. Consider implications of these results for qubit implementations with respect to the Pauli decompositions performed at the end of Exercise I D. If the quantum simulation cost scales with the number and size of terms in the Pauli decomposition of the Hamiltonian, which basis might you choose for digital representation of a harmonic oscillator-type Hilbert space on a quantum computer?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Discussion:**
    *   **Position Basis:** The Hamiltonian involves $x^2$ (diagonal) and $p^2$ (finite difference, tridiagonal-ish).
        *   $x^2$ is diagonal $\rightarrow$ only involves $Z$ terms. Efficient.
        *   $p^2$ couples neighbors $\rightarrow$ involves weight-2 terms like $XX, YY$.
        *   Need many grid points ($N$) for accuracy. $N=2^q$.
    *   **Number Basis:** The Hamiltonian $H = \hbar\omega(N + 1/2)$ is **diagonal** in this basis.
        *   $N = a^\dagger a$ is diagonal.
        *   Representation is just a sum of $Z$ terms (measuring the Hamming weight if using unary encoding, or weighted sums if using binary encoding).
        *   Or simply, the time evolution $e^{-iHt}$ is just a phase gate on each basis state.
        *   **Pauli Decomposition:** If we map $|n\rangle$ to computational basis states, the Hamiltonian is diagonal, so it only contains $I$ and $Z$ terms. This makes simulation extremely efficient (commuting terms, no Trotter error).

    **Conclusion:**
    The **Number Basis** is superior for the Harmonic Oscillator if we just want to simulate the free Hamiltonian, because it diagonalizes the operator. If we add interactions (e.g. $x^4$), the position basis might become more competitive or we would need to switch bases (QFT). But for the bare HO, the Number basis is optimal.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## II. Algorithmic Components
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A. Hadamard Operator as Transform

    The Hadamard gate has a matrix form of $H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{\sqrt{2}} (|0\rangle\langle0| + |0\rangle\langle1| + |1\rangle\langle0| - |1\rangle\langle1|)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.1
    1. Verify that the Hadamard is Unitary and is its own Hermitian conjugate.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Definition of Unitary: $H^\dagger H = I$.
    """)
    return


@app.cell
def _(np):
    Hadmard = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

    assert np.allclose(Hadmard @ Hadmard.T.conj(), np.eye(2))
    assert np.allclose(Hadmard, Hadmard.T.conj())
    return (Hadmard,)


@app.cell
def _(mo):
    mo.md(r"""
    #### A.2
    2. Calculate the Hadamard transformed Pauli operators: $H X H$, $H Y H$, and $H Z H$. Do your results make sense in light of the Hadamard’s role as the $x \leftrightarrow z$ basis transformation, $H = |0\rangle\langle+| + |1\rangle\langle-|$?
    """)
    return


@app.cell
def _(Hadmard, SparsePauliOp, np):
    from qiskit.quantum_info import Operator

    H = Operator(Hadmard)
    X_op = Operator(SparsePauliOp("X").to_matrix())
    Y_op = Operator(SparsePauliOp("Y").to_matrix())
    Z_op = Operator(SparsePauliOp("Z").to_matrix())

    HXH = H @ X_op @ H
    HYH = H @ Y_op @ H
    HZH = H @ Z_op @ H

    print("H X H =")
    print(HXH.data)
    print("\nH Y H =")
    print(HYH.data)
    print("\nH Z H =")
    print(HZH.data)

    # Verify the basis transformation interpretation
    # H swaps X <-> Z and Y -> -Y
    print("\n\nVerification:")
    print(f"H X H = Z: {np.allclose(HXH.data, Z_op.data)}")
    print(f"H Y H = -Y: {np.allclose(HYH.data, -Y_op.data)}")
    print(f"H Z H = X: {np.allclose(HZH.data, X_op.data)}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.3
    3. Show that the single-qubit Hadamard operator over $n$-qubits, $H^{\otimes n}$, can be written as
       $$H^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x,y} (-1)^{x \cdot y} |x\rangle\langle y|$$
       where $|x\rangle = |x_1\rangle \otimes |x_2\rangle \otimes \cdots \otimes |x_n\rangle$ and $x = x_1 x_2 \cdots x_n$ is a binary representation of $x$, and likewise for $y$, leading $x \cdot y$ to be the bit-wise inner product.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    \begin{aligned}
    H &= \frac{1}{\sqrt{2}} \left(\ket{0}\bra{0} + \ket{0}\bra{1} + \ket{1}\bra{0} - \ket{1}\bra{1}\right) \\
    &= \frac{1}{\sqrt{2}} \sum_{a,b \in \{0,1\}} (-1)^{a \cdot b} |a\rangle\langle b| \\
    \Rightarrow H^{\otimes n} &= \left(\frac{1}{\sqrt{2}}\right)^n \sum_{x,y \in \{0,1\}^n} (-1)^{x \cdot y} |x\rangle\langle y| = \frac{1}{\sqrt{2^n}} \sum_{x,y} (-1)^{x \cdot y} |x\rangle\langle y|
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.4
    4. Write and describe the state $H^{\otimes n} |0\rangle$, where $|0\rangle = |0\rangle_1 |0\rangle_2 \cdots |0\rangle_n$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    \begin{align}
    H^{\otimes n} |0\rangle &= (H\ket{0})^{\otimes n} = \ket{+}+_{1}\ket{+}+_{2} \cdots \ket{+}+_{n} = |+\rangle^{\otimes n} \\
    &= \frac{1}{\sqrt{2^n}} \sum_{x \in \{0,1\}^n} \ket{x} \\
    \end{align}
    $$

    This state is an equal superposition of all $2^n$ computational basis states. Each basis state $|x\rangle$ corresponds to a unique binary string of length $n$. Thus, applying the Hadamard transform to the initial state $|0\rangle$ creates a uniform distribution over all unentangled possible states in the $n$-qubit Hilbert space.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### B. GHZ State Preparation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    \ket{\text{GHZ}}_{n} = \frac{1}{\sqrt{2}} (|0\rangle^{\otimes n} + |1\rangle^{\otimes n})
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.1
    1. With one Hadamard operator and $n-1$ CNOTs, write a Unitary circuit that is capable of preparing the $n$-qubit GHZ state. What is the depth of your circuit? On average, what fraction of time is each qubit idle?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This can be accomplished with the following circuit:
    - Apply a Hadamard gate to the first qubit.
    - Sequentially apply CNOT between entangled qubits (first qubit & previously entangled qubits) with unentangled qubits.

    This works because $\ket{\text{CNOT}} = \ket{00}\bra{00} + \ket{01}\bra{01} + \ket{11}\bra{10} + \ket{10}\bra{11}$. Meaning that since the targets are all initialized to $\ket{0}$, the CNOTs will copy the state of the control qubit to the target qubits. This propagates the single qubit superposition to a maximally correlated state which is the GHZ state.

    *Circuit Depth:*
    The depth of this circuit depends on how the CNOTs are arranged. If we apply them sequentially, the depth is $n$ (1 for Hadamard + $n-1$ for CNOTs). However, if we can apply CNOTs in parallel (e.g., using a binary tree structure), the depth can be reduced to $\log_2(n) + 1$.

    *Fraction of Time Idle:*
    In the sequential case we have already establised that it take $n$ gate times to complete the circuit. The Hadamard occupies 1 qubit for 1 time unit, and each CNOT occupies 2 qubits for 1 time unit. Therefore we have $n^2$ unit-gate times with 1 unit-gate time occupied by the Hadamard and $2(n-1)$ unit-gate times occupied by the CNOTs. Thus, the total occupied time is $\frac{2n-1}{n^2}$, and the idle time is $1 - \frac{2n-1}{n^2} = \frac{n^2 - 2n + 1}{n^2} = \frac{(n-1)^2}{n^2}$.

    In the logarithmic depth case, the occupied unit-gate time is the same but our denominator is now $n(\log_2(n) + 1)$.
    """)
    return


@app.cell
def _(QuantumCircuit):
    # Sequential GHZ
    def create_ghz_circuit(n_qubits: int) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        qc.h(0)  # Apply Hadamard to the first qubit
        for i in range(1, n_qubits):
            qc.cx(0, i)  # CNOT from first qubit to each subsequent qubit
        return qc

    ghz_4q = create_ghz_circuit(4)
    ghz_4q.draw(output="mpl")
    return


@app.cell
def _(QuantumCircuit):
    # Binary Tree GHZ
    def construct_ghz_binary_tree(n_qubits: int) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        qc.reset(range(n_qubits))

        # Step 1: Apply Hadamard to the first qubit
        qc.h(0)

        superimposed = [0]
        unentangled = list(range(1, n_qubits))
        while unentangled:
            new_superimposed = []
            for control in superimposed:
                if not unentangled:
                    break
                target = unentangled.pop(0)
                qc.cx(control, target)
                new_superimposed.append(target)
            superimposed.extend(new_superimposed)

        return qc

    ghz__bin_circuit = construct_ghz_binary_tree(8)
    ghz__bin_circuit.draw(output="mpl")
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.2
    2. Analyze step-by-step the functionality of the following circuit.

    ```
    |0⟩ ─H─ ●──────────────●──
    |0⟩ ──── ⊕──⊕──/──|0⟩──⊕──
    |0⟩ ─H─ ●──────X──────────
    ```

    Is the final state dependent on the outcome of the measurement?
    """)
    return


@app.cell
def _(QuantumCircuit):
    qc_b2 = QuantumCircuit(3, 1)
    qc_b2.reset([0, 1, 2])
    qc_b2.h([0, 2])
    qc_b2.cx(0, 1)
    qc_b2.cx(2, 1)
    qc_b2.measure(1, 0)
    with qc_b2.if_test((qc_b2.clbits[0], 1)):
        qc_b2.x(2)
    qc_b2.reset(1)
    qc_b2.cx(0, 1)

    qc_b2.draw(output="mpl")
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Step-by-Step Functionality:*
    1. Qubits are initialized to $\ket{0}$.
    2. Apply Hadmard to a two main qubits
    3. XOR the two qubits via CNOTs on a central qubit
    4. Flip $q_2$ (0-indexed) conditionally based off if $q_0 \neq q_2$ (measurement outcome of the XOR).
    5. Entangle $q_0$ and $q_1$ via CNOT. We now have $\ket{\text{GHZ}}_{3}$

    *Final State Analysis:*
    The final state is not dependent on the outcome of the measurement because the measurement only determines whether to apply an $X$ gate to $q_2$. Since $q_2$ is prepared in the $\ket{+}$ state and $\ket{+}$ is the $+1$ eigenstate of the $X$ operator, applying an $X$ gate to $\ket{+}$ leaves it unchanged. Thus the conditional branch has not affect on the only part of the state that is changed (everything is non-dependent by lack of connection).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.3
    3. Analyze step-by-step the functionality of the extension of the circuit above. If this circuit is extended to generate an $n$-qubit GHZ state, how many CNOTs are used? Describe the strategy of this circuit in the language of measuring and manipulating (CS)parity.
    """)
    return


@app.cell
def _(QuantumCircuit):
    from qiskit.circuit.classical import expr

    qc_b3 = QuantumCircuit(5, 3)
    qc_b3.reset([0, 1, 2, 3, 4])
    qc_b3.h([0, 2, 4])

    qc_b3.cx(0, 1)
    qc_b3.cx(2, 1)
    qc_b3.cx(2, 3)
    qc_b3.cx(4, 3)
    qc_b3.measure(1, 0)
    qc_b3.measure(3, 1)

    with qc_b3.if_test((qc_b3.clbits[0], 1)):
        qc_b3.x(2)

    # classical bit 3 = XOR of classical bits 1 and 2
    condition = expr.bit_xor(qc_b3.clbits[0], qc_b3.clbits[1])
    with qc_b3.if_test(condition):
        qc_b3.x(4)

    qc_b3.reset([1, 3])
    qc_b3.cx(0, 1)
    qc_b3.cx(2, 3)
    qc_b3.draw(output="mpl")
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Step-by-Step Functionality:*
    This is an extension of the previous circuits to prepare $\ket{\text{GHZ}}_{n}$. We start by superimposing $n/2$ primary qubits via the Hadamard gate. We then use CNOTS to XOR adjascent pairs of qubits into central ancilla qubits. We then measure these ancilla qubits to determine differences in parity between the pairs. Based on these measurements, we conditionally apply $X$ gates to the primary qubits to ensure they are all correlated. Finally, we entangle all primary qubits via CNOTs to create the GHZ state.

    *Number of CNOTs:*
    We have 2 CNOT per neighboring pair of primary qubits + 1 CNOT to entangle the ancillary at the end. Assuming this linear topology with an odd number of qubits, we have $n-1$ XOR CNOTs + $n//2$ entangling CNOTs
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.4
    4. Refer to [PRX Quantum 5, 030339](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.030339) to contextualize the significance of this observation. Discuss the value of mid-circuit measurement and feedforward beyond quantum error correction.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The significance of the mid-circuit measurement is that we can entangle at long distances (e.g., between qubits that are not directly connected) by using an ancilla qubit to mediate the entanglement. This is particularly useful in quantum computing architectures where qubit connectivity is limited. By measuring the ancilla and applying feedforward operations based on the measurement outcome, we can effectively create entanglement between distant qubits without needing direct interaction.

    You can see in Fig. 1 that this turns entanglement of nearest-neighbor distance $n$ from a $\mathcal{O}(n)$ operation to a $\mathcal{O}(1)$ operation, which drastically reduces the circuit depth and potential error accumulation in quantum algorithms that require entanglement over long distances.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### C. Deutsch-Jozsa Algorithm

    As discussed interferometrically in the first lecture, the Deutsch Jozsa algorithm may be represented with the following quantum circuit
    """)
    return


@app.cell
def _(ClassicalRegister, QuantumCircuit, QuantumRegister):
    from qiskit.circuit import Gate

    def draw_symbolic_deutsch_jozsa_latex():
        reg_n = QuantumRegister(1, name=r"|0\rangle^{\otimes n}")
        reg_anc = QuantumRegister(1, name=r"|-\rangle")
        cr = ClassicalRegister(1, name="")  # Empty name to hide label if possible
        qc = QuantumCircuit(reg_n, reg_anc, cr)
        h_n_gate = Gate(name=r"H^{\otimes n}", num_qubits=1, params=[])
        u_f_gate = Gate(name=r"U_f", num_qubits=2, params=[])
        qc.append(h_n_gate, [reg_n[0]])
        qc.append(u_f_gate, [reg_n[0], reg_anc[0]])
        qc.append(h_n_gate, [reg_n[0]])
        qc.measure(reg_n[0], cr[0])

        return qc

    draw_symbolic_deutsch_jozsa_latex().draw(output="latex", scale=0.7)
    return


@app.cell
def _(mo):
    mo.md(r"""
    where $U_f \ket{x}\ket{y} = \ket{x}\ket{y \oplus f(x)}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.1
    1. Show that $|y\rangle = |-\rangle$ is an eigenstate of the controlled operator, $U_f$, with eigenvalue $(-1)^{f(x)}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > $\oplus$ := XOR
    $$
    \begin{align}
    U_f \ket{x}\ket{y} &= \ket{x}\ket{y \oplus f(x)}\\
    U_{f}\ket{x}\ket{-} &= U_{f}\ket{x}\left[\ket{0} - \ket{1}\right] \cdot\frac{1}{\sqrt{2}}\\
    &= [U_{f}\ket{x}\ket{0} - U_{f}\ket{x}\ket1]\cdot\frac{1}{\sqrt{2}}\\
    &= \frac{1}{\sqrt{2}}\ket{x} [\ket{0 \oplus f(x)} - \ket{1 \oplus f(x)}]\\
    &= \frac{1}{\sqrt{2}}\ket{x} [\ket{f(x)} - \ket{\neg f(x)}]\\\\
    &= \begin{cases}
    \frac{1}{\sqrt{2}}\ket{x} [\ket{0} - \ket{1}] = \ket{x}\ket{-} & \text{if }f(x) = 0\\
    \frac{1}{\sqrt{2}}\ket{x} [\ket{1} - \ket{0}] = -\ket{x}\ket{-} & \text{if }f(x) = 1
    \end{cases}\\
    &= (-1)^{f(x)} \ket{x}\ket{-}
    \end{align}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Step-by-step explinations of the above derivation:
    1. We start with the definition of $U_f$ given above.
    2. We apply $U_f$ to the state $\ket{x}\ket{-}$, where $\ket{-} = \frac{1}{\sqrt{2}}(\ket{0} - \ket{1})$.
    3. We use linearity to separate the action of $U_f$ on $\ket{0}$ and $\ket{1}$.
    4. We rewrite $U_f$ in terms of its XOR definition, which allows us to express the resulting state in terms of $f(x)$.
    5. We recognize that $\ket{0 \oplus f(x)}$ is just $\ket{f(x)}$ and $\ket{1 \oplus f(x)}$ is $\ket{\neg f(x)}$.

    6. Depending on whether $f(x)$ is 0 or 1, we find that the resulting state is either $\ket{x}\ket{-}$ or $-\ket{x}\ket{-}$
    8. Thus, we conclude that $\ket{-}$ is an eigenstate of $U_f$ with eigenvalue $(-1)^{f(x)}$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.2
    2. Utilizing the multi-qubit local Hadamard transform, analyze the above circuit to find the final state prior to measurement. Describe the logic applied to the measurement results that allows a single implementation of this circuit to determine whether the function is constant, $f(x) = c \forall x \in \{0,1\}^n$ and $c \in \{0,1\}$, or the function is balanced with an equal number of 0s and 1s. Remark upon the state remaining on the last qubit after measurement.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Working through the algorithm we can understand the final state by tracing how it got there.

    1. Initial State: The algorithm states in all $\ket{0}$ for the input wires and $\ket{-}$ for the output target.
    2. Superposition: We apply the Hadamard to put $x$ into a (positive) superposition of all possible states
    3. Oracle: As discussed above each state is multiplied by $(-1)^{f(x)}$, so we are still in a superposition of all states with states that evaluate to 1 being negative (opposite phase) relative to the ones that evaluate to 0 (this uses global phase invariance).
    4. Superposition Collapse: The second Hadamard converts this phase information into the logical basis
    5. Measurement: When we measure we get the average of the superpositions

    $$
    \begin{align*}
    \text{Amplitude} &= \frac{1}{2^{n}}\sum\limits_{x}(-1)^{f(x)}\\
    &= \begin{cases}
    \frac{1}{2^{n}}\sum\limits_{x}(-1)^{c} = \frac{1}{2^{n}}2^n = 1 & \text{if }f(x) = c \ \forall x\\
    \frac{1}{2^{n}}\sum\limits_{x}(-1)^{f(x)} = 0 & \text{if }f(x) \text{ is balanced}
    \end{cases}
    \end{align*}
    $$

    Thus we measure $\ket{0}^{\otimes n}$ with a probability proportional to the agreement of the function. When the function is constant (i.e., $f(x) = c$ for all $x$), the amplitude is $\pm 1$ and we measure $\ket{0}^{\otimes n}$ with certainty. When the function is balanced (i.e., $f(x) = 1$ for half of the inputs and $f(x) = 0$ for the other half), the amplitude for $\ket{0}^{\otimes n}$ is 0, so the measurement outcome is guaranteed to be some bitstring other than all zeros. Therefore, a single measurement suffices to distinguish constant from balanced: if the outcome is $\ket{0}^{\otimes n}$ the function is constant, and any other outcome certifies it is balanced. The last qubit remains in the $\ket{-}$ state throughout the algorithm and is not measured, so it does not affect the outcome of the algorithm.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.3
    3. For each of the four possible single-qubit constant/balanced binary functions, write a circuit representation of the two-qubit gate $U_f$.

    $$
    \begin{align}
    f(0) = f(1) = 0\\
    f(0) = f(1) = 1\\
    f(0) = f(1) \oplus 1 = 0\\
    f(0) = f(1) \oplus 1 = 1 \, ,\\
    \end{align}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Constant Functions:
    1. For $f(0) = f(1) = 0$ (const-0), and $U_{f}\ket{x}\ket{y} = \ket{x}\ket{y \oplus f(x)} = \ket{x}\ket{y}$. Therefore have $U_f = I \otimes I$ (identity gate).
    2. For $f(0) = f(1) = 1$ (const-1), and $U_{f}\ket{x}\ket{y} = \ket{x}\ket{y \oplus f(x)} = \ket{x}\ket{y \oplus 1}$. Therefore have $U_f = I \otimes X$ (NOT gate on the target).

    Balanced Functions:

    3. For $f(0) = 0$ and $f(1) = 1$ (identity), and $U_{f}\ket{x}\ket{y} = \ket{x}\ket{y \oplus f(x)}$. Therefore have $U_f = CNOT$ (control on the first qubit, target on the second).
    4. For $f(0) = 1$ and $f(1) = 0$ (negation), and $U_{f}\ket{x}\ket{y} = \ket{x}\ket{y \oplus f(x)}$. Therefore have $U_f = CNOT \cdot (I\otimes X)$ (Pauli-X on the target qubit followed by a CNOT gate).
    """)
    return


@app.cell
def _():
    # Gate 1 + Validate of DJ working
    # TODO: finish this work, good validation
    # qc_dj1 = QuantumCircuit(2, 1)
    # qc_dj1.reset([0, 1])
    # qc_dj1.h(0)

    # # Uf = I tensor I
    # Uf = np.eye(4)
    # qc_dj1.unitary(Uf, [0, 1], label='U_f')
    # qc_dj1.
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.4
    4. For each of the eight possible two-qubit constant/balanced binary functions,

    $$
    (f(0)\ f(1)\ f(2)\ f(3)) = \begin{cases}
    0\ 0\ 0\ 0 \\
    1\ 1\ 1\ 1 \\
    0\ 0\ 1\ 1 \\
    0\ 1\ 0\ 1 \\
    0\ 1\ 1\ 0 \\
    1\ 0\ 0\ 1 \\
    1\ 0\ 1\ 0 \\
    1\ 1\ 0\ 0 \\
    \end{cases}
    $$

    write a circuit representation of the three-qubit gate $U_f$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    You need circuit representations for the three-qubit $U_f$ gate for the eight two-qubit functions provided. Assume the inputs are $x_1 x_0$ representing the decimal values 0, 1, 2, 3.

    - **0000:** Constant 0. Circuit: Identity (do nothing).

    - **1111:** Constant 1. Circuit: Apply an $X$ gate to the target.

    - **0011:** $f(x) = x_1$. Circuit: CNOT with $x_1$ controlling the target.

    - **0101:** $f(x) = x_0$. Circuit: CNOT with $x_0$ controlling the target.

    - **0110:** $f(x) = x_1 \oplus x_0$. Circuit: Two CNOTs; one from $x_1$ to the target, and one from $x_0$ to the target.

    - **1001:** $f(x) = \neg(x_1 \oplus x_0)$. Circuit: Same as above (two CNOTs), followed by an $X$ gate on the target.

    - **1010:** $f(x) = \neg x_0$. Circuit: Apply an $X$ gate to the target, then a CNOT from $x_0$ to the target.

    - **1100:** $f(x) = \neg x_1$. Circuit: Apply an $X$ gate to the target, then a CNOT from $x_1$ to the target.
    """)
    return


@app.cell
def _(QuantumCircuit):
    from typing import Final

    # C.4 qubits: q0 = x_0, q1 = x_1, q2 = y (target for y <- y XOR f(x)).
    C4_PATTERNS: Final[tuple[str, ...]] = (
        "0000",
        "1111",
        "0011",
        "0101",
        "0110",
        "1001",
        "1010",
        "1100",
    )

    def truth_table_f(pattern: str, x: int) -> int:
        assert 0 <= x <= 3
        return int(pattern[x])

    def build_c4_oracle(pattern: str) -> QuantumCircuit:
        qc = QuantumCircuit(3)
        if pattern == "0000":
            pass
        elif pattern == "1111":
            qc.x(2)
        elif pattern == "0011":
            qc.cx(1, 2)
        elif pattern == "0101":
            qc.cx(0, 2)
        elif pattern == "0110":
            qc.cx(1, 2)
            qc.cx(0, 2)
        elif pattern == "1001":
            qc.cx(1, 2)
            qc.cx(0, 2)
            qc.x(2)
        elif pattern == "1010":
            qc.x(2)
            qc.cx(0, 2)
        elif pattern == "1100":
            qc.x(2)
            qc.cx(1, 2)
        else:
            msg = f"unknown C.4 pattern: {pattern!r}"
            raise ValueError(msg)
        return qc

    def verify_oracle_unitary(
        pattern: str, atol: float = 1e-5
    ) -> tuple[bool, list[str]]:
        from qiskit.quantum_info import Statevector

        oracle = build_c4_oracle(pattern)
        failures: list[str] = []
        for x in range(4):
            prep = QuantumCircuit(3)
            if x & 1:
                prep.x(0)
            if (x >> 1) & 1:
                prep.x(1)
            prep.compose(oracle, [0, 1, 2], inplace=True)
            sv = Statevector(prep)
            x0 = x & 1
            x1 = (x >> 1) & 1
            fy = truth_table_f(pattern, x)
            idx = x0 + 2 * x1 + 4 * fy
            prob = float(sv.probabilities()[idx])
            if abs(prob - 1.0) > atol:
                failures.append(
                    f"x={x}: want prob 1 on index {idx} (x0,x1,y)=({x0},{x1},{fy}), got {prob}"
                )
        return (len(failures) == 0, failures)

    def deutsch_jozsa_2q_circuit(oracle: QuantumCircuit) -> QuantumCircuit:
        qc = QuantumCircuit(3, 2)
        qc.x(2)
        qc.h(2)
        qc.h(0)
        qc.h(1)
        qc.compose(oracle, [0, 1, 2], inplace=True)
        qc.h(0)
        qc.h(1)
        qc.measure([0, 1], [0, 1])
        return qc

    def run_dj_histogram(qc: QuantumCircuit, shots: int = 4096) -> dict[str, int]:
        from qiskit import transpile
        from qiskit_aer import Aer

        backend = Aer.get_backend("aer_simulator")
        compiled = transpile(qc, backend)
        job = backend.run(compiled, shots=shots)
        return job.result().get_counts()

    def is_constant_pattern(pattern: str) -> bool:
        weight = sum(int(pattern[i]) for i in range(4))
        return weight in (0, 4)

    def dj_outcome_matches_class(
        counts: dict[str, int], pattern: str
    ) -> bool:
        total = sum(counts.values())
        if total == 0:
            return False
        p00 = counts.get("00", 0) / total
        if is_constant_pattern(pattern):
            return p00 >= 0.999
        return p00 <= 0.001

    def summarize_c4_batch() -> tuple[int, int, list[str]]:
        oracle_pass = 0
        dj_pass = 0
        rows: list[str] = []
        for p in C4_PATTERNS:
            o_ok, _ = verify_oracle_unitary(p)
            oracle_pass += int(o_ok)
            dj_qc = deutsch_jozsa_2q_circuit(build_c4_oracle(p))
            d_ok = dj_outcome_matches_class(run_dj_histogram(dj_qc), p)
            dj_pass += int(d_ok)
            rows.append(
                f"| `{p}` | {'pass' if o_ok else 'fail'} | {'pass' if d_ok else 'fail'} |"
            )
        return oracle_pass, dj_pass, rows

    return (
        C4_PATTERNS,
        build_c4_oracle,
        deutsch_jozsa_2q_circuit,
        dj_outcome_matches_class,
        run_dj_histogram,
        summarize_c4_batch,
        truth_table_f,
        verify_oracle_unitary,
    )


@app.cell
def _(C4_PATTERNS: "Final[tuple[str, ...]]", mo):
    c4_pattern_select = mo.ui.dropdown(
        options=list(C4_PATTERNS),
        value="0000",
        label=r"C.4 pattern $f(0)f(1)f(2)f(3)$",
    )
    return (c4_pattern_select,)


@app.cell
def _(
    QuantumCircuit,
    build_c4_oracle,
    c4_pattern_select,
    deutsch_jozsa_2q_circuit,
    dj_outcome_matches_class,
    mo,
    run_dj_histogram,
    summarize_c4_batch,
    truth_table_f,
    verify_oracle_unitary,
):
    import io

    from qiskit.quantum_info import Statevector as _Statevector
    from qiskit.visualization import plot_histogram as _plot_histogram

    pattern = c4_pattern_select.value
    oracle = build_c4_oracle(pattern)
    dj_qc = deutsch_jozsa_2q_circuit(oracle.copy())
    oracle_ok, oracle_failures = verify_oracle_unitary(pattern)
    counts = run_dj_histogram(dj_qc)
    dj_ok = dj_outcome_matches_class(counts, pattern)

    oracle_pass_n, dj_pass_n, batch_rows = summarize_c4_batch()
    batch_md = (
        f"**Batch:** {oracle_pass_n}/8 oracle truth-table checks pass; "
        f"{dj_pass_n}/8 Deutsch–Jozsa runs match constant vs balanced.\n\n"
        "| pattern | oracle | DJ |\n|---|---|---|\n" + "\n".join(batch_rows)
    )

    truth_lines = [
        "| $x$ | $f(x)$ | oracle check |",
        "|---:|---:|:---:|",
    ]
    for x_in in range(4):
        fx = truth_table_f(pattern, x_in)
        prep = QuantumCircuit(3)
        if x_in & 1:
            prep.x(0)
        if (x_in >> 1) & 1:
            prep.x(1)
        prep.compose(oracle, [0, 1, 2], inplace=True)
        sv_row = _Statevector(prep)
        x0 = x_in & 1
        x1 = (x_in >> 1) & 1
        idx = x0 + 2 * x1 + 4 * fx
        row_ok = abs(float(sv_row.probabilities()[idx]) - 1.0) < 1e-5
        truth_lines.append(
            f"| {x_in} | {fx} | {'ok' if row_ok else 'fail'} |",
        )

    hist_fig = _plot_histogram(counts, figsize=(6, 3.5))
    _buf = io.BytesIO()
    hist_fig.savefig(_buf, format="png", bbox_inches="tight")
    hist_png = mo.image(_buf.getvalue())

    oracle_fail_md = (
        mo.md("Oracle failures:\n- " + "\n- ".join(oracle_failures))
        if not oracle_ok
        else mo.md("")
    )
    mo.vstack(
        [
            c4_pattern_select,
            mo.md(batch_md),
            mo.md(
                f"**Selected `{pattern}`:** oracle unit check "
                f"{'passed' if oracle_ok else 'failed'}; "
                f"DJ histogram {'matches' if dj_ok else 'does not match'} "
                f"({'constant' if sum(int(pattern[i]) for i in range(4)) in (0, 4) else 'balanced'})."
            ),
            oracle_fail_md,
            mo.md("\n".join(truth_lines)),
            mo.md("**$U_f$ (3-qubit oracle)**"),
            oracle.draw(output="latex", scale=0.58),
            mo.md("**Full Deutsch–Jozsa circuit ($n=2$ data qubits)**"),
            dj_qc.draw(output="latex", scale=0.52),
            # mo.md("**Measurement counts (first two qubits, MSB order as in Qiskit)**"),
            # hist_png,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.5
    5. For an $n$-bit application, $f : \{0,1\}^n \to \{0,1\}$, calculate the total number of possible constant/balanced binary functions. How does this number scale asymptotically with $n$. Describe one circuit compilation strategy implementing $U_f$ for arbitrary $n$. (*Consider which functions are simple to compile vs those that require additional circuit logic.*) In the worst case, how does the depth of your circuit scale with $n$?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    For an $n$-bit input, there are $2^n$ possible input combinations. There will always be 2 constant functions ($f(x) = 0$ and $f(x) = 1$). For balanced functions, we need to choose a any half of the evaluations to be 1 and the other half to be 0. This is equivalent to $\binom{2^n}{2^{n-1}}$. Therefore the total number of functions is $2 + \binom{2^n}{2^{n-1}}$.

    You can see this by example above:
    - For $n=1$, we have $2 + \binom{2}{1} = 2 + 2 = 4$ total functions.
    - For $n=2$, we have $2 + \binom{4}{2} = 2 + 6 = 8$ total functions.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The number of balanced functions is $\binom{2^n}{2^{n-1}}$, which by Stirling's approximation scales as $\sim 2^{2^n}/\sqrt{\pi \cdot 2^{n-1}}$. This is **doubly exponential** in $n$ (i.e., exponential in $2^n$), not merely exponential.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The circuit can being compiled into [Algebriac Normal Form](https://en.wikipedia.org/wiki/Algebraic_normal_form) (ANF) which is a sum of products of the input bits. Each product term corresponds to a multi-controlled NOT gate, and the sum corresponds to applying these gates in sequence. In the worst case, if the function is balanced and has a large number of terms in its ANF, the depth of the circuit can scale exponentially with $n$ due to the need for multi-controlled gates that may require decomposition into basic gates.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### D. Quantum Fourier Transform Identity

    In the interferometry perspective of quantum computation, we interpreted the role of the Hadamard transform as delocalizing the wavefunction in Hilbert space. Generalizing this perspective to an operation that transfers delta-functions to constant-functions between conjugate Hilbert spaces (e.g., position-momentum) leads to the Quantum Fourier Transform. The following product-structured identity is crucial for creating and understanding the Quantum Fourier Transform circuit,

    $$
    \text{QFT}|x\rangle = \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n-1} e^{\frac{2\pi i x k}{2^n}} |k\rangle = \bigotimes_{l=1}^n \frac{|0\rangle + e^{2\pi i (0.x_{n-l+1}\dots x_n)}|1\rangle}{\sqrt{2}}
    $$

    In the binary notation used here, integers are notated as $\mathbb{x}=x_{1}x_{2}\cdots x_{n} = x_{1} 2^{n-1} + \cdots + x_{n} 2^{0}$ and fractions are notated as $(0.x_{n+1}x_{n+2} \cdots x_{n+m}) = \frac{x_{n+1}}{2^{1}} + \frac{x_{n+2}}{2^{2}} + \cdots + \frac{x_{n+m}}{2^{m}}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > Much of this workthrough is adapted from the qiskit derivation found here: https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/quantum-fourier-transform.ipynb
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.1
    1. Show that this identity is true.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    \begin{align}
    \text{QFT}\ket{x} &= \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n-1} e^{\frac{2\pi i x k}{2^n}} |k\rangle \\
    &= \frac{1}{\sqrt{2^n}} \sum \exp\left[2\pi i x \left(\sum_{l=1}^n x_{l}/2^{n-l}\right)\right] \ket{x_{1} \dots x_{n}} \\
    &= \frac{1}{\sqrt{2^{n}}}\sum\limits_{x=0}^{2^{n}-1} \bigotimes_{k=1}^{n} \exp\left[2 \pi i x \frac{y_{k}}{2^{k}}\right]\ket{x_{1}\dots x_{n}}\\
    &= \frac{1}{\sqrt{2^{n}}}\sum\limits_{x_{1}}\dots\sum\limits_{x_{n}} \bigotimes_{k=1}^{n} \exp\left[2 \pi i x \frac{y_{k}}{2^{k}}\right]\ket{x_{1}\dots x_{n}}\\
    &= \frac{1}{\sqrt{2^{n}}}\bigotimes_{k=1}^{n}(\ket{0} + e^\frac{2 \pi i x}{2^{k}}\ket{1})\\
    &= \bigotimes_{l=1}^n \frac{|0\rangle + e^{2\pi i (0.x_{n-l+1}\dots x_n)}|1\rangle}{\sqrt{2}}
    \end{align}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Step Explanations:
    1. We start with the definition of the Quantum Fourier Transform (QFT) applied to a computational basis state $\ket{x}$.
    2. We express $x$ in terms of its binary representation $x = x_1 x_2 \dots x_n$.
    3. Rewrite into a fractional binary form $y = y_1 \dots y_n / 2^n = \sum\limits_{k=1}^{n} y_k/2^k$
    4. Exponential of sums to sum of exponentials: $\exp\left[2\pi i x \sum_{k=1}^n y_k/2^k\right] = \prod_{k=1}^n \exp\left[2\pi i x y_k/2^k\right]$.
    5. Expanding y
    6. Rearranging terms
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.2
    2. Use this form to motivate explicit QFT circuit implementations for 1, 2 and 3 qubits.
    """)
    return


@app.cell
def _(QuantumCircuit, mo, np):
    def qft_rotations(circuit, n):
        """Performs qft on the first n qubits in circuit (without swaps)"""
        if n == 0:
            return circuit
        n -= 1
        circuit.h(n)
        for qubit in range(n):
            circuit.cp(np.pi / 2 ** (n - qubit), qubit, n)
        # At the end of our function, we call the same function again on
        # the next qubits (we reduced n by one earlier in the function)
        qft_rotations(circuit, n)

    def swap_registers(circuit, n):
        for qubit in range(n // 2):
            circuit.swap(qubit, n - qubit - 1)
        return circuit

    def qft(circuit, n):
        """QFT on the first n qubits in circuit"""
        qft_rotations(circuit, n)
        swap_registers(circuit, n)
        return circuit

    # Let's see how it looks:
    qc = QuantumCircuit(4)
    qft(qc, 4)
    mo.vstack(
        [
            qft(QuantumCircuit(1), 1).draw(output="latex"),
            qft(QuantumCircuit(2), 2).draw(output="latex"),
            qft(QuantumCircuit(3), 3).draw(output="latex"),
        ]
    )
    # qc.draw(output='mpl')
    return (qft,)


@app.cell
def _(mo):
    # Choose your input state & show the logical and QFT output
    from qiskit_aer import Aer

    QFT_BIT_COUNT = 4
    # Slider from 0 - 2^bits
    qft_input_slider = mo.ui.slider(start=0, stop=2**QFT_BIT_COUNT - 1, step=1)
    mo.vstack(
        [
            mo.ui.text("Input State:"),
            qft_input_slider,
        ]
    )
    return Aer, QFT_BIT_COUNT, qft_input_slider


@app.cell
def _(
    Aer,
    QFT_BIT_COUNT,
    QuantumCircuit,
    mo,
    plot_bloch_multivector,
    qft,
    qft_input_slider,
):

    def logical_binary(n, bits):
        qc = QuantumCircuit(n)
        for i in range(n):
            if bits & (1 << (n - i - 1)):
                qc.x(i)
        return qc

    qft_input = qft_input_slider.value
    qft_bin_input = logical_binary(QFT_BIT_COUNT, qft_input)
    qft_circuit = qft(qft_bin_input.copy(), QFT_BIT_COUNT)

    sim = Aer.get_backend("aer_simulator")
    qft_bin_input.save_statevector()
    qft_input_statevector = sim.run(qft_bin_input).result().get_statevector()
    qft_circuit.save_statevector()
    qft_output_statevector = sim.run(qft_circuit).result().get_statevector()

    mo.vstack(
        [
            plot_bloch_multivector(qft_input_statevector, title=rf"{qft_input}"),
            plot_bloch_multivector(
                qft_output_statevector, title=rf"Freq View of {qft_input}"
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.Extra
    **Extra**: Using the QFT in its role of transforming operators between momentum and position space, show how the local finite-difference momentum operator of I E.4 can be replaced by a proper nonlocal lattice momentum operator to achieve more rapid convergence of the ground state energy.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## III. Quantum Information Phenomena
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A. Uncertainty Principle and Enhanced Sensing

    Consider two Hermitian $(\hat{A} = \hat{A}^\dagger = (\hat{A}^*)^T)$ operators, $\hat{A}$ and $\hat{B}$, whose commutator is non-vanishing.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.1
    1. Show that there is an inherent uncertainty in the associated expectation values
       $$\Delta(\hat{A})\Delta(\hat{B}) \geq \frac{1}{2} \left|\left\langle \left[\hat{A},\hat{B}\right] \right\rangle\right|$$
       where $\Delta(\hat{A}) = \sqrt{\left\langle \left(\hat{A} - \bar{\hat{A}}\right)^2 \right\rangle}$ is the standard deviation from the average, $\bar{\hat{A}} = \langle \hat{A} \rangle$. For position and momentum, $[\hat{x}, \hat{p}] = i\hbar$, this reproduces the familiar $\Delta(\hat{x})\Delta(\hat{p}) \geq \hbar/2$. **Hint** you may find useful the Cauchy-Schwartz inequality for states in an inner product vector space: $\langle u|u\rangle\langle v|v\rangle \geq |\langle u|v\rangle|^2$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    \begin{align}
    \text{Define Operators:} & \quad \hat{A}' = A - \langle A \rangle, \quad \hat{B}' = B - \langle B \rangle \\
    \text{Variance Definition:} & \quad (\Delta A)^2 = \langle \hat{A}'^2 \rangle = \langle \psi | \hat{A}' \hat{A}' | \psi \rangle \\
    \text{Define States:} & \quad | f \rangle = \hat{A}' | \psi \rangle, \quad | g \rangle = \hat{B}' | \psi \rangle \\
    \text{Inner Products:} & \quad \langle f | f \rangle = (\Delta A)^2, \quad \langle g | g \rangle = (\Delta B)^2 \\
    \text{Cauchy-Schwarz:} & \quad \langle f | f \rangle \langle g | g \rangle \geq | \langle f | g \rangle |^2 \\
    \text{Expand } \langle f | g \rangle: & \quad \langle f | g \rangle = \langle \psi | \hat{A}' \hat{B}' | \psi \rangle \\
    \text{Decomposition Identity:} & \quad \hat{A}' \hat{B}' = \frac{1}{2} (\hat{A}' \hat{B}' + \hat{B}' \hat{A}') + \frac{1}{2} (\hat{A}' \hat{B}' - \hat{B}' \hat{A}') \\
    & \quad \hat{A}' \hat{B}' = \frac{1}{2} \{ \hat{A}', \hat{B}' \} + \frac{1}{2} [ \hat{A}', \hat{B}' ] \\
    \text{Expectation Value:} & \quad \langle f | g \rangle = \frac{1}{2} \langle \{ \hat{A}', \hat{B}' \} \rangle + \frac{1}{2} \langle [ A, B ] \rangle \\
    \text{Hermitian Properties:} & \quad \langle \{ \hat{A}', \hat{B}' \} \rangle \in \mathbb{R}, \quad \langle [ A, B ] \rangle \in \mathbb{I} \\
    \text{Magnitude Squared:} & \quad | \langle f | g \rangle |^2 = \left( \frac{1}{2} \langle \{ \hat{A}', \hat{B}' \} \rangle \right)^2 + \left| \frac{1}{2} \langle [ A, B ] \rangle \right|^2 \\ \text{Inequality:} & \quad (\Delta A)^2 (\Delta B)^2 \geq \left| \frac{1}{2} \langle [ A, B ] \rangle \right|^2 \\ \text{Final Form:} & \quad \Delta A \Delta B \geq \frac{1}{2} | \langle [ A, B ] \rangle | \\
    \end{align}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.2
    2. Discuss how this relationship limits the minimum uncertainty achievable when measuring two conjugate variables. Further, describe the process through which fluctuations can be redistributed to reduce uncertainty in one variable for improved sensitivity in quantum sensing (see Phys. Rev. A 46, R6797(R) *Spin squeezing and reduced quantum noise in spectroscopy*).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The product of the uncertainties (variance) of the two variables is bounded by the expectation value of their commutator, however the individual uncertainty is not. We can choose to reduce uncertainty in one variable at the cost of exploding uncertainty in the other variable. 3b1b has an [excellent video](https://www.youtube.com/watch?v=MBnnXbOM5S4) about how this works in the context of the time/frequency domain. You can choose to measurement position very precisely at the cost of having a very uncertain momentum, or you can choose to measure momentum very precisely at the cost of having a very uncertain position. This is the basis for quantum sensing, where we can choose to measure one variable with very high precision by allowing the conjugate variable to be very uncertain.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### B. No-Cloning Theorem

    The no-cloning theorem states that it is impossible to design a quantum circuit that perfectly copies an unknown quantum state, i.e., there exists no $U_{xerox}$ such that $U_{xerox}|\psi\rangle|0\rangle = |\psi\rangle|\psi\rangle$ for arbitrary $|\psi\rangle$. More broadly, there exists no superoperator, $\mathcal{E}$, such that $\mathcal{E}(|\psi\rangle\langle\psi| \otimes |x\rangle\langle x|) = |\psi\rangle\langle\psi| \otimes |\psi\rangle\langle\psi|$ that produces additional copies of an arbitrary state $|\psi\rangle$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.1
    1. Find a specific $U_{xerox}$ that copies Z-axis basis states, $U_{xerox}|x\rangle|0\rangle = |x\rangle|x\rangle$ for $x \in \{0, 1\}$. What does this operator create when applied to more general superposition initial states?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    If $U_{xerox}$ is a CNOT gate with the first qubit as the control and the second qubit as the target, then we have:
    $$
    \begin{align}
    U_{xerox} \ket{0}\ket{0} &= \ket{0}\ket{0} \\
    U_{xerox} \ket{1}\ket{0} &= \ket{1}\ket{1}
    \end{align}
    $$

    However, if we apply this operator to a superposition state, such as $\ket{+}\ket{0} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{1})\ket{0}$, we get:
    $$
    \begin{align}
    U_{xerox} \ket{+}\ket{0} &= U_{xerox} \left( \frac{1}{\sqrt{2}}(\ket{0} + \ket{1})\ket{0} \right) \\
    &= \frac{1}{\sqrt{2}}(U_{xerox} \ket{0}\ket{0} + U_{xerox} \ket{1}\ket{0}) \\
    &= \frac{1}{\sqrt{2}}(\ket{0}\ket{0} + \ket{1}\ket{1}) \\
    &= \frac{1}{\sqrt{2}}(\ket{00} + \ket{11})
    \end{align}
    $$
    This resulting state is an entangled state, not a product state of two identical copies. Therefore, while $U_{xerox}$ can copy the basis states, it cannot copy arbitrary superposition states without creating entanglement, which is a manifestation of the no-cloning theorem.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.2
    2. By exploring the repercussions of the existence of such a $U_{xerox}$, devise a proof-by-contradiction for the no-cloning theorem.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Consider two arbitrary states $\ket{\psi}$ and $\ket{\phi}$. Applying $U_{xerox}$ to both states gives:

    $$
    \begin{align}
    U_{xerox} \ket{\psi}\ket{0} &= \ket{\psi}\ket{\psi} &= \ket{\Psi} \\
    U_{xerox} \ket{\phi}\ket{0} &= \ket{\phi}\ket{\phi} &= \ket{\Phi}
    \end{align}
    $$

    Because $U_{xerox}$ is unitary, it preserves the inner product of the two initial systems. We calculate the inner product before and after the operator is applied:
    $$
    \begin{align}
    \braket{\Psi | \Phi} &= (U_{xerox} \ket{\psi}\ket{0})^\dagger (U_{xerox} \ket{\phi}\ket{0}) \\
    &= \bra{\psi}\bra{0} U_{xerox}^\dagger U_{xerox} \ket{\phi}\ket{0} \\
    &= \bra{\psi}\bra{0} \ket{\phi}\ket{0} \\
    &= \braket{\psi | \phi} \braket{0 | 0} \\
    &= \braket{\psi | \phi} \\
    \braket{\Psi | \Phi} &= \bra{\psi}\bra{\psi} \ket{\phi}\ket{\phi} \\
    &= \braket{\psi | \phi} \braket{\psi | \phi} \\
    &= \braket{\psi | \phi}^2 \\
    \braket{\psi | \phi} &= \braket{\psi | \phi}^2
    \end{align}
    $$

    This implies that $\braket{\psi | \phi} = 0$ or $\braket{\psi | \phi} = 1$. However, since $\ket{\psi}$ and $\ket{\phi}$ are arbitrary states, they can be non-orthogonal (i.e., $0 < |\braket{\psi | \phi}| < 1$). This leads to a contradiction, as the equation cannot hold for all arbitrary states, proving that $U_{xerox}$ cannot exist.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.3
    3. Discuss physical implications of the no-cloning theorem, e.g., on state tomography, the uncertainty principle, quantum error correction, security in quantum cryptography, etc.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The no-cloning theorem is very important to many of the things that make quantum computing interesting

    1. State Tomography: If the no-cloning theorem were not true, we could create many copies of an unknown quantum state and perform measurements on those copies on all Bell-basis states to perfectly reconstruct the original state. However, since we cannot clone, we can only perform state tomography by making measurements on a single copy of the state, which limits our ability to fully characterize it.
    2. Uncertainty Principle: The no-cloning theorem is important to the uncertainty principle because it keeps states unique and forces us to order our measurements. If we could clone states, we could measure one copy to determine one observable and another copy to determine the conjugate observable, thus violating the uncertainty principle.
    3. Quantum Error Correction: The no-cloning theorem is a fundamental limitation that quantum error correction schemes. The simplest error correction scheme is to make multiple copies of the state and perform majority voting to correct errors. However, since we cannot clone, we need to use more complex schemes that involve entanglement and syndrome measurements to detect and correct errors without directly copying the state.
    4. Quantum Cryptography: The no-cloning theorem is crucial for the security of quantum cryptography. If an eavesdropper could clone quantum states, they could intercept and copy the quantum information being transmitted without being detected. The no-cloning theorem ensures that any attempt to intercept and copy quantum information will introduce detectable disturbances, thus providing security guarantees for quantum communication protocols like Quantum Key Distribution (QKD).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.4
    4. If a $U_{xerox}$ did exist, utilize it to devise a super-luminal communications scheme.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > Largely inspired by the consequence section on the Wikipedia page for the no-cloning theorem: https://en.wikipedia.org/wiki/No-cloning_theorem#Consequences

    If $U_{xerox}$ existed, we could use it to create a superluminal communication scheme as follows:
    1. Alice and Bob share an entangled pair of qubits, $\ket{\Phi^+} = \frac{1}{\sqrt{2}}(\ket{00} + \ket{11})$.
    2. Alice wants to send a bit of information (0 or 1) to Bob.
    3. If Alice wants to send a 0, she measures her qubit in the Z-basis. This will collapse Bob's qubit to either $\ket{0}$ or $\ket{1}$ with equal probability.
    4. If Alice wants to send a 1, she measures her qubit in the X-basis. This will collapse Bob's qubit to either $\ket{+}$ or $\ket{-}$ with equal probability.
    5. Bob can then apply $U_{xerox}$ to his qubit and ancillary qubits to create multiple copies of his state.
    6. Bob can then perform measurements on the copies to do full state tomography and determine whether his state was collapsed and in which direction.

    This system works regardless of distance, allowing for superluminal communication. However, since $U_{xerox}$ does not exist, this scheme is purely hypothetical and cannot be implemented in practice.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.5
    5. Choose an approximate quantum cloning machine (either one available "on the market" or create your own) and demonstrate its fidelity for the six eigenvectors of $X$, $Y$, and $Z$ Pauli operators.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **N&C Exercise 1.2**: Explain how a device that could perfectly distinguish between two non-orthogonal states could be used to create $U_{xerox}$ to clone them. Or alternatively, show how $U_{xerox}$ could be used to distinguish non-orthogonal states.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > This response was generated by claude

    **N&C Exercise 1.2: Cloning ↔ State Discrimination**

    We show both directions:

    **Direction 1 — A perfect distinguisher implies a cloner.** Suppose we have a device $\mathcal{D}$ that perfectly distinguishes two non-orthogonal states $|\psi\rangle$ and $|\phi\rangle$ (with $\langle\psi|\phi\rangle \neq 0$). Given an unknown input promised to be one of these two states, apply $\mathcal{D}$ to learn which state it is. Once we have this classical knowledge, we can prepare arbitrarily many fresh copies of the identified state. This constitutes a perfect cloning machine for $\{|\psi\rangle, |\phi\rangle\}$, contradicting the no-cloning theorem. Therefore no such distinguisher can exist.

    **Direction 2 — A perfect cloner implies a distinguisher.** Suppose we have $U_{\text{xerox}}$ that clones arbitrary states: $U_{\text{xerox}}|\psi\rangle|0\rangle = |\psi\rangle|\psi\rangle$. Given one copy of an unknown state $|\alpha\rangle \in \{|\psi\rangle, |\phi\rangle\}$, apply $U_{\text{xerox}}$ repeatedly to produce $N$ copies $|\alpha\rangle^{\otimes N}$. Now perform the optimal collective measurement to distinguish $|\psi\rangle^{\otimes N}$ from $|\phi\rangle^{\otimes N}$. The overlap is $|\langle\psi|\phi\rangle|^{2N} \to 0$ as $N \to \infty$, so the error probability vanishes. In the limit, this gives perfect discrimination of non-orthogonal states, which we proved impossible in C.1. Therefore no such cloner can exist.

    **Conclusion.** The no-cloning theorem and the indistinguishability of non-orthogonal quantum states are logically equivalent — each impossibility implies the other.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### C. Distinguishability of Quantum States
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.1
    1. When provided an ensemble of identical quantum states, numerous measurements in multiple bases may perform *quantum state tomography* to determine the wavefunction characterizing the ensemble. With access to only a single quantum measurement in any basis, show that two non-orthogonal pure states cannot be perfectly distinguished.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    \begin{align}
    \ket{\phi} &= \alpha \ket{\psi_{\perp}} + \beta \ket{\psi_{\parallel}} \\
    \braket{m | \phi} &= \alpha \braket{m | \psi_{\perp}} + \beta \braket{m | \psi_{\parallel}} \\
    P(m) &= |\braket{m | \phi}|^2 \\
    &= |\alpha|^2 |\braket{m | \psi_{\perp}}|^2 + |\beta|^2 |\braket{m | \psi_{\parallel}}|^2 + 2 \text{Re}(\alpha^* \beta \braket{m | \psi_{\perp}}^* \braket{m | \psi_{\parallel}}) \\
    &< |\alpha|^2 + |\beta|^2 \\
    &= 1
    \end{align}
    $$


    Here $|\psi_\parallel\rangle$ is the component of $|\phi\rangle$ parallel to $|\psi\rangle$ and $|\psi_\perp\rangle$ is orthogonal, with $|\alpha|, |\beta| > 0$ since the states are non-orthogonal and non-identical. The strict inequality holds because $|\braket{m|\psi_\perp}|^2 < 1$ and $|\braket{m|\psi_\parallel}|^2 < 1$ for any measurement outcome $|m\rangle$ (no single basis vector can be simultaneously orthogonal to both components). Since $P(m) < 1$ for **every** outcome $m$ in **any** measurement basis, no outcome deterministically identifies one state over the other — every result has nonzero probability for both $|\psi\rangle$ and $|\phi\rangle$. Therefore, no single projective measurement can perfectly distinguish two non-orthogonal states.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.2
    2. Your experimentalist friend devises a game in which one instance of a quantum state is randomly generated that is promised to be prepared in either $|\psi_1\rangle$ or $|\psi_2\rangle$. The two possible wavefunctions are provided and generated with an even 50/50 probability. You are allowed to projectively measure this instance in a basis of your choice prior to guessing the state prepared.

    - Consider the two possible states of
      $$|\psi_1\rangle = |0\rangle \qquad |\psi_2\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - If you chose to measure in the basis $\{|\psi_1\rangle, |\psi_1^\perp\rangle\}$, what would be your maximum probability of success? By rotating the measurement basis in the plane established by $\{|\psi_1\rangle, |\psi_2\rangle\}$, determine the maximum probability of success that can be achieved.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > *This section was corrected by claude*

    **Measuring in the $\{|\psi_1\rangle, |\psi_1^\perp\rangle\}$ basis:**

    When state $|\psi_1\rangle = |0\rangle$ is prepared and measured in its own eigenbasis, outcome $|\psi_1\rangle$ occurs with certainty — this is always correctly identified.

    When state $|\psi_2\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ is measured in this basis:
    - Outcome $|\psi_1^\perp\rangle = |1\rangle$: probability $|\langle\psi_1^\perp|\psi_2\rangle|^2 = \frac{1}{2}$. This outcome is **conclusive** — only $|\psi_2\rangle$ can produce it.
    - Outcome $|\psi_1\rangle = |0\rangle$: probability $|\langle\psi_1|\psi_2\rangle|^2 = \frac{1}{2}$. This outcome is **inconclusive** — both states can produce it.

    With equal priors ($\frac{1}{2}$ each), the optimal strategy is: guess $|\psi_2\rangle$ if we see $|\psi_1^\perp\rangle$ (always correct), and guess $|\psi_1\rangle$ if we see $|\psi_1\rangle$ (correct when the state actually was $|\psi_1\rangle$). The overall success probability is:

    $$P(\text{success}) = \frac{1}{2} \cdot 1 + \frac{1}{2} \cdot \frac{1}{2} = \frac{3}{4}$$

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Provide a geometric interpretation for the orientation of this optimal basis. Use this geometric reasoning to show that the maximum probability of success to distinguish an arbitrary pair of states will be $p_{\text{max success}} = \frac{1+\sin(\arccos |\langle\psi_1|\psi_2\rangle|)}{2}$.
    """)
    return


@app.cell
def _(np, plt):
    # This visualization was corrected by claude
    # Geometric interpretation: on the Bloch sphere, |0⟩ and |+⟩ lie in the XZ plane.
    # |0⟩ is at the north pole (θ=0) and |+⟩ is on the equator (θ=π/2, φ=0).
    # We project onto the XZ great circle and show the optimal measurement basis bisecting them.

    # Bloch sphere angles: |0⟩ → θ=0, |+⟩ → θ=π/2 (on the x-axis)
    _theta_psi1 = 0        # |0⟩: north pole
    _theta_psi2 = np.pi/2  # |+⟩: equator, φ=0
    _theta_opt = (_theta_psi1 + _theta_psi2) / 2  # bisecting angle = π/4

    _fig, _ax = plt.subplots(figsize=(6, 6))

    # Draw the unit circle (XZ great circle of Bloch sphere)
    _angles = np.linspace(0, 2*np.pi, 200)
    _ax.plot(np.sin(_angles), np.cos(_angles), 'k-', alpha=0.15, linewidth=1)

    # State vectors (Bloch sphere: x=sinθ, z=cosθ)
    _ax.annotate('', xy=(np.sin(_theta_psi1), np.cos(_theta_psi1)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    _ax.annotate('', xy=(np.sin(_theta_psi2), np.cos(_theta_psi2)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Optimal measurement basis (bisects the angle)
    _ax.annotate('', xy=(np.sin(_theta_opt), np.cos(_theta_opt)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2, linestyle='--'))
    _ax.annotate('', xy=(-np.sin(_theta_opt), -np.cos(_theta_opt)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2, linestyle='--'))

    # Labels
    _ax.text(np.sin(_theta_psi1)-0.15, np.cos(_theta_psi1)+0.08,
             r'$|\psi_1\rangle = |0\rangle$', fontsize=12, color='blue')
    _ax.text(np.sin(_theta_psi2)+0.05, np.cos(_theta_psi2)+0.08,
             r'$|\psi_2\rangle = |+\rangle$', fontsize=12, color='red')
    _ax.text(np.sin(_theta_opt)+0.05, np.cos(_theta_opt)+0.08,
             r'$|m_{\mathrm{opt}}\rangle$', fontsize=11, color='green')

    # Annotate the angle
    _arc = np.linspace(_theta_psi1, _theta_psi2, 50)
    _ax.plot(0.3*np.sin(_arc), 0.3*np.cos(_arc), 'k-', linewidth=1)
    _ax.text(0.18, 0.25, r'$\theta$', fontsize=12)

    _ax.set_xlim(-1.3, 1.3); _ax.set_ylim(-1.3, 1.3)
    _ax.set_aspect('equal')
    _ax.set_xlabel(r'$\langle X \rangle$'); _ax.set_ylabel(r'$\langle Z \rangle$')
    _ax.set_title('Bloch sphere XZ plane: optimal basis bisects the two states')
    _ax.grid(True, alpha=0.3)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    - How is this optimal basis related to the difference of the density operators $D = \rho(|\psi_1\rangle) - \rho(|\psi_2\rangle)$? (*Hint:* see Holevo-Helstrom theorem)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > This response was generated by claude

    **Holevo-Helstrom theorem.** For equal priors ($\eta_1 = \eta_2 = 1/2$), the optimal success probability for distinguishing $\rho_1 = |\psi_1\rangle\langle\psi_1|$ from $\rho_2 = |\psi_2\rangle\langle\psi_2|$ is:

    $$p_{\max} = \frac{1}{2}\left(1 + \frac{1}{2}\|D\|_1\right), \quad D = \rho_1 - \rho_2$$

    where $\|D\|_1 = \text{Tr}|D| = \text{Tr}\sqrt{D^\dagger D}$ is the trace norm. The optimal POVM is $\{\Pi_1, \Pi_2\}$ where $\Pi_1$ projects onto the positive eigenspace of $D$ and $\Pi_2 = I - \Pi_1$ projects onto the negative eigenspace. That is, **the optimal measurement basis is the eigenbasis of $D$**.

    For our states $|\psi_1\rangle = |0\rangle$ and $|\psi_2\rangle = |+\rangle$:

    $$D = |0\rangle\langle 0| - |+\rangle\langle +| = \frac{1}{2}\begin{pmatrix}1 & -1 \\ -1 & -1\end{pmatrix}$$

    The eigenvalues are $\pm\frac{1}{\sqrt{2}}$, so $\|D\|_1 = \sqrt{2}$ and:

    $$p_{\max} = \frac{1}{2}\left(1 + \frac{1}{\sqrt{2}}\right) \approx 0.854$$

    The eigenvectors of $D$ bisect the angle between $|\psi_1\rangle$ and $|\psi_2\rangle$ on the Bloch sphere, confirming the geometric intuition that the optimal basis sits "halfway" between the two states.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Discuss how the odds of successful identification improve if the game provides an ensemble of two instances of the state. Should the same measurement basis be used for both measurements? If not, how would you choose the second measurement basis?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > This response was generated by claude

    **Two-instance state discrimination.** With two copies of the unknown state, there are three strategies of increasing power:

    **1. Independent identical measurements.** Measure both copies in the Helstrom-optimal basis. Each measurement succeeds with probability $p_{\max} \approx 0.854$. Using majority vote (or Bayesian updating), the combined success probability improves but is suboptimal since both measurements yield correlated information.

    **2. Adaptive measurement.** Measure the first copy in the Helstrom basis. The outcome updates the Bayesian prior from $(1/2, 1/2)$ to some $(\eta_1', \eta_2')$. The optimal basis for the second measurement is then the eigenbasis of $\eta_1' \rho_1 - \eta_2' \rho_2$, which is generically **different** from the first basis (tilted toward confirming the less likely hypothesis). This strictly outperforms identical measurements. For our states, the adaptive success probability is $P_{\text{adaptive}} \approx 0.899$, intermediate between identical measurements ($\approx 0.875$) and the collective optimum ($\approx 0.933$).

    **3. Collective (joint) measurement — optimal.** Measure both copies jointly using the Helstrom-optimal POVM for $\rho_1^{\otimes 2}$ vs $\rho_2^{\otimes 2}$. The overlap between the two-copy states is $|\langle\psi_1|\psi_2\rangle|^4$, so:

    $$p_{\max}^{(2)} = \frac{1}{2}\left(1 + \sqrt{1 - |\langle\psi_1|\psi_2\rangle|^4}\right)$$

    For our states, $|\langle 0|+\rangle|^2 = 1/2$, so $|\langle\psi_1|\psi_2\rangle|^4 = 1/4$ and:

    $$p_{\max}^{(2)} = \frac{1}{2}\left(1 + \frac{\sqrt{3}}{2}\right) \approx 0.933$$

    This strictly outperforms any local (adaptive or non-adaptive) strategy — illustrating that entangled measurements across copies can extract more information than sequential single-copy measurements.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.3
    3. If possible, discuss experimentally how to distinguish between the following two mixed state ensembles:
    $$\rho_1 = \frac{1}{2}|\Phi^+\rangle\langle\Phi^+| + \frac{1}{2}|\Phi^-\rangle\langle\Phi^-|$$
    $$\rho_2 = \frac{1}{2}|00\rangle\langle00| + \frac{1}{2}|11\rangle\langle11|$$
    where $|\Phi^\pm\rangle = \frac{|00\rangle \pm |11\rangle}{\sqrt{2}}$ are two entangled Bell states. If not possible, explain why.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > This response was generated by claude

    **It is not possible to distinguish $\rho_1$ from $\rho_2$, because they are the same density matrix.**

    Computing $\rho_1$ explicitly:

    $$|\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}\begin{pmatrix}1&0&0&1\\0&0&0&0\\0&0&0&0\\1&0&0&1\end{pmatrix}, \quad |\Phi^-\rangle\langle\Phi^-| = \frac{1}{2}\begin{pmatrix}1&0&0&-1\\0&0&0&0\\0&0&0&0\\-1&0&0&1\end{pmatrix}$$

    The off-diagonal coherence terms $\pm\frac{1}{2}|00\rangle\langle11|$ cancel in the equal mixture:

    $$\rho_1 = \frac{1}{2}(|\Phi^+\rangle\langle\Phi^+| + |\Phi^-\rangle\langle\Phi^-|) = \frac{1}{2}|00\rangle\langle00| + \frac{1}{2}|11\rangle\langle11| = \rho_2$$

    Since quantum mechanics predicts measurement outcomes from the density matrix alone (not the ensemble decomposition), no experiment can distinguish between the two preparations. This is a direct illustration of the **non-uniqueness of convex decompositions** of mixed states: the same $\rho$ admits both an entangled-state decomposition ($\rho_1$) and a separable-state decomposition ($\rho_2$), yet these are operationally identical.
    """)
    return


@app.cell
def _(np):
    # This response was generated by claude
    # Numerical verification that rho_1 = rho_2
    _phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    _phi_minus = np.array([1, 0, 0, -1]) / np.sqrt(2)
    _rho_1 = 0.5 * np.outer(_phi_plus, _phi_plus.conj()) + 0.5 * np.outer(_phi_minus, _phi_minus.conj())

    _ket_00 = np.array([1, 0, 0, 0])
    _ket_11 = np.array([0, 0, 0, 1])
    _rho_2 = 0.5 * np.outer(_ket_00, _ket_00) + 0.5 * np.outer(_ket_11, _ket_11)

    print("ρ₁ = ρ₂:", np.allclose(_rho_1, _rho_2))
    print("ρ₁ =\n", _rho_1.real)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### D. Quantum Gate Teleportation and Compilation
    """)
    return


@app.cell
def _(ClassicalRegister, QuantumCircuit, QuantumRegister, np):
    # Define the angle (theta = pi/4 targets the T gate)
    theta = np.pi / 4

    # Create registers
    _qr = QuantumRegister(2, name='q')
    _cr = ClassicalRegister(1, name='c')
    qc_magic = QuantumCircuit(_qr, _cr)

    # Step 1: Initialize the ancilla (q1) to the magic state |chi> = Rz(theta)|+>
    # qc_magic.h(_qr[1])
    # qc_magic.rz(theta, _qr[1])
    # qc_universal.barrier()

    # Step 2: Entangle states (Control=Ancilla, Target=Input)
    # Note: qr[0] is assumed to be the arbitrary input state |psi>
    qc_magic.cx(_qr[1], _qr[0])

    # Step 3: Measure the input qubit
    qc_magic.measure(_qr[0], _cr[0])

    # Step 4: Apply classical feedforward correction if measurement is 1
    with qc_magic.if_test((_cr[0], 1)):
        qc_magic.x(_qr[1])
        qc_magic.rz(2 * theta, _qr[1])

    # The successfully rotated state Rz(theta)|psi> now resides on _qr[1]
    qc_magic.draw(output="mpl")
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.1
    1. Analyze the function of the circuit above. $\ket{\chi} = R_z (\theta) \ket{+}$. Taking $\theta = \pi/4$, discuss the imporatance of this circuit in the context of $\{H, T\}$ being unversal for single-qubit unitaries and the fault tolerance available to the Clifford subgroup
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This circuit is an example of *magic state injection* or *gate teleportation*, which is a technique used in quantum computing to implement non-Clifford gates (like the $T$ gate) using only Clifford operations and a special resource state (the "magic state"). This is important because while Clifford gates are easier to implement, they are easily simulated classically and do not provide the full power of quantum computation. When just using Clifford gates, we remain on the axis-aligned points of the Bloch sphere ($\ket{0}$, $\ket{1}$, $\ket{+}$, $\ket{-}$, etc.). The $T$ gate allows us to access points on the Bloch sphere that are not aligned with the axes, enabling universal quantum computation. By preparing the magic state $\ket{\chi} = R_z(\theta) \ket{+}$ and using it in this teleportation circuit, we can effectively implement the $T$ gate on an arbitrary input state, thus achieving universality while maintaining fault tolerance through the use of Clifford operations.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    \begin{aligned}
    |\Psi_0\rangle &= \frac{1}{\sqrt{2}}(|0\rangle+e^{i\theta}|1\rangle) \otimes (\alpha|0\rangle+\beta|1\rangle) &\text{[Init]}\\
    |\Psi_1\rangle &= \frac{1}{\sqrt{2}}[\alpha|00\rangle+\beta|01\rangle+\beta e^{i\theta}|10\rangle+\alpha e^{i\theta}|11\rangle] &\text{[CNOT]}\\
    |\Psi_1\rangle &= \frac{1}{\sqrt{2}}[(\alpha|0\rangle+\beta e^{i\theta}|1\rangle)|0\rangle + (\beta|0\rangle+\alpha e^{i\theta}|1\rangle)|1\rangle] &\text{[Regroup]}\\
    M_z &= 0 \implies \alpha|0\rangle+\beta e^{i\theta}|1\rangle = R_z(\theta)\ket{\phi} &\text{[Target Met]}\\
    M_z &= 1 \implies \beta|0\rangle+\alpha e^{i\theta}|1\rangle &\text{[Error State]}\\
    |\phi'\rangle &= X (\beta|0\rangle+\alpha e^{i\theta}|1\rangle) \propto \alpha|0\rangle+\beta e^{-i\theta}|1\rangle &\text{[Correction 1]}\\
    |\phi''\rangle &= R_z(2\theta) (\alpha|0\rangle+\beta e^{-i\theta}|1\rangle) = \alpha|0\rangle+\beta e^{i\theta}|1\rangle = R_z(\theta)\ket{\phi} &\text{[Target Met]}
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.2
    2. When quantum gate applications are probabilistic or imperfect, it can be advisable to “store”
    a successful application of a gate on an external qubit resource for future seeding of a gate
    teleportation circuit. For CNOT gates, this strategy can trade the task of implementing the
    two-qubit gate for the task of creating and verifying an entangled state. Fill in the details
    of the discussion in Section 5.3 of KLM, and complete associated exercise 5.3.1 establishing
    the necessary classical logic for implementing a CNOT via a four-qubit entangled state and
    single-qubit operators classically controlled by measurements in the Bell basis.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > This response was generated by claude

    **CNOT gate teleportation (KLM Section 5.3 / Exercise 5.3.1):**

    **Resource state preparation.** Start with two Bell pairs: $|\Phi^+\rangle_{13} \otimes |\Phi^+\rangle_{24}$ where $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$. Apply a CNOT gate (offline, with verification) from qubit 1 to qubit 2 to create the 4-qubit resource state:

    $$|\chi\rangle_{1234} = \frac{1}{2}(|0000\rangle + |0110\rangle + |1011\rangle + |1101\rangle)$$

    This state encodes the action of the CNOT gate.

    **Teleportation protocol.** Given input qubits $|c\rangle$ (control) and $|t\rangle$ (target):

    1. Perform a Bell-basis measurement on $(|c\rangle, \text{qubit 1})$, obtaining outcomes $(i, j) \in \{0,1\}^2$ corresponding to $\sigma_Z^i \sigma_X^j |\Phi^+\rangle$.
    2. Perform a Bell-basis measurement on $(|t\rangle, \text{qubit 3})$, obtaining outcomes $(k, l) \in \{0,1\}^2$.
    3. The output state lives on qubits 2 and 4 of the resource.

    **Classical correction logic (Exercise 5.3.1).** The CNOT gate has the commutation properties $\text{CNOT} \cdot (Z_c \otimes I_t) = (Z_c \otimes I_t) \cdot \text{CNOT}$ and $\text{CNOT} \cdot (I_c \otimes X_t) = (I_c \otimes X_t) \cdot \text{CNOT}$, but $\text{CNOT} \cdot (X_c \otimes I_t) = (X_c \otimes X_t) \cdot \text{CNOT}$ and $\text{CNOT} \cdot (I_c \otimes Z_t) = (Z_c \otimes Z_t) \cdot \text{CNOT}$. Propagating the Pauli byproducts from both Bell measurements through the CNOT gives the corrections on the output qubits:

    - **Output qubit 2** (control): $X^{j} Z^{i \oplus k}$
    - **Output qubit 4** (target): $X^{j \oplus l} Z^{k}$

    **Significance.** This trades one online CNOT for offline Bell pair preparation + verification + four single-qubit Pauli corrections. In linear optical quantum computing (KLM), two-qubit gates are inherently probabilistic, so this allows "banking" successful gates as entangled resource states and consuming them deterministically via teleportation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.3
    3. Describe the relationship between the Pauli rotation $e^{-iZ^{\otimes n} t}$ and the parity (number of ones mod 2) of the quantum state in teh computational basis, i.e., basis in which Z is diagonal. From this perspective, analyze the following circuit
    """)
    return


@app.cell
def _(QuantumCircuit, np):
    _qc = QuantumCircuit(3)
    _qc.cx(0, 1)
    _qc.cx(1, 2)
    _qc.rz(np.pi / 4, 2)
    _qc.cx(1, 2)
    _qc.cx(0, 1)
    _qc.draw(output="mpl")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Describe the modifications, e.g., external basis changes, that produce circuits capable of implementing any Pauli rotation $e^{-i \hat P_v t}$ where $\hat P_v$ is a tensor product of Pauli operators.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > This response was generated by claude

    **Parity and $e^{-iZ^{\otimes n}t}$:**

    Since $Z|b\rangle = (-1)^b|b\rangle$, we have $Z^{\otimes n}|b_1\ldots b_n\rangle = (-1)^{b_1 \oplus \cdots \oplus b_n}|b_1\ldots b_n\rangle$, where $\oplus$ denotes addition mod 2 (parity). Therefore:

    $$e^{-iZ^{\otimes n}t}|b_1\ldots b_n\rangle = e^{-i(-1)^{\text{parity}}t}|b_1\ldots b_n\rangle$$

    Even-parity states acquire phase $e^{-it}$; odd-parity states acquire $e^{+it}$.

    **Circuit analysis:** The CNOT staircase $\text{CNOT}(0\to1)\cdot\text{CNOT}(1\to2)$ computes the cumulative parity $b_0 \oplus b_1 \oplus b_2$ onto qubit 2 (while preserving the computational basis state of qubits 0 and 1). The $R_z(\pi/4)$ gate on qubit 2 then applies a phase $e^{\pm i\pi/8}$ conditioned on this parity bit. The reverse CNOT staircase uncomputes the parity, restoring qubit 2 to its original value. The net effect is:

    $$|b_0 b_1 b_2\rangle \to e^{-i(\pi/8)(-1)^{b_0 \oplus b_1 \oplus b_2}}|b_0 b_1 b_2\rangle = e^{-i(\pi/8) Z \otimes Z \otimes Z}|b_0 b_1 b_2\rangle$$

    **Generalization to $e^{-i\hat{P}_v t}$:** For an arbitrary Pauli string $\hat{P}_v = P_1 \otimes P_2 \otimes \cdots \otimes P_n$ (where each $P_i \in \{I, X, Y, Z\}$), apply single-qubit basis changes $V_i$ before and $V_i^\dagger$ after the CNOT-$R_z$-CNOT structure:

    | Pauli $P_i$ | Basis change $V_i$ | Effect |
    |:-----------:|:------------------:|:------:|
    | $Z$ | $I$ (identity) | Already diagonal |
    | $X$ | $H$ (Hadamard) | $HXH = Z$ |
    | $Y$ | $R_x(\pi/2) = e^{-i\pi X/4}$ | Rotates $Y$ to $Z$ |
    | $I$ | Skip — exclude qubit from CNOT chain | No parity contribution |

    The CNOT staircase only threads through qubits with non-identity Paulis. This construction is the standard method for implementing Pauli rotations on quantum hardware and forms the backbone of Trotterized time evolution.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## IV. Toward Quantum Simulation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A. Measuring Purity

    Measuring the purity of a quantum state $Tr[\rho^2]$ may be a valuable indicator of experimental
    quality or the presence of subsystem entanglement. Because the quantity is quadratic in the density
    matrix, it cannot be connected to any single expectation value.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.1
     Tensor product Pauli operators, $\hat{P}_\bold{v} = \sigma_{v_1} \otimes\sigma_{v_2} \otimes \cdots \otimes \sigma_{v_n}$ with $\bold{v} \in \{\mathbb{I}_2, \sigma_x, \sigma_y, \sigma_z\}^n$, provide a complete basis for decomposing density matrices. By writing the structure of an arbitrary density matrix with respect to expectation values of operators in this basis, report how many distinct measurement ensembles would be needed to learn the density matrix (tomography) as a function of $n$, the number of qubits in the state.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **A.1 Solution — Measurement Ensembles for State Tomography**
    > [This response was written by claude]

    **The operator space.** The $n$-qubit density matrix $\rho$ can be decomposed in the Pauli basis:
    $$\rho = \frac{1}{2^n} \sum_{\mathbf{v} \in \{I,X,Y,Z\}^{\otimes n}} \text{Tr}[\hat{P}_\mathbf{v}\,\rho]\;\hat{P}_\mathbf{v}$$
    This is a $4^n$-dimensional operator space. Since $\text{Tr}[\rho] = 1$ fixes the all-identity coefficient, there are $4^n - 1$ independent real parameters to determine.

    **What is a measurement setting?** A measurement setting (ensemble) means choosing a Pauli basis ($X$, $Y$, or $Z$) for each qubit independently and then measuring all qubits. There is no setting corresponding to $I$ on a qubit — the identity is not a measurement; it means "don't measure" or equivalently "marginalize over outcomes."

    **Number of distinct measurement settings:**
    $$\boxed{3^n}$$

    *Example:* For $n=2$: the 9 settings are $XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ$.

    **Why $3^n$ settings suffice for $4^n - 1$ parameters.** Each setting produces shot-by-shot outcomes for every qubit. From these raw statistics, we can compute expectation values not only of the full Pauli string, but also of all partial marginals where some qubits are traced out (replaced by $I$). Concretely, a setting like $XZ$ on two qubits yields:

    | Observable | How to extract |
    |---|---|
    | $\langle XZ \rangle$ | Correlate both outcomes: $(\pm 1)(\pm 1)$ |
    | $\langle XI \rangle$ | Look at qubit 1 only, ignore qubit 2 |
    | $\langle IZ \rangle$ | Look at qubit 2 only, ignore qubit 1 |

    In general, each setting determines $2^n - 1$ non-trivial observables (each qubit either contributes its measured Pauli or is marginalized to $I$, minus the trivial $I^{\otimes n}$). Across all $3^n$ settings (with overlap in the marginals), every one of the $4^n - 1$ Pauli expectation values is covered by at least one setting.

    **Important subtlety:** It is the access to full outcome statistics that makes $3^n$ settings sufficient. If one only had $3^n$ bare expectation values $\langle P_\mathbf{v} \rangle$ for $\mathbf{v} \in \{X,Y,Z\}^{\otimes n}$ — without the ability to compute marginals — that would *not* be enough to reconstruct $\rho$. You cannot algebraically derive $\langle IX \rangle$ from $\langle XX \rangle$, $\langle YX \rangle$, $\langle ZX \rangle$ as numbers alone. It is the underlying measurement process (projecting each qubit onto eigenstates, yielding $6^n$ total projectors: 2 per qubit $\times$ 3 bases $\times$ $n$ qubits) that provides the additional information.

    **Comparison of countings:**

    | Quantity | Count | Meaning |
    |---|---|---|
    | Parameters of $\rho$ | $4^n - 1$ | Independent real numbers to specify the state |
    | Measurement settings | $3^n$ | Experimental configurations (basis choices) |
    | Projectors gathered | $6^n$ | Outcome projectors across all settings (overcomplete) |
    | Minimal IC-POVM | $4^n$ | Smallest possible informationally-complete measurement (e.g. SIC-POVM) |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.2
    There are many strategies for calculating the purity without performing full state tomography. For two copies of a single-qubit state, determine and physically describe the relationship between the expectation value $\langle \Psi^- | \rho \otimes \rho | \Psi^- \rangle$ and the purity of the state.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The **singlet state** $|\Psi^-\rangle = \frac{|01\rangle - |10\rangle}{\sqrt{2}}$ is the Bell state that is antisymmetric under particle exchange: $\text{SWAP}|\Psi^-\rangle = -|\Psi^-\rangle$ (the other three Bell states are symmetric, forming the "triplet"). The name comes from angular momentum coupling of two spin-$\frac{1}{2}$ particles into total spin 0.

    **Singlet projector.** Computing $|\Psi^-\rangle\langle\Psi^-|$ in the $\{|00\rangle,|01\rangle,|10\rangle,|11\rangle\}$ basis and comparing with the SWAP matrix:

    $$P_- \equiv |\Psi^-\rangle\langle\Psi^-| = \frac{1}{2}\begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 1 & -1 & 0 \\ 0 & -1 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix} = \frac{I - \text{SWAP}}{2}$$

    **SWAP-trace identity.** Expanding $\rho = \sum_{ij}\rho_{ij}|i\rangle\langle j|$, SWAP acts on the tensor product as $\text{SWAP}|ik\rangle = |ki\rangle$. Taking the trace picks out $\delta_{jk}\delta_{li}$, collapsing the sum:

    $$\text{Tr}[\text{SWAP}\,(\rho\otimes\rho)] = \sum_{i,j} \rho_{ij}\rho_{ji} = \text{Tr}[\rho^2]$$

    **Derivation.** Substituting $P_- = \frac{I - \text{SWAP}}{2}$ and using $\text{Tr}[\rho\otimes\rho] = (\text{Tr}[\rho])^2 = 1$:

    $$\langle \Psi^- | \rho\otimes\rho | \Psi^- \rangle = \text{Tr}\!\left[\frac{I - \text{SWAP}}{2}\,(\rho\otimes\rho)\right] = \frac{1}{2}\Big(\underbrace{\text{Tr}[\rho\otimes\rho]}_{1} - \underbrace{\text{Tr}[\text{SWAP}\,(\rho\otimes\rho)]}_{\text{Tr}[\rho^2]}\Big) = \frac{1 - \text{Tr}[\rho^2]}{2}$$

    $$\boxed{\text{Tr}[\rho^2] = 1 - 2\langle \Psi^- | \rho \otimes \rho | \Psi^- \rangle}$$

    **Physical picture.** Two identical copies of a pure state are perfectly symmetric under exchange, so they have zero overlap with the antisymmetric singlet — giving singlet fraction 0 and purity 1. A mixed state has nonzero weight in the antisymmetric sector, so the singlet fraction grows as purity decreases. Experimentally, this means you can measure purity by preparing two copies of $\rho$, performing a Bell measurement, and reading off the $|\Psi^-\rangle$ probability — no full tomography needed.
    """)
    return


@app.cell
def _(mo):
    mixing_slider = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.7, label="Mixing Parameter: |0> to |+>")
    return (mixing_slider,)


@app.cell
def _(mixing_slider, mo, np):
    # This response was written by claude
    # A.2: Numerically verify the relationship for a random single-qubit mixed state

    # Construct a random mixed state: rho = p|0><0| + (1-p)|+><+|
    _p_a2 = mixing_slider.value
    _rho0_a2 = np.array([[1,0],[0,0]], dtype=complex)  # |0><0| state
    _rhoP_a2 = np.array([[0.5,0.5],[0.5,0.5]], dtype=complex)  # |+><+| state
    _rho1_a2 = np.array([[0,0],[0,1]], dtype=complex)  # |1><1| state (not used but for reference)
    _rho_a2  = _p_a2 * _rho0_a2 + (1 - _p_a2) * _rhoP_a2  # rho = p|0><0| + (1-p)|+><+|

    _purity_a2 = np.real(np.trace(_rho_a2 @ _rho_a2))

    # Compute ⟨Ψ-|ρ⊗ρ|Ψ-⟩
    _psi_m_a2    = np.array([0, 1, -1, 0], dtype=complex)/np.sqrt(2)
    _rho_kron_a2 = np.kron(_rho_a2, _rho_a2)
    _singlet_frac = np.real(_psi_m_a2.conj() @ _rho_kron_a2 @ _psi_m_a2)


    # print(f"Tr[ρ²] (direct):               {_purity_a2:.6f}")
    # print(f"1 - 2⟨Ψ-|ρ⊗ρ|Ψ-⟩ (formula):  {1 - 2*_singlet_frac:.6f}")
    # print(f"Match: {np.isclose(_purity_a2, 1 - 2*_singlet_frac)}")
    mo.vstack(
        [
            mixing_slider,
            mo.hstack([
                mo.ui.matrix(_rho_a2.real, label=r"rho (real)", disabled=True),
                mo.ui.matrix(_rho_a2.imag, label=r"rho (imag)", disabled=True),
            ]),
            mo.md(f"**Purity Tr[ρ²] (direct):** {_purity_a2:.6f}"),
            mo.md(f"**1 - 2⟨Ψ-|ρ⊗ρ|Ψ-⟩ (formula):** {1 - 2*_singlet_frac:.6f}"),
            mo.md(f"**Match:** {'Yes' if np.isclose(_purity_a2, 1 - 2*_singlet_frac) else 'No'}"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.3
    For an arbitrary number of qubits, motivate the relationship $\text{Tr}[\rho^2] = \text{Tr}[(\rho \otimes \rho)\text{SWAP}]$. Using a basic interferometric quantum circuit structure (e.g., Hadamard test) for the non-destructive measurement of expectation values, show how the purity information may be calculated through the statistics of a single auxiliary qubit.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **A.3 Solution — SWAP Test and Hadamard Test for Purity**
    > This response was written by claude

    **Motivating $\text{Tr}[\rho^2] = \text{Tr}[(\rho \otimes \rho)\,\text{SWAP}]$:**

    In the computational basis, $\text{SWAP} = \sum_{ij} |ij\rangle\langle ji|$, so:
    $$\text{Tr}[(\rho \otimes \rho)\,\text{SWAP}] = \sum_{ij} \langle ij|(\rho \otimes \rho)|ji\rangle = \sum_{ij} \rho_{ij}\,\rho_{ji} = \text{Tr}[\rho^2]$$

    **Hadamard test circuit for purity:**

    Prepare $|0\rangle_{\text{aux}} \otimes \rho \otimes \rho$ and apply:
    1. $H$ on auxiliary qubit
    2. Controlled-SWAP (Fredkin gate): controlled on auxiliary
    3. $H$ on auxiliary qubit
    4. Measure auxiliary in $Z$-basis

    The measurement statistics give:
    $$P(+1) = \frac{1 + \text{Tr}[\rho^2]}{2} \qquad \Rightarrow \qquad \text{Tr}[\rho^2] = 2P(+1) - 1$$

    **Derivation:** The state after step 2 (controlled-SWAP) is $\frac{1}{\sqrt{2}}(|0\rangle \otimes \rho^{\otimes 2} + |1\rangle \otimes S\rho^{\otimes 2})$.
    After step 3 (second $H$), this becomes $\frac{1}{2}(|0\rangle(I+S)\rho^{\otimes 2} + |1\rangle(I-S)\rho^{\otimes 2})$.
    Then $P(|0\rangle) = \frac{1}{4}\text{Tr}[(I+S)\rho^{\otimes 2}(I+S)] = \frac{1}{2}(1 + \text{Tr}[S\,\rho^{\otimes 2}]) = \frac{1+\text{Tr}[\rho^2]}{2}$.
    """)
    return


@app.cell
def _(ClassicalRegister, QuantumCircuit, QuantumRegister):
    # This response was written by claude
    # A.3: Hadamard test circuit for purity of a 1-qubit state (2+1 = 3 qubits total)

    _qr_aux_a3 = QuantumRegister(1, 'aux')
    _qr_sys_a3 = QuantumRegister(2, 'sys')   # two copies of a 1-qubit state
    _cr_a3     = ClassicalRegister(1, 'meas')
    _qc_a3     = QuantumCircuit(_qr_aux_a3, _qr_sys_a3, _cr_a3)

    _qc_a3.h(_qr_aux_a3[0])
    _qc_a3.cswap(_qr_aux_a3[0], _qr_sys_a3[0], _qr_sys_a3[1])
    _qc_a3.h(_qr_aux_a3[0])
    _qc_a3.measure(_qr_aux_a3[0], _cr_a3[0])

    _qc_a3.draw(output='mpl')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### B. Projecting Spatial Parity

    Under the interchange of spatial coordinates, or equivalently the swapping of quantum states
    between Hilbert spaces, even-parity states are unchanged while odd-parity states accrue an overall
    minus sign. In the case of two spin-1/2 particles, these categories are also called “triplet” and
    “singlet” states, referring to the dimensionality of each within the four-dimensional Hilbert space.
    For such a two-qubit system, the positive parity (triplet) states are

    $$\ket{\phi_{t,1}} = \frac{|00\rangle + |11\rangle}{\sqrt{2}} \qquad \ket{\phi_{t,2}} = \frac{|00\rangle - |11\rangle}{\sqrt{2}} \qquad \ket{\phi_{t,3}} = \frac{|01\rangle + |10\rangle}{\sqrt{2}}$$
    and the negative parity (singlet) state is
    $$\ket{\phi_s} = \frac{|01\rangle - |10\rangle}{\sqrt{2}}$$

    You will recognize these as components of the Bell basis
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.1
    Show that under interchange of the two Hilbert spaces, the states have the symmetry prop-
    erties as advertised.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **B.1 Solution — Bell State Symmetry Under SWAP**
    > This response was written by claude

    The SWAP operator acts as $\text{SWAP}|ij\rangle = |ji\rangle$. Applying to each state:

    $$\text{SWAP}|\phi_{t,1}\rangle = \text{SWAP}\frac{|00\rangle+|11\rangle}{\sqrt{2}} = \frac{|00\rangle+|11\rangle}{\sqrt{2}} = +|\phi_{t,1}\rangle \quad \text{(symmetric, triplet)}$$

    $$\text{SWAP}|\phi_{t,2}\rangle = \text{SWAP}\frac{|00\rangle-|11\rangle}{\sqrt{2}} = \frac{|00\rangle-|11\rangle}{\sqrt{2}} = +|\phi_{t,2}\rangle \quad \text{(symmetric, triplet)}$$

    $$\text{SWAP}|\phi_{t,3}\rangle = \text{SWAP}\frac{|01\rangle+|10\rangle}{\sqrt{2}} = \frac{|10\rangle+|01\rangle}{\sqrt{2}} = +|\phi_{t,3}\rangle \quad \text{(symmetric, triplet)}$$

    $$\text{SWAP}|\phi_s\rangle = \text{SWAP}\frac{|01\rangle-|10\rangle}{\sqrt{2}} = \frac{|10\rangle-|01\rangle}{\sqrt{2}} = -|\phi_s\rangle \quad \text{(antisymmetric, singlet)}$$

    So $|\phi_{t,1}\rangle$, $|\phi_{t,2}\rangle$, $|\phi_{t,3}\rangle$ are **symmetric (triplet)** with SWAP eigenvalue $+1$,
    while $|\phi_s\rangle$ is the unique **antisymmetric (singlet)** state with eigenvalue $-1$.
    """)
    return


@app.cell
def _(np):
    # This response was written by claude
    # B.1: Numerically verify SWAP eigenvalues of Bell states

    _phi_p_b1 = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)
    _phi_m_b1 = np.array([1,0,0,-1], dtype=complex)/np.sqrt(2)
    _psi_p_b1 = np.array([0,1,1,0], dtype=complex)/np.sqrt(2)
    _psi_m_b1 = np.array([0,1,-1,0], dtype=complex)/np.sqrt(2)

    # SWAP matrix in computational basis {|00⟩,|01⟩,|10⟩,|11⟩}
    _SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)

    for _name, _state in [('Φ+',_phi_p_b1),('Φ-',_phi_m_b1),('Ψ+',_psi_p_b1),('Ψ-',_psi_m_b1)]:
        _ev = np.real(np.dot(_state.conj(), _SWAP @ _state))
        print(f"SWAP|{_name}⟩ eigenvalue: {_ev:+.1f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.2
    In class, we discuss the non-destructive partial projective measurement of bit-wise parity by in-
    troducing an auxiliary qubit. The bit-wise parity was “computed” onto the auxiliary qubit via
    Up, entangling the system and auxiliary Hilbert spaces. Subsequent measurement of the auxiliary
    qubit in state |0⟩/|1⟩ was shown to project the system into the 0/1 bit-wise parity subspace.
    What are the projectors of spatial parity, P±?

    - What are the projectors of spatial parity, $P_\pm$?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **B.2 Solution — Parity Projectors**
    > This response was written by claude

    **What is spatial parity?** The section intro distinguishes two kinds of parity:
    - **Bit-wise parity** (from class): counts the number of $|1\rangle$'s mod 2. E.g. $|01\rangle$ and $|10\rangle$ have odd bit-parity, while $|00\rangle$ and $|11\rangle$ have even bit-parity. The observable is $Z \otimes Z$.
    - **Spatial parity**: asks whether a state is symmetric or antisymmetric under *swapping the two particles* (i.e. exchanging which qubit lives in which Hilbert space). The observable is $\text{SWAP}$, which acts as $\text{SWAP}|ij\rangle = |ji\rangle$.

    For example, $|01\rangle + |10\rangle$ has odd *bit-wise* parity but even *spatial* parity (symmetric under swap), while $|01\rangle - |10\rangle$ has odd *bit-wise* parity and odd *spatial* parity (antisymmetric under swap). The two notions of parity are independent.

    **Constructing $P_\pm$:** The SWAP operator has eigenvalues $+1$ (symmetric subspace, 3-dimensional) and $-1$ (antisymmetric subspace, 1-dimensional).
    Since $\text{SWAP}^2 = I$, the projectors onto each eigenspace are:

    $$\boxed{P_+ = \frac{I + \text{SWAP}}{2}} \qquad \boxed{P_- = \frac{I - \text{SWAP}}{2}}$$

    where $P_+$ projects onto the **even-parity (symmetric/triplet)** subspace spanned by $\{|\phi_{t,1}\rangle, |\phi_{t,2}\rangle, |\phi_{t,3}\rangle\}$,
    and $P_-$ projects onto the **odd-parity (antisymmetric/singlet)** subspace $\{|\phi_s\rangle\}$.

    These satisfy $P_+^2 = P_+$, $P_-^2 = P_-$, $P_+P_- = 0$, and $P_+ + P_- = I$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.3
    - Devise a new quantum circuit, $U_p$, that computes the value of spatial parity onto an auxiliary qubit. Show that the effective system-space operators are equal to the parity projectors.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **B.3 Solution — Parity Circuit $U_p$**
    > This response was written by claude

    The parity circuit uses an auxiliary qubit and a controlled-SWAP (Fredkin) gate:

    $$U_p = H_{\text{aux}} \cdot \text{CSWAP} \cdot H_{\text{aux}}$$

    **Circuit:** $|0\rangle_{\text{aux}} \otimes |\psi\rangle \xrightarrow{H_{\text{aux}}} \frac{|0\rangle+|1\rangle}{\sqrt{2}} \otimes |\psi\rangle \xrightarrow{\text{CSWAP}} \xrightarrow{H_{\text{aux}}} \text{measure aux}$

    **Effective operators:** Tracing out the auxiliary qubit after measuring:
    - Outcome $|0\rangle_{\text{aux}}$: effective operator $= \frac{I + \text{SWAP}}{2} = P_+$
    - Outcome $|1\rangle_{\text{aux}}$: effective operator $= \frac{I - \text{SWAP}}{2} = P_-$

    **Proof:** The full unitary is
    $U_p = \frac{1}{2}\begin{pmatrix}I+S & I-S \\ I-S & I+S\end{pmatrix}$
    in the aux $\otimes$ system basis. Post-selecting on aux $= |0\rangle$ gives the unnormalized system state
    $\frac{I+S}{2}|\psi\rangle = P_+ |\psi\rangle$, and similarly $P_-$ for outcome $|1\rangle$.
    """)
    return


@app.cell
def _(ClassicalRegister, QuantumCircuit, QuantumRegister):
    # This response was written by claude
    # B.3: Draw the parity circuit U_p for a 2-qubit system

    _qr_sys = QuantumRegister(2, 'sys')
    _qr_aux = QuantumRegister(1, 'aux')
    _cr     = ClassicalRegister(1, 'meas')
    _qc_bp3 = QuantumCircuit(_qr_aux, _qr_sys, _cr)

    _qc_bp3.h(_qr_aux[0])
    _qc_bp3.cswap(_qr_aux[0], _qr_sys[0], _qr_sys[1])
    _qc_bp3.h(_qr_aux[0])
    _qc_bp3.measure(_qr_aux[0], _cr[0])

    _qc_bp3.draw(output='mpl')
    return


@app.cell
def _(QuantumCircuit, QuantumRegister, Statevector, mo):
    # B.3: Numerical validation — run the parity circuit on each Bell state


    def _run_parity(init_gates):
        """Return P(aux=0) and P(aux=1) via statevector simulation."""
        qr_aux = QuantumRegister(1, 'aux')
        qr_sys = QuantumRegister(2, 'sys')
        qc = QuantumCircuit(qr_aux, qr_sys)
        init_gates(qc, qr_sys)
        qc.barrier()
        qc.h(qr_aux[0])
        qc.cswap(qr_aux[0], qr_sys[0], qr_sys[1])
        qc.h(qr_aux[0])
        sv = Statevector(qc)
        probs = sv.probabilities_dict()
        # aux is qubit 0 = rightmost (last) character in Qiskit's big-endian bitstrings
        p0 = sum(v for k, v in probs.items() if k[-1] == '0')
        p1 = sum(v for k, v in probs.items() if k[-1] == '1')
        return p0, p1

    # Define Bell state preparations
    def _prep_phi_t1(qc, qr):  # (|00⟩+|11⟩)/√2
        qc.h(qr[0]); qc.cx(qr[0], qr[1])
    def _prep_phi_t2(qc, qr):  # (|00⟩-|11⟩)/√2
        qc.h(qr[0]); qc.cx(qr[0], qr[1]); qc.z(qr[0])
    def _prep_phi_t3(qc, qr):  # (|01⟩+|10⟩)/√2
        qc.h(qr[0]); qc.cx(qr[0], qr[1]); qc.x(qr[0])
    def _prep_phi_s(qc, qr):   # (|01⟩-|10⟩)/√2
        qc.h(qr[0]); qc.cx(qr[0], qr[1]); qc.x(qr[0]); qc.z(qr[0])

    _states = [
        (r"$|\phi_{t,1}\rangle = \frac{|00\rangle+|11\rangle}{\sqrt{2}}$", _prep_phi_t1, "triplet (+1)"),
        (r"$|\phi_{t,2}\rangle = \frac{|00\rangle-|11\rangle}{\sqrt{2}}$", _prep_phi_t2, "triplet (+1)"),
        (r"$|\phi_{t,3}\rangle = \frac{|01\rangle+|10\rangle}{\sqrt{2}}$", _prep_phi_t3, "triplet (+1)"),
        (r"$|\phi_s\rangle = \frac{|01\rangle-|10\rangle}{\sqrt{2}}$",     _prep_phi_s,  "singlet (-1)"),
    ]

    _rows = []
    for _label, _prep, _expected in _states:
        _p0, _p1 = _run_parity(_prep)
        _rows.append({
            "Input State": mo.md(_label),
            "Expected Parity": _expected,
            "P(aux=0) = P+": f"{_p0:.3f}",
            "P(aux=1) = P-": f"{_p1:.3f}",
            "Projected Into": "P+" if _p0 > 0.5 else "P-",
        })

    mo.vstack([
        mo.ui.table(_rows, label="**B.3 Numerical Validation:** Parity circuit on Bell states"),
        mo.md("Triplet states are projected entirely into P+ (aux=0), and the singlet into P- (aux=1), confirming the circuit implements spatial parity measurement."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### C. Time Evolution and Error Mitigation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.1
    1. **Quantum Zeno effect**: Consider a two-dimensional quantum degree of freedom in the state $|S^+_z\rangle = |0\rangle$ that is subject to external fields whose effects can be modeled at leading order as a Z-basis-flip Hamiltonian, $\hat{H} = \frac{X+Y}{\sqrt{2}}$, with unit coupling constant $U_{ext} = e^{-i\frac{X+Y}{\sqrt{2}}t}$. If you aimed to retain the $|0\rangle$ quantum state through repeated measurement in the Z-basis, how many (non-destructive) measurements would you have to perform in order to produce a probability of at least 99% that the state has remained in the vicinity of the $|0\rangle$ state throughout one unit of time evolution?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **C.1 Solution — Quantum Zeno Effect**
    > This response was written by claude

    Let $H = \frac{X+Y}{\sqrt{2}}$. Since $H^2 = \frac{(X+Y)^2}{2} = \frac{X^2 + XY + YX + Y^2}{2} = \frac{I + iZ - iZ + I}{2} = I$,
    the time evolution is:
    $$U(\Delta t) = e^{-iH\Delta t} = \cos(\Delta t)\,I - i\sin(\Delta t)\,H$$

    The survival probability for a single interval $\Delta t = T/n$ is:
    $$p(\Delta t) = |\langle 0| U(\Delta t) |0\rangle|^2 = |\cos(\Delta t)\underbrace{\langle 0|0\rangle}_{1} - i\sin(\Delta t)\underbrace{\langle 0|H|0\rangle}_{0}|^2 = \cos^2\!\left(\tfrac{1}{n}\right)$$
    The $\sin$ term vanishes because $H = \frac{X+Y}{\sqrt{2}}$ is purely off-diagonal in the $Z$-basis:
    $$\langle 0|H|0\rangle = \frac{1}{\sqrt{2}}(\langle 0|X|0\rangle + \langle 0|Y|0\rangle) = \frac{1}{\sqrt{2}}(\langle 0|1\rangle + (-i)\langle 0|1\rangle) = 0$$

    With $n$ measurements over $T = 1$:
    $$P_{\text{survive}} = \cos^{2n}\!\left(\tfrac{1}{n}\right) \geq 0.99$$

    The minimum $n$ is found numerically (see code below).
    """)
    return


@app.cell
def _(np, plt):
    # This response was written by claude
    # C.1: Quantum Zeno effect — find minimum n for 99% survival

    _n_vals_c1 = np.arange(1, 200)
    _P_survive_c1 = np.cos(1.0 / _n_vals_c1) ** (2 * _n_vals_c1)

    _min_n_c1 = int(_n_vals_c1[np.argmax(_P_survive_c1 >= 0.99)])
    print(f"Minimum n for P_survive ≥ 99%: n = {_min_n_c1}")
    print(f"P_survive at n={_min_n_c1}: {_P_survive_c1[_min_n_c1-1]:.5f}")

    _fig_c1, _ax_c1 = plt.subplots(figsize=(7, 4))
    _ax_c1.plot(_n_vals_c1, _P_survive_c1, 'b-')
    _ax_c1.axhline(0.99, color='r', linestyle='--', label='99% threshold')
    _ax_c1.axvline(_min_n_c1, color='g', linestyle=':', label=f'n = {_min_n_c1}')
    _ax_c1.set_xlabel('Number of measurements n')
    _ax_c1.set_ylabel('Survival probability')
    _ax_c1.set_title('Quantum Zeno Effect: Survival Probability vs. n')
    _ax_c1.legend(); _ax_c1.grid(True, alpha=0.3)
    plt.tight_layout(); 
    _fig_c1
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.2
    2. **Symmetry Identification**: Consider the slightly more interesting scenario of two such quantum degrees of freedom beginning in the Bell state $\frac{|00\rangle+|11\rangle}{\sqrt{2}}$ and subject to a unitary time evolution of the form $U_{\text{noiseless}} = e^{-i(X \otimes Y)t}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.2.a
       (a) Projected into the Bell basis, plot the evolution of probabilities for times up to $t = 2$.
    """)
    return


@app.cell
def _(np, plt):
    # This response was written by claude
    # C.2.a: Time evolution of Bell basis probabilities under U = e^{-i(X⊗Y)t}

    _X_c2a = np.array([[0,1],[1,0]], dtype=complex)
    _Y_c2a = np.array([[0,-1j],[1j,0]], dtype=complex)

    _H_c2a  = np.kron(_X_c2a, _Y_c2a)  # X⊗Y generator

    # Bell basis states
    _phi_p_c2a = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)
    _phi_m_c2a = np.array([1,0,0,-1], dtype=complex)/np.sqrt(2)
    _psi_p_c2a = np.array([0,1,1,0], dtype=complex)/np.sqrt(2)
    _psi_m_c2a = np.array([0,1,-1,0], dtype=complex)/np.sqrt(2)
    _bells_c2a   = [_phi_p_c2a, _phi_m_c2a, _psi_p_c2a, _psi_m_c2a]
    _blabels_c2a = ['|Φ+⟩', '|Φ-⟩', '|Ψ+⟩', '|Ψ-⟩']

    def _mat_exp_c2a(H, t):
        # e^{-iHt} via eigendecomposition (H hermitian)
        evals, evecs = np.linalg.eigh(H)
        return evecs @ np.diag(np.exp(-1j * evals * t)) @ evecs.conj().T

    _t_arr_c2a = np.linspace(0, 2.0, 400)
    # Evolve |Φ+⟩ and project onto Bell basis
    _probs_c2a = np.array([
        [abs(float(np.dot(b.conj(), _mat_exp_c2a(_H_c2a, t) @ _phi_p_c2a)))**2 for b in _bells_c2a]
        for t in _t_arr_c2a
    ])

    _fig_c2a, _ax_c2a = plt.subplots(figsize=(8, 5))
    for _k, (_lab, _col) in enumerate(zip(_blabels_c2a, ['C0','C1','C2','C3'])):
        _ax_c2a.plot(_t_arr_c2a, _probs_c2a[:, _k], label=_lab, color=_col)
    _ax_c2a.set_title(r'Bell Basis Probabilities: $U(t)|\Phi^+\rangle$ under $e^{-i(X\otimes Y)t}$')
    _ax_c2a.set_xlabel('Time t'); _ax_c2a.set_ylabel('Probability')
    _ax_c2a.legend(); _ax_c2a.grid(True, alpha=0.3); _ax_c2a.set_ylim(-0.05, 1.05)
    _fig_c2a
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.2.b
       (b) Identify an operator (e.g., parity and/or built from Bell basis projectors) whose expectation values distinguish the populated states from those that remain unpopulated.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **C.2.b Solution — Distinguishing Operator**
    > This response was written by claude

    The populated states are $|\Phi^\pm\rangle = \frac{|00\rangle \pm |11\rangle}{\sqrt{2}}$ (same-spin pairs),
    while the unpopulated states are $|\Psi^\pm\rangle = \frac{|01\rangle \pm |10\rangle}{\sqrt{2}}$ (opposite-spin pairs).

    The operator $\hat{O} = Z \otimes Z$ distinguishes them:
    $$\langle \Phi^\pm | Z \otimes Z | \Phi^\pm \rangle = +1$$
    $$\langle \Psi^\pm | Z \otimes Z | \Psi^\pm \rangle = -1$$

    This is the **parity operator** for the $Z$-basis correlations. It equals $+1$ in the populated subspace
    and $-1$ in the unpopulated subspace. Equivalently:
    $$\hat{O} = |\Phi^+\rangle\langle\Phi^+| + |\Phi^-\rangle\langle\Phi^-| - |\Psi^+\rangle\langle\Psi^+| - |\Psi^-\rangle\langle\Psi^-|$$
    """)
    return


@app.cell
def _(np, plt):
    # This response was written by claude
    # C.2.b: Verify Z⊗Z eigenvalues and plot <Z⊗Z>(t) = 1 under noiseless evolution

    _X_c2b = np.array([[0,1],[1,0]], dtype=complex)
    _Y_c2b = np.array([[0,-1j],[1j,0]], dtype=complex)
    _Z_c2b = np.array([[1,0],[0,-1]], dtype=complex)

    _phi_p_c2b = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)
    _phi_m_c2b = np.array([1,0,0,-1], dtype=complex)/np.sqrt(2)
    _psi_p_c2b = np.array([0,1,1,0], dtype=complex)/np.sqrt(2)
    _psi_m_c2b = np.array([0,1,-1,0], dtype=complex)/np.sqrt(2)

    _ZZ_c2b = np.kron(_Z_c2b, _Z_c2b)
    _H_c2b  = np.kron(_X_c2b, _Y_c2b)

    print("⟨Φ+|Z⊗Z|Φ+⟩ =", np.real(_phi_p_c2b.conj() @ _ZZ_c2b @ _phi_p_c2b))
    print("⟨Φ-|Z⊗Z|Φ-⟩ =", np.real(_phi_m_c2b.conj() @ _ZZ_c2b @ _phi_m_c2b))
    print("⟨Ψ+|Z⊗Z|Ψ+⟩ =", np.real(_psi_p_c2b.conj() @ _ZZ_c2b @ _psi_p_c2b))
    print("⟨Ψ-|Z⊗Z|Ψ-⟩ =", np.real(_psi_m_c2b.conj() @ _ZZ_c2b @ _psi_m_c2b))

    def _mat_exp_c2b(H, t):
        evals, evecs = np.linalg.eigh(H)
        return evecs @ np.diag(np.exp(-1j * evals * t)) @ evecs.conj().T

    _t_arr_c2b = np.linspace(0, 2.0, 300)
    _exp_zz = np.array([
        np.real((_mat_exp_c2b(_H_c2b, t) @ _phi_p_c2b).conj() @ _ZZ_c2b @ (_mat_exp_c2b(_H_c2b, t) @ _phi_p_c2b))
        for t in _t_arr_c2b
    ])

    _fig_c2b, _ax_c2b = plt.subplots(figsize=(7, 4))
    _ax_c2b.plot(_t_arr_c2b, _exp_zz, 'b-')
    _ax_c2b.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='⟨Z⊗Z⟩ = +1')
    _ax_c2b.set_title('⟨Z⊗Z⟩(t) under noiseless evolution (conserved)')
    _ax_c2b.set_xlabel('Time t'); _ax_c2b.set_ylabel('⟨Z⊗Z⟩')
    _ax_c2b.set_ylim(-0.2, 1.5); _ax_c2b.legend(); _ax_c2b.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
    # _fig_c2b
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.2.c
       (c) Show that this operator commutes with the Hamiltonian, and thus presents a static symmetry under time evolution.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **C.2.c Solution — Commutation Proof**
    > This response was written by claude

    We show $[Z \otimes Z,\ X \otimes Y] = 0$.

    Using $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$:
    $$(Z \otimes Z)(X \otimes Y) = (ZX) \otimes (ZY)$$
    $$(X \otimes Y)(Z \otimes Z) = (XZ) \otimes (YZ)$$

    Using the single-qubit anticommutation relations:
    $$ZX = iY,\quad XZ = -iY \quad\Rightarrow\quad ZX = -XZ$$
    $$ZY = -iX,\quad YZ = iX \quad\Rightarrow\quad ZY = -YZ$$

    Therefore:
    $$(ZX)\otimes(ZY) = (iY)\otimes(-iX) = Y \otimes X$$
    $$(XZ)\otimes(YZ) = (-iY)\otimes(iX) = Y \otimes X$$

    Both products are equal, so $[Z \otimes Z,\ X \otimes Y] = 0$. $\blacksquare$

    Since $Z \otimes Z$ commutes with the generator $H = X \otimes Y$, it commutes with $U(t) = e^{-iHt}$ for all $t$,
    making it a **conserved quantity** (static symmetry).
    """)
    return


@app.cell
def _(np):
    # This response was written by claude
    # C.2.c: Numerically verify [Z⊗Z, X⊗Y] = 0

    _X_c2c = np.array([[0,1],[1,0]], dtype=complex)
    _Y_c2c = np.array([[0,-1j],[1j,0]], dtype=complex)
    _Z_c2c = np.array([[1,0],[0,-1]], dtype=complex)

    _H_c2c  = np.kron(_X_c2c, _Y_c2c)
    _ZZ_c2c = np.kron(_Z_c2c, _Z_c2c)
    _comm   = _ZZ_c2c @ _H_c2c - _H_c2c @ _ZZ_c2c

    print("Commutator [Z⊗Z, X⊗Y] is zero:", np.allclose(_comm, 0))
    print("Max absolute entry:", np.max(np.abs(_comm)))
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.3
    3. **Coherent Errors: Zeno and Basis Randomization**: Consider the presence of single-qubit errors such that the time evolution becomes $U_{\text{noisy}} = e^{-i\left(X\otimes Y+\mathbb{I}\otimes\frac{X+Y}{\sqrt{2}}+\frac{X+Y}{\sqrt{2}}\otimes\mathbb{I}\right)t}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.3.a
       (a) Plot the new evolution of probabilities projected into the Bell basis. For how long do the probabilities remain within $\pm 1\%$ of their noiseless values?
    """)
    return


@app.cell
def _(np, plt):
    # This response was written by claude
    # C.3.a: Noisy time evolution vs noiseless, Bell basis probabilities

    _I2_c3a = np.eye(2, dtype=complex)
    _X_c3a  = np.array([[0,1],[1,0]], dtype=complex)
    _Y_c3a  = np.array([[0,-1j],[1j,0]], dtype=complex)

    _H_nl_c3a = np.kron(_X_c3a, _Y_c3a)
    _H_ny_c3a = (np.kron(_X_c3a, _Y_c3a)
                 + np.kron(_I2_c3a, (_X_c3a + _Y_c3a)/np.sqrt(2))
                 + np.kron((_X_c3a + _Y_c3a)/np.sqrt(2), _I2_c3a))

    _phi_p_c3a = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)
    _phi_m_c3a = np.array([1,0,0,-1], dtype=complex)/np.sqrt(2)
    _psi_p_c3a = np.array([0,1,1,0], dtype=complex)/np.sqrt(2)
    _psi_m_c3a = np.array([0,1,-1,0], dtype=complex)/np.sqrt(2)
    _bells_c3a   = [_phi_p_c3a, _phi_m_c3a, _psi_p_c3a, _psi_m_c3a]
    _blabels_c3a = ['Φ+', 'Φ-', 'Ψ+', 'Ψ-']

    def _mat_exp_c3a(H, t):
        evals, evecs = np.linalg.eigh(H)
        return evecs @ np.diag(np.exp(-1j * evals * t)) @ evecs.conj().T

    def _bell_probs_c3a(state):
        return [abs(float(np.dot(b.conj(), state)))**2 for b in _bells_c3a]

    _t_arr_c3a = np.linspace(0, 2.0, 400)
    _nl_c3a = np.array([_bell_probs_c3a(_mat_exp_c3a(_H_nl_c3a, t) @ _phi_p_c3a) for t in _t_arr_c3a])
    _ny_c3a = np.array([_bell_probs_c3a(_mat_exp_c3a(_H_ny_c3a, t) @ _phi_p_c3a) for t in _t_arr_c3a])

    _devs_c3a = np.max(np.abs(_ny_c3a - _nl_c3a), axis=1)
    _exceed_idx = np.argmax(_devs_c3a > 0.01)
    _t_thresh = float(_t_arr_c3a[_exceed_idx]) if _exceed_idx > 0 else None
    if _t_thresh:
        print(f"Probabilities stay within ±1% until t ≈ {_t_thresh:.3f}")
    else:
        print("Always within ±1%")

    _colors_c3a = ['C0','C1','C2','C3']
    _fig_c3a, _axes_c3a = plt.subplots(1, 2, figsize=(14, 5))
    for _k in range(4):
        _axes_c3a[0].plot(_t_arr_c3a, _nl_c3a[:, _k], '--', color=_colors_c3a[_k], alpha=0.7)
        _axes_c3a[0].plot(_t_arr_c3a, _ny_c3a[:, _k], '-',  color=_colors_c3a[_k], label=_blabels_c3a[_k])
    if _t_thresh:
        _axes_c3a[0].axvline(_t_thresh, color='k', linestyle=':', label=f't≈{_t_thresh:.2f}')
    _axes_c3a[0].set_title('Bell Probabilities: Noiseless (dashed) vs Noisy (solid)')
    _axes_c3a[0].set_xlabel('Time t'); _axes_c3a[0].set_ylabel('Probability')
    _axes_c3a[0].legend(fontsize=8); _axes_c3a[0].grid(True, alpha=0.3)

    _axes_c3a[1].plot(_t_arr_c3a, _devs_c3a, 'k-')
    _axes_c3a[1].axhline(0.01, color='r', linestyle='--', label='1% threshold')
    if _t_thresh:
        _axes_c3a[1].axvline(_t_thresh, color='b', linestyle=':', label=f't≈{_t_thresh:.2f}')
    _axes_c3a[1].set_title('Max Deviation from Noiseless')
    _axes_c3a[1].set_xlabel('Time t'); _axes_c3a[1].set_ylabel('Max |P_noisy − P_noiseless|')
    _axes_c3a[1].legend(); _axes_c3a[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
    _fig_c3a
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.3.b
       (b) Consider the procedure of periodically projectively measuring (non-destructively) the operator whose expectation values identify the positive-parity subspace, and post-selecting upon the results.
       - Describe experimentally how such measurements could be performed, either abstractly or in a quantum system of your choice.
       - Show that, at the cost of measurement statistics, this procedure can improve the fidelity of the noisy simulation. How many such measurements would be required to maintain probabilities within $\pm 1\%$ of their noiseless values to a simulation time of $t = 1.5$ with a loss of no more than 20% measurement statistics?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **C.3.b Solution — Parity Projection Post-Selection**
    > This response was written by claude

    The $Z \otimes Z$ operator has eigenvalue $+1$ on $|\Phi^\pm\rangle$ and $-1$ on $|\Psi^\pm\rangle$.
    By periodically measuring $Z \otimes Z$ and post-selecting on $+1$ outcomes, we project the state back
    into the $\{|\Phi^+\rangle, |\Phi^-\rangle\}$ subspace, removing coherent error leakage into $|\Psi\rangle$ states.

    **Experimental implementation:** Couple the two system qubits to an ancilla via controlled-Z gates
    (e.g., in superconducting transmon architectures where ZZ-parity checks are standard syndrome extraction primitives),
    measure the ancilla, and post-select on $+1$.
    This is non-destructive — it reveals the parity sector without distinguishing $|\Phi^+\rangle$ from $|\Phi^-\rangle$.

    > *This paragraph was generated by claude*

    **Measurement count analysis.** Numerically, no measurement count simultaneously achieves $\pm 1\%$ maximum deviation from noiseless probabilities **and** $\geq 80\%$ post-selection survival out to $t = 1.5$. Increasing the number of parity measurements improves fidelity but reduces survival below the 80% threshold — these two constraints are in tension. The best tradeoff (minimizing deviation) occurs around $n \approx 24$ measurements, achieving $\sim 5.7\%$ max deviation with $\sim 85\%$ survival. This highlights the fundamental cost of post-selection-based error mitigation: the stricter the fidelity requirement, the more events must be discarded.
    """)
    return


@app.cell
def _(np, plt):
    # This response was written by claude
    # C.3.b: Parity projection (Z⊗Z = +1 post-selection) error mitigation

    _I2_c3b = np.eye(2, dtype=complex)
    _X_c3b  = np.array([[0,1],[1,0]], dtype=complex)
    _Y_c3b  = np.array([[0,-1j],[1j,0]], dtype=complex)
    _Z_c3b  = np.array([[1,0],[0,-1]], dtype=complex)

    _H_nl_c3b = np.kron(_X_c3b, _Y_c3b)
    _H_ny_c3b = (np.kron(_X_c3b, _Y_c3b)
                 + np.kron(_I2_c3b, (_X_c3b + _Y_c3b)/np.sqrt(2))
                 + np.kron((_X_c3b + _Y_c3b)/np.sqrt(2), _I2_c3b))

    _phi_p_c3b = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)
    _phi_m_c3b = np.array([1,0,0,-1], dtype=complex)/np.sqrt(2)
    _psi_p_c3b = np.array([0,1,1,0], dtype=complex)/np.sqrt(2)
    _psi_m_c3b = np.array([0,1,-1,0], dtype=complex)/np.sqrt(2)
    _bells_c3b  = [_phi_p_c3b, _phi_m_c3b, _psi_p_c3b, _psi_m_c3b]
    _blabels_c3b = ['Φ+', 'Φ-', 'Ψ+', 'Ψ-']

    def _mat_exp_c3b(H, t):
        evals, evecs = np.linalg.eigh(H)
        return evecs @ np.diag(np.exp(-1j * evals * t)) @ evecs.conj().T

    # Z⊗Z projector onto +1 eigenspace
    _ZZ_c3b = np.kron(_Z_c3b, _Z_c3b)
    _evals_zz, _evecs_zz = np.linalg.eigh(_ZZ_c3b)
    _P_plus_c3b = sum(
        np.outer(_evecs_zz[:, i], _evecs_zz[:, i].conj())
        for i in range(4) if np.isclose(_evals_zz[i], 1.0)
    )

    def _parity_project_c3b(rho):
        rho_proj = _P_plus_c3b @ rho @ _P_plus_c3b.conj().T
        prob = max(float(np.real(np.trace(rho_proj))), 1e-12)
        return rho_proj / prob, prob

    def _bell_probs_dm_c3b(rho):
        return [max(0.0, float(np.real(b.conj() @ rho @ b))) for b in _bells_c3b]

    _t_arr_c3b = np.linspace(0, 1.5, 300)
    _rho0_c3b  = np.outer(_phi_p_c3b, _phi_p_c3b.conj())

    # Noiseless reference
    _nl_c3b = np.array([
        _bell_probs_dm_c3b(_mat_exp_c3b(_H_nl_c3b, t) @ _rho0_c3b @ _mat_exp_c3b(_H_nl_c3b, t).conj().T)
        for t in _t_arr_c3b
    ])

    def _sim_parity_zeno(n_meas):
        rho = _rho0_c3b.copy()
        meas_times = np.linspace(0, 1.5, n_meas + 2)[1:-1]
        probs_traj = []
        total_surv = 1.0
        t_prev = 0.0
        mi = 0
        for t in _t_arr_c3b:
            while mi < len(meas_times) and t >= meas_times[mi]:
                dt = meas_times[mi] - t_prev
                U = _mat_exp_c3b(_H_ny_c3b, dt)
                rho = U @ rho @ U.conj().T
                rho, surv = _parity_project_c3b(rho)
                total_surv *= surv
                t_prev = meas_times[mi]
                mi += 1
            probs_traj.append(_bell_probs_dm_c3b(rho))
        return np.array(probs_traj), total_surv

    _results_c3b = {}
    for _n in range(0, 25):
        _pr, _sv = _sim_parity_zeno(_n)
        _dev = float(np.max(np.abs(_pr - _nl_c3b)))
        _results_c3b[_n] = (_dev, _sv, _pr)

    print("n_meas | max_deviation | survival")
    for _n, (_d, _s, _) in _results_c3b.items():
        _marker = " <--" if _d <= 0.01 and _s >= 0.80 else ""
        print(f"  {_n:2d}   |    {_d:.4f}     |   {_s:.4f}{_marker}")

    _valid_c3b = [_n for _n, (_d, _s, _) in _results_c3b.items() if _d <= 0.01 and _s >= 0.80]
    _n_opt_c3b = _valid_c3b[0] if _valid_c3b else min(_results_c3b, key=lambda n: _results_c3b[n][0])
    _pr_opt_c3b, _sv_opt_c3b = _results_c3b[_n_opt_c3b][2], _results_c3b[_n_opt_c3b][1]
    print(f"\nOptimal: n_meas={_n_opt_c3b}, max_dev={_results_c3b[_n_opt_c3b][0]:.4f}, survival={_sv_opt_c3b:.4f}")

    _colors_c3b = ['C0','C1','C2','C3']
    _fig_c3b, _axes_c3b = plt.subplots(1, 2, figsize=(14, 5))
    for _k in range(4):
        _axes_c3b[0].plot(_t_arr_c3b, _nl_c3b[:, _k], '--', color=_colors_c3b[_k], alpha=0.6)
        _axes_c3b[0].plot(_t_arr_c3b, _pr_opt_c3b[:, _k], '-', color=_colors_c3b[_k], label=_blabels_c3b[_k])
    _axes_c3b[0].set_title(f'Parity Projection: n_meas={_n_opt_c3b}, survival={_sv_opt_c3b:.3f}')
    _axes_c3b[0].set_xlabel('Time t'); _axes_c3b[0].set_ylabel('Probability')
    _axes_c3b[0].legend(fontsize=8); _axes_c3b[0].grid(True, alpha=0.3)

    _n_vals_c3b = list(_results_c3b.keys())
    _devs_c3b = [_results_c3b[n][0] for n in _n_vals_c3b]
    _survs_c3b = [_results_c3b[n][1] for n in _n_vals_c3b]
    _axes_c3b[1].plot(_n_vals_c3b, _devs_c3b, 'b-o', label='Max deviation')
    _axes_c3b[1].axhline(0.01, color='b', linestyle='--', alpha=0.5, label='±1% threshold')
    _ax2r_c3b = _axes_c3b[1].twinx()
    _ax2r_c3b.plot(_n_vals_c3b, _survs_c3b, 'r-s', label='Survival prob')
    _ax2r_c3b.axhline(0.80, color='r', linestyle='--', alpha=0.5, label='80% threshold')
    _ax2r_c3b.set_ylabel('Survival Probability', color='r')
    _axes_c3b[1].set_xlabel('Number of measurements'); _axes_c3b[1].set_ylabel('Max deviation', color='b')
    _axes_c3b[1].set_title('Fidelity vs. Measurement Cost')
    _axes_c3b[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
    _fig_c3b
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.3.c
       (c) Consider an alternate error mitigation technique, *twirling*, that introduces a dynamic randomization of Pauli bases throughout the time evolution to turn coherent noise into incoherent channels, in some cases destructively interfering coherent noise entirely.
       - Viable basis transformations of the noiseless operators leave invariant the logical circuits of time evolution. Find all Pauli operators, within the set of 16 $P_j = \{\mathbb{I}_2, X, Y, Z\} \otimes \{\mathbb{I}_2, X, Y, Z\}$, that satisfy the criteria $U_{\text{noiseless}} = P^\dagger U_{\text{noiseless}} P$.
       - Devise a stochastic strategy of these invariant basis changes that maintains, up to $t = 1.5$, the Bell basis measurement probabilities within $\pm 10\%$. How many layers of basis changes were used? Is the utilization of all basis transformations necessary to achieve this goal? (*Hint: while probabilistic numerical simulation could be used, try utilizing density matrix methods to capture the stochastic features of the procedure*)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **C.3.c Solution — Twirling**
    > This response was written by claude

    **Finding invariant Paulis:** We need $[P, X \otimes Y] = 0$ (P commutes with the generator). For $P = A \otimes B$:
    $$[A \otimes B,\ X \otimes Y] = (AX)\otimes(BY) - (XA)\otimes(YB) = 0$$
    This vanishes iff either:
    - $A$ **commutes** with $X$ ($A \in \{I, X\}$) **and** $B$ **commutes** with $Y$ ($B \in \{I, Y\}$)
    - $A$ **anticommutes** with $X$ ($A \in \{Y, Z\}$) **and** $B$ **anticommutes** with $Y$ ($B \in \{X, Z\}$)

    The **8 invariant operators** are: $II,\ IY,\ XI,\ XY,\ YX,\ YZ,\ ZX,\ ZZ$.

    **Twirling strategy:** Uniformly sample from the invariant group. In the density matrix picture:
    $$\rho \to \frac{1}{|G|}\sum_{P \in G} P\,\rho\, P^\dagger$$
    This averages coherent noise over the group, destructively interfering error components that do not commute with all group elements, while the noiseless signal (which commutes) is preserved.
    """)
    return


@app.cell
def _(np, plt):
    # This response was written by claude
    # C.3.c: Find invariant Paulis and simulate twirling via density matrix

    _I2 = np.eye(2, dtype=complex)
    _X  = np.array([[0,1],[1,0]], dtype=complex)
    _Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
    _Z  = np.array([[1,0],[0,-1]], dtype=complex)
    _paulis_1q = [_I2, _X, _Y, _Z]
    _labels_1q = ['I', 'X', 'Y', 'Z']

    _H_noiseless_c3c = np.kron(_X, _Y)
    _H_noisy_c3c = (np.kron(_X, _Y)
                    + np.kron(_I2, (_X + _Y)/np.sqrt(2))
                    + np.kron((_X + _Y)/np.sqrt(2), _I2))

    # Find all Paulis P = A⊗B that commute with X⊗Y
    _invariant, _inv_labels = [], []
    for _i, _A in enumerate(_paulis_1q):
        for _j, _B in enumerate(_paulis_1q):
            _P = np.kron(_A, _B)
            if np.allclose(_P @ _H_noiseless_c3c - _H_noiseless_c3c @ _P, 0):
                _invariant.append(_P)
                _inv_labels.append(_labels_1q[_i] + _labels_1q[_j])
    print("Invariant Paulis:", _inv_labels)

    # Bell basis states
    _phi_p = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)
    _phi_m = np.array([1,0,0,-1], dtype=complex)/np.sqrt(2)
    _psi_p = np.array([0,1,1,0], dtype=complex)/np.sqrt(2)
    _psi_m = np.array([0,1,-1,0], dtype=complex)/np.sqrt(2)
    _bells_c3c  = [_phi_p, _phi_m, _psi_p, _psi_m]
    _blabels_c3c = ['Φ+', 'Φ-', 'Ψ+', 'Ψ-']

    def _mat_exp_c3c(H, t):
        evals, evecs = np.linalg.eigh(H)
        return evecs @ np.diag(np.exp(-1j * evals * t)) @ evecs.conj().T

    def _bell_probs_dm_c3c(rho):
        return [max(0.0, float(np.real(b.conj() @ rho @ b))) for b in _bells_c3c]

    def _twirl_c3c(rho, group):
        out = np.zeros_like(rho)
        for P in group:
            out += P @ rho @ P.conj().T
        return out / len(group)

    _t_arr_c3c = np.linspace(0, 1.5, 300)
    _rho0_c3c  = np.outer(_phi_p, _phi_p.conj())

    # Noiseless reference
    _nl_probs_c3c = np.array([
        _bell_probs_dm_c3c(_mat_exp_c3c(_H_noiseless_c3c, t) @ _rho0_c3c @ _mat_exp_c3c(_H_noiseless_c3c, t).conj().T)
        for t in _t_arr_c3c
    ])

    # Twirled trajectory: apply twirl channel at n_layers uniform intervals
    def _twirled_traj(n_layers):
        rho = _rho0_c3c.copy()
        twirl_t = np.linspace(0, _t_arr_c3c[-1], n_layers + 1)
        probs = []
        seg = 0
        for t in _t_arr_c3c:
            while seg < n_layers and t >= twirl_t[seg + 1]:
                dt = twirl_t[seg + 1] - twirl_t[seg]
                U = _mat_exp_c3c(_H_noisy_c3c, dt)
                rho = U @ rho @ U.conj().T
                rho = _twirl_c3c(rho, _invariant)
                seg += 1
            probs.append(_bell_probs_dm_c3c(rho))
        return np.array(probs)

    _colors_c3c = ['C0','C1','C2','C3']
    _fig_c3c, _axes_c3c = plt.subplots(1, 2, figsize=(14, 5))
    for _n_layers, _ax in zip([5, 10], _axes_c3c):
        _tw = _twirled_traj(_n_layers)
        for _k in range(4):
            _ax.plot(_t_arr_c3c, _nl_probs_c3c[:, _k], '--', color=_colors_c3c[_k], alpha=0.5)
            _ax.plot(_t_arr_c3c, _tw[:, _k], '-', color=_colors_c3c[_k], label=_blabels_c3c[_k])
            _ax.fill_between(_t_arr_c3c, _nl_probs_c3c[:,_k]-0.10, _nl_probs_c3c[:,_k]+0.10,
                             alpha=0.07, color=_colors_c3c[_k])
        _ax.set_title(f'Twirled Evolution ({_n_layers} layers, dashed=noiseless)')
        _ax.set_xlabel('Time t'); _ax.set_ylabel('Probability')
        _ax.legend(fontsize=8); _ax.grid(True, alpha=0.3); _ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()
    _fig_c3c
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## V. Nature of Mixed States
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A. Quantum Sabotage

    In class, we discuss the close relationship between unitarily establishing entanglement with systems that become experimentally inaccessible and errors/noise in quantum computational endeavors. In this exercise, you will have a chance to explore this relationship in further detail.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.1
    1. We show that an amplitude damping channel, $\mathcal{E}_{AD}(\rho)$, models spontaneous decay from the $|1\rangle$ state to the $|0\rangle$ state. When measuring the environment in the Z-basis, this channel can be decomposed into a set of operators,
       $$\rho' = \mathcal{E}_{AD}(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger \qquad E_0 = \begin{pmatrix}1 & 0 \\ 0 & \sqrt{1-p}\end{pmatrix} \qquad E_1 = \begin{pmatrix}0 & \sqrt{p} \\ 0 & 0\end{pmatrix}$$
       This *channel* represents the non-unitary interaction seen from the perspective of the qubit system alone. Choose a few pure states on the surface of the Bloch sphere and plot their trajectories as a function of time (or decay time steps) when subject to this type of environment interaction.
    """)
    return


@app.cell
def _(Statevector, mo, np, plot_bloch_multivector):
    # Choose |1>, |+>, S|+> as our three states (amplitude dampening)
    ad_psi_1 = np.array([0, 1])  # |1>
    ad_psi_2 = 1 / np.sqrt(2) * np.array([1, 1])  # |+>
    ad_psi_3 = 1 / np.sqrt(2) * np.array([1, 1j])  # S|+>

    # First plot each on the bloch sphere
    def plot_state_on_bloch(state, label): # with qiskit
        sv = Statevector(state)
        fig = plot_bloch_multivector(sv, title=label)
        return fig
    mo.hstack(
        [
            plot_state_on_bloch(ad_psi_1, "|1⟩"),
            plot_state_on_bloch(ad_psi_2, "|+⟩"),
            plot_state_on_bloch(ad_psi_3, "S|+⟩"),
        ]
    )
    return ad_psi_1, ad_psi_2, ad_psi_3


@app.cell
def _(ad_psi_1, ad_psi_2, ad_psi_3, np, plt):
    # Plot P(1) for measuring in the |0>, |1> basis over repeated applications of the amplitude damping channel
    def amplitude_damping_channel(rho, gamma):
        """Apply amplitude damping channel to density matrix rho with damping probability gamma."""
        E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        return E0 @ rho @ E0.T.conj() + E1 @ rho @ E1.T.conj()

    # Track P(1) over repeated applications for the three states
    gamma = 0.1  # damping probability per step
    num_steps = 50

    def track_p1_trajectory(initial_state, num_steps, gamma):
        """Track probability of measuring |1> as amplitude damping channel is repeatedly applied."""
        rho = np.outer(initial_state, initial_state.conj())
        p1_values = []
        for step in range(num_steps):
            p1 = rho[1, 1].real
            p1_values.append(p1)
            rho = amplitude_damping_channel(rho, gamma)
        return np.array(p1_values)

    p1_psi1 = track_p1_trajectory(ad_psi_1, num_steps, gamma)
    p1_psi2 = track_p1_trajectory(ad_psi_2, num_steps, gamma)
    p1_psi3 = track_p1_trajectory(ad_psi_3, num_steps, gamma)

    _steps = np.arange(num_steps)
    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.plot(_steps, p1_psi1, 'o-', label=r'$|1\rangle$', markersize=3)
    _ax.plot(_steps, p1_psi2, 's-', label=r'$|+\rangle$', markersize=3)
    _ax.plot(_steps, p1_psi3, '^-', label=r'$S|+\rangle$', markersize=3)
    _ax.set_xlabel('Decay Steps'); _ax.set_ylabel(r'Probability of Measuring $|1\rangle$')
    _ax.set_title('Amplitude Damping: P(1) Trajectories')
    _ax.set_ylim(0, 1); _ax.legend(); _ax.grid(True, alpha=0.3)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.2
    2. The phase damping channel may arise from fluctuations in unitary controls or scattering events that do not shift the state of the computational Hilbert space. Show that in the latter scenario, the qubit observes a channel of the form
       $$\rho' = \mathcal{E}_{PD}(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger \qquad E_0 = \begin{pmatrix}1 & 0 \\ 0 & \sqrt{1-p}\end{pmatrix} \qquad E_1 = \begin{pmatrix}0 & 0 \\ 0 & \sqrt{p}\end{pmatrix}$$
       Again, choose a few pure states on the surface of the Bloch sphere and plot their trajectories as a function of time (or decay time steps) when subject to this type of environment interaction.
    """)
    return


@app.cell
def _(ad_psi_1, ad_psi_2, ad_psi_3, np, plt):
    # Phase Damping Channel: Plot trajectories for the three states
    def phase_damping_channel(rho, p):
        """Apply phase damping channel to density matrix rho with dephasing probability p."""
        E0 = np.array([[1, 0], [0, np.sqrt(1 - p)]])
        E1 = np.array([[0, 0], [0, np.sqrt(p)]])
        return E0 @ rho @ E0.T.conj() + E1 @ rho @ E1.T.conj()

    p_pd = 0.1  # dephasing probability per step
    num_steps_pd = 50

    def track_coherence_trajectory(initial_state, num_steps, p):
        """Track coherence (|rho_01|) as phase damping channel is repeatedly applied."""
        rho = np.outer(initial_state, initial_state.conj())
        coherence_values = []
        for step in range(num_steps):
            coherence = np.abs(rho[0, 1])
            coherence_values.append(coherence)
            rho = phase_damping_channel(rho, p)
        return np.array(coherence_values)

    coh_psi1 = track_coherence_trajectory(ad_psi_1, num_steps_pd, p_pd)
    coh_psi2 = track_coherence_trajectory(ad_psi_2, num_steps_pd, p_pd)
    coh_psi3 = track_coherence_trajectory(ad_psi_3, num_steps_pd, p_pd)

    _steps = np.arange(num_steps_pd)
    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.plot(_steps, coh_psi1, 'o-', label=r'$|1\rangle$', markersize=3)
    _ax.plot(_steps, coh_psi2, 's-', label=r'$|+\rangle$', markersize=3)
    _ax.plot(_steps, coh_psi3, '^-', label=r'$S|+\rangle$', markersize=3)
    _ax.set_xlabel('Dephasing Steps'); _ax.set_ylabel(r'Coherence Magnitude $|\rho_{01}|$')
    _ax.set_title('Phase Damping: Coherence Trajectories')
    _ax.legend(); _ax.grid(True, alpha=0.3)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.3
    3. When analyzing the teleportation process from the perspective of the receiver, we recognized the depolarizing channel,
    $$
    \mathcal{E}_D(\rho) = p\frac{\mathbb{I}}{2} + (1-p)\rho
    $$
       One last time, choose a few pure states on the surface of the Bloch sphere and plot their trajectories as a function of time (or decay time steps) when subject to this type of environment interaction. In terms of the vector of Pauli matrices, $\vec{\sigma}$, find the local operators $E_k$ that express the depolarizing channel in the form $\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$.
    """)
    return


@app.cell
def _(ad_psi_1, ad_psi_2, ad_psi_3, np, plt):
    # Depolarizing Channel: Plot trajectories for the three states
    def depolarizing_channel(rho, p):
        """Apply depolarizing channel to density matrix rho with depolarizing probability p."""
        return p * np.eye(2) / 2 + (1 - p) * rho

    p_depol = 0.1  # depolarizing probability per step
    num_steps_depol = 50

    def track_purity_trajectory(initial_state, num_steps, p):
        """Track purity Tr(ρ²) as depolarizing channel is repeatedly applied."""
        rho = np.outer(initial_state, initial_state.conj())
        purity_values = []
        for step in range(num_steps):
            purity = np.trace(rho @ rho).real
            purity_values.append(purity)
            rho = depolarizing_channel(rho, p)
        return np.array(purity_values)

    pur_psi1 = track_purity_trajectory(ad_psi_1, num_steps_depol, p_depol)
    pur_psi2 = track_purity_trajectory(ad_psi_2, num_steps_depol, p_depol)
    pur_psi3 = track_purity_trajectory(ad_psi_3, num_steps_depol, p_depol)

    _steps = np.arange(num_steps_depol)
    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.plot(_steps, pur_psi1, 'o-', label=r'$|1\rangle$', markersize=3)
    _ax.plot(_steps, pur_psi2, 's-', label=r'$|+\rangle$', markersize=3)
    _ax.plot(_steps, pur_psi3, '^-', label=r'$S|+\rangle$', markersize=3)
    _ax.set_xlabel('Depolarization Steps'); _ax.set_ylabel(r'Purity $\mathrm{Tr}(\rho^2)$')
    _ax.set_title('Depolarizing Channel: Purity Trajectories')
    _ax.set_ylim(0, 1); _ax.legend(); _ax.grid(True, alpha=0.3)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    > This response was generated by claude

    **Kraus operator decomposition of the depolarizing channel.**

    Starting from $\mathcal{E}_D(\rho) = p\frac{I}{2} + (1-p)\rho$, we use the identity $\frac{I}{2} = \frac{1}{4}(I\rho I + X\rho X + Y\rho Y + Z\rho Z)$, valid for any single-qubit $\rho$. Substituting:

    $$\mathcal{E}_D(\rho) = \frac{p}{4}(I\rho I + X\rho X + Y\rho Y + Z\rho Z) + (1-p)\rho$$

    $$= \left(1 - \frac{3p}{4}\right)\rho + \frac{p}{4}(X\rho X + Y\rho Y + Z\rho Z)$$

    Reading off the Kraus operators:

    $$E_0 = \sqrt{1 - \tfrac{3p}{4}}\; I, \quad E_1 = \frac{\sqrt{p}}{2}\; X, \quad E_2 = \frac{\sqrt{p}}{2}\; Y, \quad E_3 = \frac{\sqrt{p}}{2}\; Z$$

    **Completeness check:** $\sum_k E_k^\dagger E_k = (1 - \tfrac{3p}{4})I + \tfrac{p}{4}(I + I + I) = I$ ✓

    **Physical interpretation:** With probability $1 - 3p/4$, nothing happens; with probability $p/4$ each, a random Pauli error (bit-flip $X$, bit-phase-flip $Y$, or phase-flip $Z$) is applied. The channel is physical (CP) for $p \in [0, 4/3]$; for $p > 1$ the "identity" coefficient becomes negative, corresponding to a entanglement-breaking channel that overshoots the maximally mixed state.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.4
    4. In a desire to erase your quantum hard drives, but not destroy the drives themselves, you decide to bring in destructable auxiliary qubits to implement the completely depolarizing channel ($p = 1$). For a single-qubit drive, construct a circuit describing a unitary interaction, $U$, between your drive and the auxiliary qubits that will produce this depolarizing channel.

    *Hint:* what effect would a CNOT have?
    *Hint:* you may consider using more than one qubit in the auxiliary qubit "environment"?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **A.4 Solution — Completely Depolarizing Circuit**
    > This response was written by claude

    **Effect of a CNOT:** If ancilla starts in $|0\rangle$, CNOT$(drive \to anc)$ creates: $(\alpha|0\rangle+\beta|1\rangle)|0\rangle \to \alpha|00\rangle+\beta|11\rangle$. Tracing out ancilla gives $\rho$ unchanged. But if ancilla starts in $|+\rangle = H|0\rangle$: tracing out gives $(I\rho I + X\rho X)/2$ — random X flip applied with probability 1/2.

    **Full depolarizing with 2 ancillas:**
    1. Ancilla 1 in $|+\rangle$ (apply $H$) → $CNOT(anc_1 \to drive)$ → traces to $(I + X)/2$ random
    2. Ancilla 2 in $|+\rangle$ (apply $H$) → $CZ(anc_2, drive)$ → introduces random Z flip

    Combined effect (tracing both ancillas):
    $$\mathcal{E}(\rho) = \frac{1}{4}(I\rho I + X\rho X + Z\rho Z + XZ\,\rho\, ZX) = \frac{\mathbb{I}}{2}$$

    **Verification:** For any $\rho$, the four Kraus operators $\{I, X, Z, XZ\}/2$ satisfy $\sum E_k^\dagger E_k = I$ and $\sum E_k \rho E_k^\dagger = \mathbb{I}/2$. ✓
    """)
    return


@app.cell
def _(QuantumCircuit, QuantumRegister):
    # This response was written by claude
    # A.4: Completely depolarizing channel circuit and numerical verification

    # Draw the circuit
    _qr_drive = QuantumRegister(1, 'drive')
    _qr_anc   = QuantumRegister(2, 'anc')
    _qc_a4    = QuantumCircuit(_qr_drive, _qr_anc)

    _qc_a4.h(_qr_anc[0])   # anc1 -> |+>
    _qc_a4.h(_qr_anc[1])   # anc2 -> |+>
    _qc_a4.cx(_qr_anc[0], _qr_drive[0])   # CNOT(anc1 -> drive): X-type
    _qc_a4.cz(_qr_anc[1],  _qr_drive[0])  # CZ(anc2, drive): Z-type

    _qc_a4.draw(output='mpl')
    return


@app.cell
def _(np):
    # This response was written by claude
    # Verify numerically: tracing ancillas gives I/2 for any input state

    _I2 = np.eye(2, dtype=complex)
    _X  = np.array([[0,1],[1,0]], dtype=complex)
    _Z  = np.array([[1,0],[0,-1]], dtype=complex)

    def _depol_verify(rho):
        """Apply E(ρ) = (IρI + XρX + ZρZ + XZρZX)/4."""
        return (rho + _X@rho@_X + _Z@rho@_Z + (_X@_Z)@rho@(_Z@_X)) / 4

    # Test on several states
    for _label, _rho_test in [
        ("|0><0|", np.array([[1,0],[0,0]], dtype=complex)),
        ("|+><+|", np.array([[0.5,0.5],[0.5,0.5]], dtype=complex)),
        ("|1><1|", np.array([[0,0],[0,1]], dtype=complex)),
    ]:
        _result = _depol_verify(_rho_test)
        print(f"E({_label}) = I/2: {np.allclose(_result, _I2/2)}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.5
    5. Generalize your unitary circuit to implement the completely depolarizing channel to a two qubit system, $\mathcal{E}(\rho) = \frac{\mathbb{I}}{4}$. How would your circuit generalize to an $n$-qubit system, $\mathcal{E}(\rho) = \frac{\mathbb{I}}{2^n}$? How many additional qubits do you anticipate using as a function of this memory capacity of your drive?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **A.5 Solution — Generalization to n Qubits**
    > This response was written by claude

    **Two-qubit system:** Apply the single-qubit depolarizing circuit independently to each qubit. For each qubit $k$:
    - Add ancilla pair $(a_{2k-1}, a_{2k})$ in $|00\rangle$
    - Apply $H$ to both ancillas
    - Apply $CNOT(a_{2k-1} \to \text{sys}_k)$
    - Apply $CZ(a_{2k}, \text{sys}_k)$

    This traces each qubit to $\mathbb{I}/2$ independently, giving $\mathcal{E}(\rho_{AB}) = \mathbb{I}/2 \otimes \mathbb{I}/2 = \mathbb{I}_4/4$. ✓

    **Generalization to $n$ qubits:**
    - Circuit: $2n$ ancilla qubits total
    - For qubit $k = 1, \ldots, n$: $H$ on $a_{2k-1}$, $H$ on $a_{2k}$, $CNOT(a_{2k-1} \to k)$, $CZ(a_{2k}, k)$
    - Result: $\mathcal{E}(\rho) = \mathbb{I}_{2^n}/2^n$

    **Number of ancilla qubits:** $\boxed{2n}$ — linear in the number of system qubits (not exponential!). This is efficient: erasing an $n$-qubit memory requires only $2n$ disposable ancilla qubits.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### B. Non-Uniqueness of Convex Decompositions
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.1
    1. Consider the following pure state $|\psi\rangle = \frac{1}{\sqrt{3}} |00\rangle + \sqrt{\frac{2}{3}} |11\rangle$. All pure-state density matrices have single-term convex decompositions such that $\rho = |\psi\rangle\langle\psi|$. The associated measurement probability vector in the density matrix eigenbasis is thus $\{1, 0, 0, \cdots, 0\}$.

    Find an alternate decomposition of the density matrix that is composed of an orthonormal basis yielding a uniform measurement probability vector.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **B.1 Solution — Uniform Measurement Basis**
    > This response was written by claude

    Given $|\psi\rangle = \frac{1}{\sqrt{3}}|00\rangle + \sqrt{\frac{2}{3}}|11\rangle$, we want an orthonormal basis $\{|f_i\rangle\}_{i=1}^4$ such that $|\langle f_i|\psi\rangle|^2 = \frac{1}{4}$ for all $i$.

    **Method:** Construct a unitary $U$ mapping $|\psi\rangle \to \frac{1}{2}(1,1,1,1)^T$ (the uniform superposition). The columns of $U^\dagger$ form the desired basis.

    Start with $|\psi\rangle$ as the first column, then use Gram-Schmidt to extend to a full orthonormal set $\{|\psi\rangle, |e_2\rangle, |e_3\rangle, |e_4\rangle\}$. Apply a Hadamard-like rotation within this basis: define $|f_i\rangle = \frac{1}{2}\sum_j H_{ij}|e_j'\rangle$ where $H$ is the $4\times 4$ Hadamard matrix.

    The code below computes the explicit basis numerically.
    """)
    return


@app.cell
def _(np):
    # This response was written by claude
    # B.1: Find uniform measurement basis for |ψ⟩ = (1/√3)|00⟩ + √(2/3)|11⟩

    _psi_b1 = np.zeros(4, dtype=complex)
    _psi_b1[0] = 1/np.sqrt(3)   # |00⟩
    _psi_b1[3] = np.sqrt(2/3)   # |11⟩

    _rho_b1 = np.outer(_psi_b1, _psi_b1.conj())

    # Build an orthonormal basis starting from |ψ⟩ via Gram-Schmidt
    _seed = np.vstack([_psi_b1, np.eye(4, dtype=complex)[:3]])
    _basis = np.zeros((4, 4), dtype=complex)
    for _i in range(4):
        _v = _seed[_i].copy()
        for _j in range(_i):
            _v -= np.dot(_basis[_j].conj(), _v) * _basis[_j]
        _nrm = np.linalg.norm(_v)
        if _nrm > 1e-10:
            _basis[_i] = _v / _nrm

    # Apply 4×4 Hadamard to mix these basis vectors uniformly
    _H4 = np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]], dtype=complex) / 2
    _uniform_basis = _H4 @ _basis  # rows are the |f_i⟩ states

    # Verify: all |⟨f_i|ψ⟩|² should equal 1/4
    _probs = np.abs(_uniform_basis @ _psi_b1)**2
    print("Measurement probabilities in uniform basis:", _probs.round(4))
    print("All equal 1/4:", np.allclose(_probs, 1/4))

    # Also verify orthonormality
    _G = _uniform_basis @ _uniform_basis.conj().T
    print("Orthonormality (should be I):", np.allclose(_G, np.eye(4)))

    # Note: Σ (1/4)|f_i⟩⟨f_i| = I/4 (maximally mixed), NOT |ψ⟩⟨ψ|.
    # This is expected — we found a measurement basis with uniform probabilities,
    # not a convex decomposition of ρ. A pure state has a unique eigendecomposition.
    _rho_reconstructed = sum((1/4) * np.outer(_uniform_basis[_i], _uniform_basis[_i].conj()) for _i in range(4))
    print("Σ(1/4)|f_i⟩⟨f_i| = I/4 (maximally mixed):", np.allclose(_rho_reconstructed, np.eye(4)/4))
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.2
    2. Through generalization of your methods, argue that quantum states, pure or mixed, of any qubit number have an associated orthonormal basis in which measurement outcomes are uniform, and thus minimally informative.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **B.2 Solution — Universal Uniform Measurement Basis**
    > This response was written by claude

    **Claim:** For any $n$-qubit state $\rho$ (pure or mixed) on $\mathbb{C}^d$ ($d = 2^n$), there exists an orthonormal basis $\{|f_i\rangle\}$ such that $\langle f_i|\rho|f_i\rangle = \frac{1}{d}$ for all $i$.

    **Generalizing the B.1 method:** In B.1, we found a uniform basis for a pure state by (1) extending $|\psi\rangle$ to an orthonormal basis via Gram-Schmidt, and (2) applying a $d \times d$ Hadamard matrix to redistribute the amplitudes uniformly. The key insight was that the Hadamard matrix $H_d$ satisfies $|H_{ij}|^2 = 1/d$ for all $i,j$, so any state's probability gets spread equally across all basis vectors.

    **Pure states (any $n$):** For an arbitrary $n$-qubit pure state $|\psi\rangle \in \mathbb{C}^d$, the same construction works directly:
    1. Extend $|\psi\rangle$ to a full orthonormal basis $\{|e_1\rangle = |\psi\rangle, |e_2\rangle, \ldots, |e_d\rangle\}$ via Gram-Schmidt.
    2. Define $|f_j\rangle = \sum_{k=1}^d F_{jk} |e_k\rangle$ where $F$ is any $d \times d$ unitary with $|F_{jk}|^2 = 1/d$ (e.g., the DFT matrix $F_{jk} = \frac{1}{\sqrt{d}} e^{2\pi i jk/d}$, or a Hadamard matrix when $d$ is a power of 2).
    3. Then $|\langle f_j|\psi\rangle|^2 = |F_{j1}|^2 = 1/d$ for all $j$, since $|\psi\rangle = |e_1\rangle$. The $\{|f_j\rangle\}$ are orthonormal because $F$ is unitary.

    **Mixed states:** Any mixed state has a spectral decomposition $\rho = \sum_k \lambda_k |\lambda_k\rangle\langle\lambda_k|$. The measurement probability of outcome $|f_j\rangle$ is:
    $$p_j = \langle f_j|\rho|f_j\rangle = \sum_k \lambda_k |\langle f_j|\lambda_k\rangle|^2$$

    We want $p_j = 1/d$ for all $j$. If we choose $|f_j\rangle$ such that $|\langle f_j|\lambda_k\rangle|^2 = 1/d$ for *all* $j,k$ (i.e., the measurement basis is *mutually unbiased* with the eigenbasis), then:
    $$p_j = \sum_k \lambda_k \cdot \frac{1}{d} = \frac{1}{d}\sum_k \lambda_k = \frac{1}{d}$$

    where the last step uses $\operatorname{tr}(\rho) = 1$. Constructing such a basis is exactly the same procedure: apply the DFT (or Hadamard) to the eigenbasis $\{|\lambda_k\rangle\}$. Since $|F_{jk}|^2 = 1/d$ for all entries, the resulting basis is mutually unbiased with the eigenbasis regardless of the eigenvalues $\lambda_k$.

    **Physical meaning:** Measuring in this basis yields maximum entropy (least information about the state). The construction works because the DFT/Hadamard "democratizes" probability — it maps any single basis state to a uniform superposition over all outputs, so every eigencomponent contributes equally to every measurement outcome. $\blacksquare$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.3
    3. This property of measurement bases for pure states is the trivial limiting case of a more general relation. Consider density matrix $\rho$ with vector of eigenvalues $\vec{\lambda}_\rho$: there exists an orthonormal basis $\{|v\rangle\}$ decomposing the density matrix as $\rho = \sum_k p_k|v_k\rangle\langle v_k|$, if and only if $\vec{p}$ is *majorized* by $\vec{\lambda}_\rho$, $\vec{p} \prec \vec{\lambda}_\rho$. Physically, vector $\ell$ is majorized by vector $\tau$ when $\ell$ can be created by $\tau$ by averaging over random permutations of $\tau$'s components. Majorization thus provides a hierarchy of disorder or mixedness. The uniform distribution is at the base of this hierarchy, consistent with the observation that the uniform distribution can be majorized by any other probability distribution.

    After elaborating upon the definition, derivation, and physical meaning of this mathematical statement, use this information to
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **B.3 Solution — Majorization and Convex Decompositions**
    > This response was written by claude

    **Definition:** Vector $\vec{p}$ is *majorized* by $\vec{\lambda}$, written $\vec{p} \prec \vec{\lambda}$, if:
    $$\sum_{i=1}^k p^\downarrow_i \leq \sum_{i=1}^k \lambda^\downarrow_i \quad \forall\, k = 1, \ldots, d, \qquad \text{with equality at } k = d$$
    where $^\downarrow$ denotes sorting in decreasing order.

    **Theorem (Quantum Schur-Horn):** $\rho = \sum_k p_k |v_k\rangle\langle v_k|$ for some orthonormal basis $\{|v_k\rangle\}$ if and only if $\vec{p} \prec \vec{\lambda}_\rho$.

    **Derivation sketch:** ($\Rightarrow$) The matrix of inner products $M_{ij} = |\langle v_i | \lambda_j \rangle|^2$ is doubly stochastic (rows and columns sum to 1). By Birkhoff's theorem, doubly stochastic matrices are convex combinations of permutations, so $\vec{p} = M \vec{\lambda}$ is a convex combination of permutations of $\vec{\lambda}$, which is exactly the definition of majorization. ($\Leftarrow$) Conversely, any $\vec{p} \prec \vec{\lambda}$ can be realized by an appropriate orthonormal basis.

    **Physical meaning:** Majorization quantifies *disorder*. The eigenvalue distribution $\vec{\lambda}$ is the most "ordered" distribution achievable — it's the unique distribution that cannot be obtained as a mixture of permutations of itself except trivially. Any valid ensemble $\{p_k, |v_k\rangle\}$ must correspond to a distribution that is "more mixed" than $\vec{\lambda}$.

    The uniform distribution $\vec{u} = (1/d, \ldots, 1/d)$ is at the *bottom* of the majorization hierarchy — every distribution majorizes it, so every state has a uniform-probability measurement basis (B.2 result).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.3.a
       - State the minimum number of pure state ensembles that could be used to prepare a mixed state.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **B.3.a Solution — Minimum Ensemble Size**
    > This response was written by claude

    The minimum number of pure states needed to represent $\rho$ as a convex mixture is the **rank** of $\rho$ — the number of nonzero eigenvalues.

    **Proof from majorization:** The eigendecomposition $\rho = \sum_{i=1}^r \lambda_i |\lambda_i\rangle\langle\lambda_i|$ (where $r = \text{rank}(\rho)$) uses exactly $r$ terms with $\vec{p} = \vec{\lambda}$. Since $\vec{\lambda} \prec \vec{\lambda}$ trivially (equality), this is a valid ensemble. Any ensemble with fewer than $r$ terms would require probabilities $\vec{p}$ with fewer nonzero entries, but for $\vec{p} \prec \vec{\lambda}$ with fewer nonzero components, the partial sum condition would fail unless $\vec{\lambda}$ itself has fewer nonzero entries.

    **Summary:**
    - Pure state ($\text{rank} = 1$): minimum 1 pure state (trivially itself)
    - Rank-$r$ mixed state: minimum $r$ pure states
    - Maximally mixed state ($\rho = \mathbb{I}/d$): minimum $d$ pure states (any orthonormal basis works)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.3.b
       - Determine whether an orthonormal basis exists associated with the following probability measurement vectors for the mixed state $\rho = \frac{1}{4}\mathbb{I}_4 + \frac{1}{8}(X \otimes X - Y \otimes Y)$:
         - $\{\frac{1}{2}, 0, 0, \frac{1}{2}\}$
         - $\{\frac{1}{3}, \frac{1}{3}, 0, \frac{1}{3}\}$
         - $\{0, \frac{1}{6}, \frac{2}{3}, \frac{1}{6}\}$
         - $\{\frac{1}{16}, \frac{3}{8}, \frac{1}{2}, \frac{1}{16}\}$
         - $\{\frac{2}{8}, \frac{3}{8}, \frac{2}{8}, \frac{1}{8}\}$

    In other words: physically, if you aimed to prepare $\rho$ in the lab and, upon scrutiny of your success, found an ensemble measurement basis yielding each of the above probability distributions, would this raise concern for your state preparation fidelity?

    For further discussion on the role and importance of majorization in quantum information, see *Phys. Rev. Lett.*, Vol 83 (2), pp 436–439 (1999) by Nielsen and *Phys. Rev. Lett.*, Vol. 86, 5184-7 (2001) by Nielsen and Kempe.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **B.3.b Solution — Majorization Check**
    > This response was written by claude

    A probability vector $\vec{p}$ corresponds to a valid ensemble decomposition of $\rho$ iff $\vec{p} \prec \vec{\lambda}_\rho$ (p is majorized by the eigenvalues of ρ).

    **Majorization check:** $\vec{p} \prec \vec{\lambda}$ iff for all $k$: $\sum_{i=1}^k p^\downarrow_i \leq \sum_{i=1}^k \lambda^\downarrow_i$, with equality at $k = d$.

    If a vector fails this test, it cannot arise as measurement probabilities in any valid ensemble decomposition of $\rho$ — so observing it **would raise concern** about state preparation fidelity.

    The code below computes the eigenvalues of $\rho = \frac{1}{4}\mathbb{I}_4 + \frac{1}{8}(X \otimes X - Y \otimes Y)$ and runs the majorization test for each of the five given probability vectors.
    """)
    return


@app.cell
def _(np):
    # This response was written by claude
    # B.3.b: Compute eigenvalues of ρ and check majorization for each probability vector

    _I4 = np.eye(4, dtype=complex)
    _X = np.array([[0,1],[1,0]], dtype=complex)
    _Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    _XX = np.kron(_X, _X)
    _YY = np.kron(_Y, _Y)

    _rho_b3b = _I4/4 + (_XX - _YY)/8
    _evals = np.sort(np.real(np.linalg.eigvals(_rho_b3b)))[::-1]
    print(f"Eigenvalues of ρ: {_evals.round(4)}")

    def _is_majorized(p, lam):
        """Check if p ≺ lam (p is majorized by lam)."""
        _p = np.sort(np.array(p, dtype=float))[::-1]
        _l = np.sort(np.array(lam, dtype=float))[::-1]
        if not np.isclose(sum(_p), 1.0) or not np.isclose(sum(_l), 1.0):
            return False, "probabilities don't sum to 1"
        for k in range(1, len(_p)+1):
            if np.sum(_p[:k]) > np.sum(_l[:k]) + 1e-9:
                return False, f"fails at k={k}: cumsum(p)={np.sum(_p[:k]):.4f} > cumsum(λ)={np.sum(_l[:k]):.4f}"
        return True, "majorized ✓"

    _test_vecs = [
        [1/2, 0, 0, 1/2],
        [1/3, 1/3, 0, 1/3],
        [0, 1/6, 2/3, 1/6],
        [1/16, 3/8, 1/2, 1/16],
        [2/8, 3/8, 2/8, 1/8],
    ]

    print("\nMajorization results (valid ensemble ↔ p ≺ λ):")
    for _i, _p in enumerate(_test_vecs):
        _ok, _msg = _is_majorized(_p, _evals)
        _concern = "NO concern" if _ok else "RAISE CONCERN"
        print(f"  vec {_i+1}: {[round(x,4) for x in sorted(_p)[::-1]]}  →  {_msg}  ({_concern})")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### C. Subsystem Entanglement Quantification

    You wish to distribute highly entangled states between two parties, $A$ and $B$. However, your entanglement resource generators will only create three-qubit entangled states, $|GHZ\rangle = \frac{|000\rangle+|111\rangle}{\sqrt{2}}$ or $|W\rangle = \frac{|001\rangle+|010\rangle+|100\rangle}{\sqrt{2}}$.

    As we discuss, the GHZ-type of entanglement is "fragile" in the sense that tracing/ignoring one of its qubits leaves the remaining two in a mixed state with density matrix identical to that of a completely classical mixture, $\frac{|00\rangle\langle00|+|11\rangle\langle11|}{2}$. However, the $|W\rangle$ state, upon tracing of a qubit retains entanglement. Thus, if the third qubit is ignored, utilization of the $|W\rangle$ state generator would be advised.

    In the following, we will explore whether this choice changes if we have a collaborative third party, $C$, who reliably reports a measurement outcome on the third qubit.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.1
    1. Quantify the entanglement yield as the probability-weighted average of that present in each post-measurement pure state (pick your favorite $E(|\psi\rangle\langle\psi|)$ pure-state entanglement measure),
       $$E_{\text{yield}} \left(\{p_i, |\psi_i\rangle\langle\psi_i|\}\right) = \sum_i p_i E\left(|\psi_i\rangle\langle\psi_i|\right)$$
       What is the $A$-$B$ entanglement yield for the GHZ and W states upon measurement of $C$ in the Z-basis?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **C.1 Solution — Entanglement Yield in Z-basis**
    > This response was written by claude

    Using **concurrence** $C(\rho)$ as the entanglement measure ($C = 1$ for Bell states, $C = 0$ for product states).

    **GHZ** $= \frac{|000\rangle+|111\rangle}{\sqrt{2}}$ measured in Z-basis on C:
    - $|0\rangle_C$ (prob $\frac{1}{2}$): AB state $= |00\rangle$ — product, $C = 0$
    - $|1\rangle_C$ (prob $\frac{1}{2}$): AB state $= |11\rangle$ — product, $C = 0$

    $$E_{\text{yield}}^{\text{GHZ}} = \frac{1}{2}(0) + \frac{1}{2}(0) = \mathbf{0}$$

    **W** $= \frac{|001\rangle+|010\rangle+|100\rangle}{\sqrt{3}}$ measured in Z-basis on C:
    - $|0\rangle_C$ (prob $\frac{2}{3}$): AB state $= \frac{|01\rangle+|10\rangle}{\sqrt{2}} = |\Psi^+\rangle$ — Bell state, $C = 1$
    - $|1\rangle_C$ (prob $\frac{1}{3}$): AB state $= |00\rangle$ — product, $C = 0$

    $$E_{\text{yield}}^{W} = \frac{2}{3}(1) + \frac{1}{3}(0) = \mathbf{\frac{2}{3}}$$

    For Z-basis measurement, **W is superior** to GHZ.
    """)
    return


@app.cell
def _(np):
    # This response was written by claude
    # C.1: Compute entanglement yields for GHZ and W under Z-basis measurement

    _GHZ_c1 = np.zeros(8, dtype=complex)
    _GHZ_c1[0] = 1/np.sqrt(2); _GHZ_c1[7] = 1/np.sqrt(2)

    _W_c1 = np.zeros(8, dtype=complex)
    _W_c1[1] = 1/np.sqrt(3); _W_c1[2] = 1/np.sqrt(3); _W_c1[4] = 1/np.sqrt(3)

    def _concurrence_c1(rho):
        _Y = np.array([[0,-1j],[1j,0]], dtype=complex)
        _YY = np.kron(_Y, _Y)
        _rt = _YY @ rho.conj() @ _YY
        _R = rho @ _rt
        _ev = np.maximum(np.sort(np.real(np.linalg.eigvals(_R)))[::-1], 0)
        _sq = np.sqrt(_ev)
        return max(0.0, float(_sq[0] - _sq[1] - _sq[2] - _sq[3]))

    def _z_basis_yield(psi_3q):
        _psi = psi_3q.reshape(2, 2, 2)  # [A,B,C]
        total = 0.0
        for _c_val in [0, 1]:
            _proj = _psi[:, :, _c_val]  # shape [A,B]
            _prob = float(np.real(np.sum(np.abs(_proj)**2)))
            if _prob > 1e-10:
                _rho_ab = np.outer(_proj.flatten(), _proj.flatten().conj()) / _prob
                total += _prob * _concurrence_c1(_rho_ab)
        return total

    print(f"GHZ Z-basis entanglement yield: {_z_basis_yield(_GHZ_c1):.4f}")
    print(f"W   Z-basis entanglement yield: {_z_basis_yield(_W_c1):.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.2
    2. Tracing of a qubit may be implemented as the result-agnostic accumulation of states after measurement in any basis, i.e., the reduced density matrix is invariant with respect to the tracing basis. However, the measurement basis can affect the particular pure state decomposition remaining (providing physical intuition for the many-to-one relationship between pure state decompositions and pure states).

    For both the $|GHZ\rangle$ and $|W\rangle$ state, find and describe a measurement basis for $C$ that maximizes the entanglement yield established between $A$ and $B$ (note: your particular values may depend on your chosen pure state entanglement measure). How do the maxima compare between the GHZ and W resource?

    *Hint:* Intuitively, the property of being in the $|000\rangle$ or $|111\rangle$ state has been distributed throughout the three Hilbert spaces in the GHZ state. In efforts to not learn this information from the measurement of qubit $C$, recall from the previous exercise that minimally informative/uniform probability vectors are available upon a change of measurement basis.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **C.2 Solution — Basis Maximizing Entanglement Yield**
    > This response was written by claude

    **GHZ:** In the X-basis $\{|+\rangle,|-\rangle\}$, both outcomes give equal probability $\frac{1}{2}$ and both post-measurement AB states are Bell states (fully entangled). Entanglement yield = $\frac{1}{2}(1) + \frac{1}{2}(1) = \mathbf{1}$ ebit.

    Any equatorial basis (uniform measurement probabilities on C) achieves this maximum. The Z-basis gives yield 0 (product states). The hint points to this: uniform C-measurement probabilities → no "which-component" information leaked → maximum AB entanglement.

    **W:** Scanning over measurement bases numerically confirms the Z-basis is optimal: outcome $|0\rangle_C$ (prob $\frac{2}{3}$) gives $|\Psi^+\rangle$ (1 ebit), outcome $|1\rangle_C$ (prob $\frac{1}{3}$) gives $|00\rangle$ (0 ebits). Yield = $\frac{2}{3}$ ebit.

    **Comparison:** GHZ achieves **1 ebit** (maximum possible), W achieves at most $\frac{2}{3}$ ebit. GHZ is superior when a collaborating third party is available.
    """)
    return


@app.cell
def _(np, plt):
    # This response was written by claude
    # C.2: Numerically scan measurement basis for GHZ and W, plot entanglement yield

    _GHZ = np.zeros(8, dtype=complex)
    _GHZ[0] = 1/np.sqrt(2); _GHZ[7] = 1/np.sqrt(2)  # |000> + |111>

    _W = np.zeros(8, dtype=complex)
    _W[1] = 1/np.sqrt(3); _W[2] = 1/np.sqrt(3); _W[4] = 1/np.sqrt(3)  # |001>+|010>+|100>

    def _concurrence_2q(rho):
        _Y = np.array([[0,-1j],[1j,0]], dtype=complex)
        _YY = np.kron(_Y, _Y)
        _rt = _YY @ rho.conj() @ _YY
        _R = rho @ _rt
        _ev = np.sort(np.real(np.linalg.eigvals(_R)))[::-1]
        _ev = np.maximum(_ev, 0)
        _sq = np.sqrt(_ev)
        return max(0.0, float(_sq[0] - _sq[1] - _sq[2] - _sq[3]))

    def _entanglement_yield(psi_3q, theta, phi):
        """Measure C in basis {|m0>,|m1>} parameterized by theta, phi."""
        _m0 = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)], dtype=complex)
        _m1 = np.array([-np.exp(-1j*phi)*np.sin(theta/2), np.cos(theta/2)], dtype=complex)
        _rho = np.outer(psi_3q, psi_3q.conj())
        total_yield = 0.0
        for _mv in [_m0, _m1]:
            # Project C (qubit 2) onto |mv>
            # State = sum_{a,b,c} psi[a,b,c] |a>|b>|c>
            _psi = psi_3q.reshape(2,2,2)  # [A,B,C]
            _proj = np.einsum('c,abc->ab', _mv.conj(), _psi)  # shape [A,B]
            _prob = float(np.real(np.sum(np.abs(_proj)**2)))
            if _prob > 1e-10:
                _rho_ab = np.outer(_proj.flatten(), _proj.flatten().conj()) / _prob
                total_yield += _prob * _concurrence_2q(_rho_ab)
        return total_yield

    _thetas = np.linspace(0, np.pi, 30)
    _phis   = np.linspace(0, 2*np.pi, 30)
    _TH, _PH = np.meshgrid(_thetas, _phis)
    _Y_GHZ = np.array([[_entanglement_yield(_GHZ, t, p) for t in _thetas] for p in _phis])
    _Y_W   = np.array([[_entanglement_yield(_W,   t, p) for t in _thetas] for p in _phis])

    _fig_c2, _axes_c2 = plt.subplots(1, 2, figsize=(12, 4))
    _c1 = _axes_c2[0].contourf(_thetas, _phis, _Y_GHZ, levels=20, cmap='viridis')
    _axes_c2[0].set_xlabel('θ'); _axes_c2[0].set_ylabel('φ')
    _axes_c2[0].set_title(f'GHZ: Entanglement Yield (max={_Y_GHZ.max():.3f})')
    plt.colorbar(_c1, ax=_axes_c2[0])
    _c2 = _axes_c2[1].contourf(_thetas, _phis, _Y_W, levels=20, cmap='viridis')
    _axes_c2[1].set_xlabel('θ'); _axes_c2[1].set_ylabel('φ')
    _axes_c2[1].set_title(f'W: Entanglement Yield (max={_Y_W.max():.3f})')
    plt.colorbar(_c2, ax=_axes_c2[1])
    plt.tight_layout(); plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.3
    3. Based on your experience, describe a $C$-collaborative protocol that begins in the $|W\rangle$ or $|GHZ\rangle$ state and deterministically produces the state $|\psi\rangle_{AB} = \frac{|00\rangle+|11\rangle}{\sqrt{2}}$.

    *Hint:* You may wish to consider a two-step procedure of the form (1) $C$-space measurement and (2) measurement-dependent $A$, $B$-space local unitaries.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **C.3 Solution — Deterministic Bell State from GHZ**
    > This response was written by claude

    **From GHZ:** Measure C in the X-basis $\{|+\rangle, |-\rangle\} = \{(|0\rangle\pm|1\rangle)/\sqrt{2}\}$.

    $|GHZ\rangle = \frac{|000\rangle+|111\rangle}{\sqrt{2}} = \frac{1}{\sqrt{2}}\left[\frac{|00\rangle+|11\rangle}{\sqrt{2}}|+\rangle_C + \frac{|00\rangle-|11\rangle}{\sqrt{2}}|-\rangle_C\right]$

    - Outcome $|+\rangle_C$ (prob $\frac{1}{2}$): AB in $|\Phi^+\rangle = \frac{|00\rangle+|11\rangle}{\sqrt{2}}$ ✓
    - Outcome $|-\rangle_C$ (prob $\frac{1}{2}$): AB in $|\Phi^-\rangle = \frac{|00\rangle-|11\rangle}{\sqrt{2}}$ → apply $Z_A$ to get $|\Phi^+\rangle$ ✓

    This is **deterministic** with a local correction.

    **From W:** $|W\rangle = \frac{|001\rangle+|010\rangle+|100\rangle}{\sqrt{3}}$

    - Outcome $|0\rangle_C$ (prob $\frac{2}{3}$): AB in $\frac{|01\rangle+|10\rangle}{\sqrt{2}} = |\Psi^+\rangle$ — entangled ✓ (apply local unitaries to convert to $|\Phi^+\rangle$)
    - Outcome $|1\rangle_C$ (prob $\frac{1}{3}$): AB in $|00\rangle$ — **not entangled** ✗

    The W protocol is **not deterministic** — fails with probability $\frac{1}{3}$. GHZ is preferred for deterministic Bell state generation.
    """)
    return


@app.cell
def _(ClassicalRegister, QuantumCircuit, QuantumRegister):
    # This response was written by claude
    # C.3: GHZ -> Bell state protocol circuit

    _qr_c3 = QuantumRegister(3, 'ABC')
    _cr_c3 = ClassicalRegister(1, 'c_meas')
    _qc_c3 = QuantumCircuit(_qr_c3, _cr_c3)

    # Prepare GHZ state
    _qc_c3.h(_qr_c3[0])
    _qc_c3.cx(_qr_c3[0], _qr_c3[1])
    _qc_c3.cx(_qr_c3[0], _qr_c3[2])
    _qc_c3.barrier()

    # Measure C in X-basis (H then Z-measure)
    _qc_c3.h(_qr_c3[2])
    _qc_c3.measure(_qr_c3[2], _cr_c3[0])
    _qc_c3.barrier()

    # Correction: if outcome=1, apply Z to A
    with _qc_c3.if_test((_cr_c3[0], 1)):
        _qc_c3.z(_qr_c3[0])

    _qc_c3.draw(output='mpl')
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.4
    4. **Entangling from a Distance**: You have two immobile qubits, labeled $A_i$ and $B_i$, that you would like to entangle. Direct interactions between $A_i$ and $B_i$ are experimentally inaccessible. However, each immobile qubit can be entangled with a mobile qubit. These mobile qubits, $A_m$ and $B_m$, may be brought together for direct interaction.

    Devise a protocol within these constraints for the production of the state $|\psi\rangle_{A_iB_i} = \frac{|00\rangle+|11\rangle}{\sqrt{2}}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **C.4 Solution — Entangling from a Distance (Entanglement Swapping)**
    > This response was written by claude

    This is the **entanglement swapping** protocol:

    1. **Prepare Bell pairs locally:**
       - $A_i$ and $A_m$ share $|\Phi^+\rangle = \frac{|00\rangle+|11\rangle}{\sqrt{2}}$
       - $B_i$ and $B_m$ share $|\Phi^+\rangle = \frac{|00\rangle+|11\rangle}{\sqrt{2}}$

    2. **Bring mobile qubits together:** $A_m$ and $B_m$ meet.

    3. **Bell measurement on $A_m B_m$:** Apply CNOT$(A_m \to B_m)$ then $H$ on $A_m$, measure both in Z-basis. Outcome $(m_1, m_2)$ projects $A_i B_i$ into one of the 4 Bell states.

    4. **Classical correction:** Communicate $(m_1, m_2)$ to either party; apply $Z^{m_1} X^{m_2}$ to $B_i$. Now $A_i B_i$ share $|\Phi^+\rangle$.

    **Key insight:** $A_i$ and $B_i$ never interact; they become entangled via the "teleported" correlations through the mobile qubits. This is the basis of quantum repeaters.
    """)
    return


@app.cell
def _(ClassicalRegister, QuantumCircuit, QuantumRegister):
    # This response was written by claude
    # C.4: Entanglement swapping circuit

    _qr_Ai = QuantumRegister(1, 'A_i')
    _qr_Am = QuantumRegister(1, 'A_m')
    _qr_Bm = QuantumRegister(1, 'B_m')
    _qr_Bi = QuantumRegister(1, 'B_i')
    _cr    = ClassicalRegister(2, 'meas')
    _qc_c4 = QuantumCircuit(_qr_Ai, _qr_Am, _qr_Bm, _qr_Bi, _cr)

    # Step 1: Create Bell pairs A_i-A_m and B_i-B_m
    _qc_c4.h(_qr_Ai[0]); _qc_c4.cx(_qr_Ai[0], _qr_Am[0])
    _qc_c4.h(_qr_Bi[0]); _qc_c4.cx(_qr_Bi[0], _qr_Bm[0])
    _qc_c4.barrier()

    # Step 2: Bell measurement on A_m, B_m
    _qc_c4.cx(_qr_Am[0], _qr_Bm[0])
    _qc_c4.h(_qr_Am[0])
    _qc_c4.measure(_qr_Am[0], _cr[0])
    _qc_c4.measure(_qr_Bm[0], _cr[1])
    _qc_c4.barrier()

    # Step 3: Classical corrections on B_i
    with _qc_c4.if_test((_cr[0], 1)):
        _qc_c4.z(_qr_Bi[0])
    with _qc_c4.if_test((_cr[1], 1)):
        _qc_c4.x(_qr_Bi[0])

    _qc_c4.draw(output='mpl')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### D. Mixed-State Entanglement Distribution

    The text remarks "[T]wo mixtures with the same density matrices are indistinguishable or equivalent (analogous to the way two pure states that differ only by a global phase are equivalent)". In this problem, we will scrutinize to what extent this analogy holds in the context of mixed state entanglement structure.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.1
    1. Discuss the physical meaning captured by the irrelevance of global phases.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **D.1 Solution — Global Phases**
    > This response was written by claude

    A **global phase** is a complex number $e^{i\alpha}$ multiplying the entire state vector: $|\psi'\rangle = e^{i\alpha}|\psi\rangle$.

    **Why it is irrelevant:** All physical observables are expectation values $\langle O \rangle = \langle\psi|O|\psi\rangle$, and the density matrix $\rho = |\psi\rangle\langle\psi|$. Both are invariant under global phases:
    $$\rho' = e^{i\alpha}|\psi\rangle\langle\psi|e^{-i\alpha} = |\psi\rangle\langle\psi| = \rho$$

    **Physical meaning:** The state space of quantum mechanics is not a vector space but a **projective Hilbert space** — the space of rays (equivalence classes of vectors differing only by a global phase). The density matrix formalism makes this manifest: $\rho$ is the fundamental physical object, not $|\psi\rangle$.

    **Contrast with relative phases:** A *relative* phase between components of a superposition — e.g., changing $\alpha|0\rangle + \beta|1\rangle$ to $\alpha|0\rangle + e^{i\phi}\beta|1\rangle$ — *is* observable (it affects interference). Only the overall phase is unobservable.

    **Deeper consequence:** This irrelevance is what makes mixed states well-defined: if $\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$, we can independently rephase each $|\psi_i\rangle$ without changing $\rho$ or any physics.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.2
    2. Consider this separable mixture of unentangled states (*Phys. Rev. Lett.* **91**, 037902),

       $$\rho_{abc} = \frac{1}{6}\sum_{k=0}^{3} |\Psi_k\rangle\langle\Psi_k| \otimes |\Psi_{-k}\rangle\langle\Psi_{-k}| \otimes |0\rangle\langle 0| + \frac{1}{6}\sum_{j=0}^{1} |j\rangle\langle j| \otimes |j\rangle\langle j| \otimes |1\rangle\langle 1|$$
       where the eigenvectors of the $X, Y$ Pauli operators are parameterized as
       $$|\Psi_k\rangle = \frac{|0\rangle + e^{ik\pi/2}|1\rangle}{\sqrt{2}}$$
       List the six pure states present in the noted pure state decomposition.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **D.2 Solution — Six Pure States**
    > This response was written by claude

    Expanding $|\Psi_k\rangle = \frac{|0\rangle + e^{ik\pi/2}|1\rangle}{\sqrt{2}}$ for $k = 0, 1, 2, 3$:

    | $k$ | $|\Psi_k\rangle$ | Name |
    |---|---|---|
    | 0 | $\frac{|0\rangle+|1\rangle}{\sqrt{2}}$ | $|+\rangle$ |
    | 1 | $\frac{|0\rangle+i|1\rangle}{\sqrt{2}}$ | $|+i\rangle$ (Y eigenstate) |
    | 2 | $\frac{|0\rangle-|1\rangle}{\sqrt{2}}$ | $|-\rangle$ |
    | 3 | $\frac{|0\rangle-i|1\rangle}{\sqrt{2}}$ | $|-i\rangle$ |

    The **six pure states** (each with weight $\frac{1}{6}$) are:

    1. $|\Psi_0\rangle_A \otimes |\Psi_0\rangle_B \otimes |0\rangle_C = |+\rangle|+\rangle|0\rangle$
    2. $|\Psi_1\rangle_A \otimes |\Psi_{-1}\rangle_B \otimes |0\rangle_C = |{+i}\rangle|{-i}\rangle|0\rangle$
    3. $|\Psi_2\rangle_A \otimes |\Psi_{-2}\rangle_B \otimes |0\rangle_C = |-\rangle|-\rangle|0\rangle$
    4. $|\Psi_3\rangle_A \otimes |\Psi_{-3}\rangle_B \otimes |0\rangle_C = |{-i}\rangle|{+i}\rangle|0\rangle$
    5. $|0\rangle_A \otimes |0\rangle_B \otimes |1\rangle_C$
    6. $|1\rangle_A \otimes |1\rangle_B \otimes |1\rangle_C$

    All six are **product states** (unentangled), confirming that $\rho_{ABC}$ is separable.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.3
    3. Label Hilbert spaces from left-to-right as $A$, $B$, and $C$. Consider a procedure for distributing entanglement between parties A and B through direct interactions only with system C by sequential application of the entangling operation $CNOT_{control,target}$: first $CNOT_{A,C}$ followed by $CNOT_{B,C}$. You may imagine a scenario of entangling two stationary Hilbert spaces, $A$ and $B$, through local interactions with a mobile Hilbert space, $C$, providing the interconnect.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **D.3 Solution — Setup**
    > This response was written by claude

    The protocol uses the separable initial state $\rho_{ABC}$ from D.2 as a resource. C acts as a "messenger" qubit that interacts locally with A, then travels to interact with B. The key question is whether C needs to carry entanglement to distribute it between A and B.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.3.a
       - Through application of $CNOT_{A,C}$ on each pure state in the initial decomposition, show that the state after the first entangling operation is
         $$\rho'_{ABC} = \frac{1}{6}\sum_{k=0}^{3} \rho_{AC}\!\left(\frac{|00\rangle + e^{ik\pi/2}|11\rangle}{\sqrt{2}}\right) \otimes \rho_B\!\left(|\Psi_{-k}\rangle\right) + \frac{1}{6}|0,0,1\rangle\langle 0,0,1| + \frac{1}{6}|1,1,0\rangle\langle 1,1,0|$$
         where $\rho(|\psi\rangle) = |\psi\rangle\langle\psi|$. Are there entangled pure states present in the resulting pure state convex decomposition?

       The terminology of mixed state entanglement is the following: If a convex decomposition (positive real weights $\sum_i p_i = 1$) of the density matrix composed of only tensor-product pure states exists,
       $$\rho_{\text{separable}} = \sum_i p_i\, \rho^A_i \otimes \rho^B_i$$
       then the mixed state is considered *separable* i.e., its density matrix could be independently created without distributing entanglement. A mixed state is said to be entangled *iff* it is not separable.

       Is the resulting state, $\rho'_{ABC}$, entangled along the $AC|B$ Hilbert space partition? By analyzing the symmetry of the density matrix under the interchange of Hilbert spaces $B$ and $C$, is the state considered entangled along the $AB|C$ Hilbert space partition?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **D.3.a Solution — After First CNOT\_{A,C}**
    > This response was written by claude

    Applying $CNOT_{A,C}$ to each of the 6 pure states in $\rho_{ABC}$:

    - $|\Psi_k\rangle_A \otimes |\Psi_{-k}\rangle_B \otimes |0\rangle_C \xrightarrow{CNOT_{A,C}} \frac{|0\rangle_A|0\rangle_C + e^{ik\pi/2}|1\rangle_A|1\rangle_C}{\sqrt{2}} \otimes |\Psi_{-k}\rangle_B$ — **entangled AC pair**
    - $|j,j,1\rangle \xrightarrow{CNOT_{A,C}} |j,j,j\oplus 1\rangle$ — product state

    **Are there entangled pure states?** Yes — the 4 terms with $k=0,1,2,3$ become entangled AC states tensored with B.

    **Entanglement along $AC|B$?** Yes — the AC subsystem is entangled with neither B for those 4 terms, but the full state has entangled structure in the AC part. More precisely: the pure state decomposition contains states of the form $|\phi_{AC}\rangle \otimes |\Psi_{-k}\rangle_B$, so $\rho'_{ABC}$ is **separable** across $AC|B$ (it's already written as a convex mixture of tensor products).

    **Entanglement along $AB|C$?** We examine symmetry under $B \leftrightarrow C$ interchange: the AC-entangled states map to AB-entangled states, and B maps to C. The density matrix is **not symmetric** under $B \leftrightarrow C$, indicating the state is **entangled along $AB|C$**.
    """)
    return


@app.cell
def _(np):
    # This response was written by claude
    # D.3.a: Build ρ_abc, apply CNOT_{A,C}, check entanglement structure

    def _psi_k_d3a(k):
        return (np.array([1, np.exp(1j * k * np.pi / 2)]) / np.sqrt(2)).astype(complex)

    _ket0_d3a = np.array([1, 0], dtype=complex)
    _ket1_d3a = np.array([0, 1], dtype=complex)

    _rho_abc_d3a = np.zeros((8, 8), dtype=complex)
    for _k in range(4):
        _s = np.kron(np.kron(_psi_k_d3a(_k), _psi_k_d3a(-_k)), _ket0_d3a)
        _rho_abc_d3a += np.outer(_s, _s.conj()) / 6
    for _j in range(2):
        _sj = _ket0_d3a if _j == 0 else _ket1_d3a
        _s = np.kron(np.kron(_sj, _sj), _ket1_d3a)
        _rho_abc_d3a += np.outer(_s, _s.conj()) / 6

    def _cnot_ac_d3a(state_abc):
        state = state_abc.reshape(2, 2, 2)
        result = np.zeros_like(state)
        for _a in range(2):
            for _b in range(2):
                for _c in range(2):
                    result[_a, _b, _c ^ _a] += state[_a, _b, _c]
        return result.flatten()

    def _apply_gate_dm_d3a(dm, gate_fn):
        d = dm.shape[0]
        U = np.column_stack([gate_fn(np.eye(d, dtype=complex)[:, i]) for i in range(d)])
        return U @ dm @ U.conj().T

    _rho_prime_d3a = _apply_gate_dm_d3a(_rho_abc_d3a, _cnot_ac_d3a)

    # Check symmetry under B<->C swap
    def _swap_BC(rho_abc):
        """Permute B and C in the density matrix (qubits 1 and 2)."""
        rho = rho_abc.reshape(2, 2, 2, 2, 2, 2)  # [A,B,C,A',B',C']
        return rho.transpose(0, 2, 1, 3, 5, 4).reshape(8, 8)

    _rho_bc_swapped = _swap_BC(_rho_prime_d3a)
    print("ρ' symmetric under B↔C:", np.allclose(_rho_prime_d3a, _rho_bc_swapped))
    print("Max asymmetry:", np.max(np.abs(_rho_prime_d3a - _rho_bc_swapped)).round(6))
    print("→ ρ'_ABC is symmetric under B↔C" if np.allclose(_rho_prime_d3a, _rho_bc_swapped) else "→ ρ'_ABC is asymmetric under B↔C")
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.3.b
       - After application of the second entangling operation $CNOT_{B,C}$, show that the final mixed state is

         $$\rho''_{ABC} = \frac{1}{6}\sum_{k=0}^{3} \rho\!\left(\frac{1}{\sqrt{2}}\left(\frac{|00\rangle+|11\rangle}{\sqrt{2}}\right)|0\rangle_C + \frac{1}{\sqrt{2}}\left(\frac{e^{-ik\pi/2}|01\rangle+e^{ik\pi/2}|10\rangle}{\sqrt{2}}\right)|1\rangle_C\right) + \frac{1}{6}\rho(|0,0,1\rangle) + \frac{1}{6}\rho(|1,1,1\rangle)$$
         Using computational basis state projectors in the $C$ Hilbert space, show that this density matrix is equivalent to one decomposed in the form
         $$\rho''_{ABC} = p_0\, \rho^{(0)}_{AB} \otimes |0\rangle\langle 0| + p_1\, \rho^{(1)}_{AB} \otimes |1\rangle\langle 1|$$
         and is thus considered separable along the $AB|C$ Hilbert space partition. Write the expressions for $\rho^{(0)}_{AB}$ and $\rho^{(1)}_{AB}$, are either of these entangled mixed states? Write the final pure state decompositions upon projective measurement of $|0\rangle_C$ or $|1\rangle_C$, with what probabilities would these two measurements be made? Are entangled states present across the $A|B$ partition in either measurement outcome?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **D.3.b Solution — After Second CNOT\_{B,C}**
    > This response was written by claude

    After $CNOT_{B,C}$ on the 6 states of $\rho'_{ABC}$:

    The final state separates as $\rho''_{ABC} = p_0\,\rho^{(0)}_{AB} \otimes |0\rangle\langle 0|_C + p_1\,\rho^{(1)}_{AB} \otimes |1\rangle\langle 1|_C$, which is **separable along $AB|C$** (C is unentangled from AB conditioned on its value).

    The numerical computation (see code below) gives:
    - $p_0 = 1/3$: $\rho^{(0)}_{AB}$ — **entangled** (concurrence $= 1.0$, indicating a maximally entangled state)
    - $p_1 = 2/3$: $\rho^{(1)}_{AB}$ — **unentangled** (concurrence $= 0$)

    Projective measurement of C:
    - Outcome $|0\rangle_C$ (prob $p_0 = 1/3$): $\rho^{(0)}_{AB}$ contains entanglement across the $A|B$ partition
    - Outcome $|1\rangle_C$ (prob $p_1 = 2/3$): $\rho^{(1)}_{AB}$ has no $A|B$ entanglement

    The key result: entanglement can be distilled across the $A|B$ partition conditioned on C's measurement outcome, despite the $AB|C$ partition being separable throughout.
    """)
    return


@app.cell
def _(np):
    # This response was written by claude
    # D.3.b: Apply CNOT_{B,C} and compute ρ^(0)_AB and ρ^(1)_AB

    _I = np.eye(2, dtype=complex)
    _X = np.array([[0,1],[1,0]], dtype=complex)

    # CNOT matrix: control=first qubit, target=second qubit (in 2-qubit space)
    _CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)

    # CNOT_{A,C}: acts on qubits A(0) and C(2) in ABC ordering
    # CNOT_{B,C}: acts on qubits B(1) and C(2) in ABC ordering
    def _cnot_ac(state_abc):
        """CNOT with A as control, C as target in 8-dim ABC space."""
        # Permute to put A,C adjacent, apply CNOT, permute back
        # ABC -> ACB, apply CNOT on first two, ACB -> ABC
        state = state_abc.reshape(2,2,2)  # [A,B,C]
        result = np.zeros_like(state)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    c_new = c ^ a  # CNOT: C XOR A
                    result[a, b, c_new] += state[a, b, c]
        return result.flatten()

    def _cnot_bc(state_abc):
        """CNOT with B as control, C as target in 8-dim ABC space."""
        state = state_abc.reshape(2,2,2)  # [A,B,C]
        result = np.zeros_like(state)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    c_new = c ^ b  # CNOT: C XOR B
                    result[a, b, c_new] += state[a, b, c]
        return result.flatten()

    def _apply_gate_to_dm(dm, gate_fn):
        """Apply a gate (given as state→state fn) to a density matrix."""
        d = dm.shape[0]
        # Build unitary matrix for the gate
        basis = np.eye(d, dtype=complex)
        U = np.column_stack([gate_fn(basis[:, i]) for i in range(d)])
        return U @ dm @ U.conj().T

    # Build ρ_abc from D.2: ρ = (1/6)Σ_k |Ψ_k⟩⟨Ψ_k| ⊗ |Ψ_{-k}⟩⟨Ψ_{-k}| ⊗ |0⟩⟨0|
    #                         + (1/6)Σ_{j=0,1} |j⟩⟨j| ⊗ |j⟩⟨j| ⊗ |1⟩⟨1|
    def _psi_k(k):
        return (np.array([1, np.exp(1j * k * np.pi / 2)]) / np.sqrt(2)).astype(complex)

    _ket0 = np.array([1, 0], dtype=complex)
    _ket1 = np.array([0, 1], dtype=complex)

    _rho_abc = np.zeros((8, 8), dtype=complex)
    for _k in range(4):
        _s = np.kron(np.kron(_psi_k(_k), _psi_k(-_k)), _ket0)
        _rho_abc += np.outer(_s, _s.conj()) / 6
    for _j in range(2):
        _sj = _ket0 if _j == 0 else _ket1
        _s = np.kron(np.kron(_sj, _sj), _ket1)
        _rho_abc += np.outer(_s, _s.conj()) / 6

    # Apply CNOT_{A,C} then CNOT_{B,C}
    _rho_prime = _apply_gate_to_dm(_rho_abc, _cnot_ac)
    _rho_dprime = _apply_gate_to_dm(_rho_prime, _cnot_bc)

    # Project onto C=|0⟩ and C=|1⟩
    def _partial_trace_C_project(rho_abc, c_val):
        """Project C onto |c_val⟩ and trace out C, returning (prob, rho_AB)."""
        rho = rho_abc.reshape(2, 2, 2, 2, 2, 2)  # [A,B,C,A',B',C']
        rho_ab = rho[:, :, c_val, :, :, c_val]  # shape [A,B,A',B']
        prob = np.real(np.trace(rho_ab.reshape(4, 4)))
        if prob > 1e-12:
            return prob, rho_ab.reshape(4, 4) / prob
        return prob, rho_ab.reshape(4, 4)

    _p0, _rho0_ab = _partial_trace_C_project(_rho_dprime, 0)
    _p1, _rho1_ab = _partial_trace_C_project(_rho_dprime, 1)

    def _concurrence(rho):
        """Compute concurrence of a 2-qubit mixed state."""
        _Y2 = np.array([[0,-1j],[1j,0]], dtype=complex)
        _YY = np.kron(_Y2, _Y2)
        _rho_tilde = _YY @ rho.conj() @ _YY
        _R = rho @ _rho_tilde
        _evals = np.sort(np.real(np.linalg.eigvals(_R)))[::-1]
        _evals = np.maximum(_evals, 0)
        _sqrts = np.sqrt(_evals)
        return max(0.0, float(_sqrts[0] - _sqrts[1] - _sqrts[2] - _sqrts[3]))

    print(f"p(C=|0⟩) = {_p0:.4f}, Concurrence(ρ^(0)_AB) = {_concurrence(_rho0_ab):.4f}")
    print(f"p(C=|1⟩) = {_p1:.4f}, Concurrence(ρ^(1)_AB) = {_concurrence(_rho1_ab):.4f}")
    print("ρ^(0)_AB eigenvalues:", np.sort(np.real(np.linalg.eigvals(_rho0_ab)))[::-1].round(4))
    print("ρ^(1)_AB eigenvalues:", np.sort(np.real(np.linalg.eigvals(_rho1_ab)))[::-1].round(4))
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.4
    4. With the understanding developed above, contextualize the conclusion of *Phys. Rev. Lett.* **91**, 037902 that "no entanglement is necessary to distribute entanglement". To what extent does the analogy hold between 1.) the non-uniqueness of pure-state decompositions for mixed states and 2.) the equivalence of pure states differing by a global phase. Would insertions of a global phase or exchanges of convex decompositions at intermediate points in a QI protocol be observable?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **D.4 Solution — Entanglement Not Necessary to Distribute**
    > This response was written by claude

    The paper's conclusion follows directly from the analysis: the intermediate state $\rho_{ABC}$ (the "carrier" C) remains **separable** throughout the protocol ($AB|C$ separable at every stage), yet A and B end up entangled after C's measurement is reported.

    **The analogy:**

    | Global phase irrelevance | Convex decomposition non-uniqueness |
    |---|---|
    | $\ket{\psi}$ and $e^{i\alpha}\ket{\psi}$ describe the same physics | $\rho = \sum p_i \ket{\psi_i}\bra{\psi_i} = \sum q_j \ket{\phi_j}\bra{\phi_j}$ describe the same physics |
    | No observable can distinguish them | No measurement on the system can distinguish them |
    | The "label" (phase) is inaccessible | The "label" (which decomposition) is inaccessible |

    The analogy holds precisely: just as two state vectors representing the same ray are physically equivalent, two pure-state ensembles yielding the same density matrix are physically equivalent. In both cases, the distinction exists only in our description, not in nature.

    **Observability:** Neither global phase insertions nor convex decomposition exchanges at *intermediate* steps are observable by any local or global measurement on the subsystems, because all observable predictions depend only on the density matrix. The process is well-defined and physically unambiguous regardless of which representative ensemble we use to track the state.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## VI. Glossary
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    In significant detail, physically and mathematically describe:
    #### 1
    1. The role, importance, and properties of:
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 1.a
       - Wavefunctions (continuous and discrete)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > Wavefunctions are significant in quantum systems because they mathematically encode the state and dynamics of quantum systems. They are complex-valued functions that provide a complete description of the quantum state, allowing us to calculate probabilities and expectation values for various observables. Wavefunctions can be represented in different bases, such as position or momentum space, and they evolve according to the Schrödinger equation. The properties of wavefunctions, such as normalization and superposition, are fundamental to understanding quantum phenomena like interference and entanglement.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 1.b
    Superpositions
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > Superpositions are one of the fundamental things that make quantum mechanics so different from our standard intuition about the world. The classic pop science explination of superposition is that a quantum system can be in multiple states at the same time, and only when we measure it does it "collapse" into one of those states. Instead, it is better to think of a superposition being "between" states. Where standard binary vectors lie on a hypercube, quantum states lie on a hypersphere. This ability to move between states is what allows quantum computers to be so powerful, and also what makes them so difficult to understand.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 1.c
    Entanglement
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Entanglement is the mutual dependence off two quantum systems. If you cannot describe $q_2$ without $q_1$ (often denoted by can you use tensor product to split density matrix into two independent density matrices) then $q_1$ and $q_2$ are entangled. If you can write $\rho_{q_1 q_2} = \rho_{q_1} \otimes \rho_{q_2}$ then they are not entangled.)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 1.d
    Density matrices
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Density matrices are concise ways to store knowledge about a quantum system. They are used to represent both pure and mixed states, and they allow us to calculate expectation values and probabilities for measurements. Their convexity property allows for straight forward weighted combinations
    >
    > Another way to view the density matrix is that it is an extension of teh calssical probability distrubtion that contains enough information on correlations to capture the quantum effects involved in chaning measurement basis.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 1.e
    Explain why the colloquial definition of superposition as "a phenomenon in which a particle can be in two states at the same time" lacks mathematical clarity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > It is partially unclear because the typical definition of a *state* is complete/representative information about an object. An single object can definitionally only be in one state at a time, so the idea of being in "two states at the same time" is not well-defined.

    > A better way to understand *super-position* is that a qubit has more accessible states than a normal bit. While a classical bit falls on a hypercube, qubits fall on a hypersphere. This ability to move inbetween states is what allows quantum computers to be so powerful, and also what makes them so difficult to understand.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 2
    2. The distinction between pure and mixed quantum states.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > A pure quantum state is a complex vector that represents a specific state of the quantum system; although typically described as a superposition of canonical basis states, all pure states are equally valid and do not have a privileged basis. A mixed quantum state is not a singluar vector but instead a probability distribution over the possible pure states of the system. We typically represent pure states as kets $|\psi\rangle$ and mixed states as density matrices $\rho$. The density matrix of a pure state is a rank-1 projector $\rho = |\psi\rangle\langle\psi|$, while the density matrix of a mixed state is a convex combination of projectors $\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$ where $p_i$ are probabilities. The key distinction is that pure states represent maximal knowledge about the quantum system, while mixed states represent statistical uncertainty or ignorance about the underlying pure state.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 2.a
    Describe the role of mixed states in incorporating noise or environmental interactions into simulations of quantum systems.
    """)
    return


app._unparsable_cell(
    r"""
    > Mixed states are used to represent a classical probability distribution - indicating best estimates given incomplete information. When information leaks into the enviroment, the system becomes entangled with the environment. If we only have access to the system, we must trace out the environment, which results in a mixed state description of the system. This allows us to model noise and decoherence effects in quantum systems, which are crucial for understanding real-world quantum devices and for developing error correction techniques.
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(r"""
    > Mixed states are used to represent a classical probability distribution over possible quantum states. Noise is the physical process of changing a quantum state in an unknown way, thereby creating uncertainty in the new state. The resulting uncertain state is represented as a mixed state, which is a weighted sum of the possible resulting pure states. The weights correspond to the probabilities of each outcome of the noise process.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 2.b
    If quantum systems could be perfectly isolated from their environments and manipulated with exactly unitary operators, describe any residual value provided by the mixed state formalism. What kind of states do local operators in entangled states see?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > This response was generated by claude

    Even with perfect isolation and purely unitary evolution, mixed states arise naturally as **reduced density matrices** of entangled pure states. When a composite system $AB$ is in an entangled pure state $|\psi\rangle_{AB}$, local operators on subsystem $A$ see the reduced state $\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$, which is generically **mixed**. For example, if $AB$ is in the Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$, then $\rho_A = \frac{I}{2}$ — the maximally mixed state — even though the global state is perfectly pure.

    The mixed-state formalism is therefore essential for describing subsystem physics whenever entanglement is present. It is not merely a tool for modeling noise or classical ignorance — it captures the fundamental fact that quantum correlations distribute information non-locally, leaving local observers with genuinely incomplete descriptions of their subsystems
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 3
    3. The process of quantum measurement (projective, complete/partial, destructive/non-destructive).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > Superpositions are powerful quantum states, but their full information is inaccessible to us. To extract information from a quantum state (vector on the hypersphere), we can choose an axis (eigenvalue) to measure along. The result of this will yield classical outputs in proportion to the projection of the state onto the measurement axis.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 3.a
    Describe an application in which a partial projective measurement may be computationally valuable for quantum information.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > In our long-range GHZ preparation circuit, we used local partial projective measurements to compute parity between neighboring qubits. The resulting classical information was much easier to communicate and then was applied to conditionally manipulate the quantum state of distant qubits. This allowed us to create long-range entanglement without needing direct interaction between the distant qubits, which is a common constraint in many quantum computing architectures.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 3.b

    [KLM Exercise 3.4.4] Explain why performing a complete Von Neumann measurement with respect to the computational basis and subsequently computing the parity of the resulting string is not equivalent to performing a projective measurement of the parity.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Complete projective measurements collapse the quantum state to a specific eigenstate of the measurement operator, yielding a definite outcome. In contrast, projecting into the parity space maintains the superposition of states that share the same parity, preserving quantum coherence within that subspace.

    > Complete measurement destroys any quantum correlations while a parity measurement retains those correlations
    """)
    return


if __name__ == "__main__":
    app.run()

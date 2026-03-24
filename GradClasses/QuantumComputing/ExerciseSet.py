# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.19.0",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "pylatexenc==2.10",
#     "pyzmq",
#     "qiskit==2.3.0",
#     "qiskit-aer==0.17.2",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import qiskit as qk
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.visualization import plot_histogram, plot_bloch_multivector
    import numpy as np
    import pylatexenc
    from matplotlib import pyplot as plt

    return (
        ClassicalRegister,
        QuantumCircuit,
        QuantumRegister,
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

    Now we have $d^N$ total states each with a complex number. We still have the same constraints as above so we need $d^N - 2$ real numbers to describe the state of N entangled d-level quantum systems.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.2
    2. Declaring the basis in Dirac notation ($|000\rangle, \cdots, |111\rangle$), write the vector representation of $|\psi\rangle^{\otimes 3}$ for $|\psi\rangle = \frac{|0\rangle + \sqrt{2}|1\rangle}{\sqrt{3}}$.
    """)
    return


@app.cell
def _(mo, np):
    _one = np.array([0, 1])
    _zero = np.array([1, 0])
    _psi = 1 / np.sqrt(2) * (_zero + np.sqrt(3) * _one)
    _res = np.kron(_psi, np.kron(_psi, _psi)).T
    mo.ui.matrix(_res, label="$\\psi \\otimes \\psi \\otimes \\psi$")
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$
    \begin{align*}
        \ket{000} = \sqrt{\frac{1}{2}}^3 \times \sqrt{\frac{3}{2}}^0 \\
        \ket{001} = \sqrt{\frac{1}{2}}^2 \times \sqrt{\frac{3}{2}}^1 \\
        \ket{010} = \sqrt{\frac{1}{2}}^2 \times \sqrt{\frac{3}{2}}^1 \\
        \ket{011} = \sqrt{\frac{1}{2}}^1 \times \sqrt{\frac{3}{2}}^2 \\
        \ket{100} = \sqrt{\frac{1}{2}}^2 \times \sqrt{\frac{3}{2}}^1 \\
        \ket{101} = \sqrt{\frac{1}{2}}^1 \times \sqrt{\frac{3}{2}}^2 \\
        \ket{110} = \sqrt{\frac{1}{2}}^1 \times \sqrt{\frac{3}{2}}^2 \\
        \ket{111} = \sqrt{\frac{1}{2}}^1 \times \sqrt{\frac{3}{2}}^3 \\
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


@app.cell
def _(mo):
    mo.md(r"""
    - Discuss the relationship between binary ordering of a Hilbert space basis and the tensor product operation. If two binary-ordered Hilbert spaces are expressed jointly, $H = H_1 \otimes H_2$, what is the natural ordering of the resulting $(d_1 \times d_2)$-dimensional Hilbert space?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The binary counting system. First 0 with 0-d, then 1 with 0-d, etc.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Use this understanding to write (or generate) the matrix form of the unitary operators (extending the SWAP operator) that reorder the Hilbert spaces in the following ways:
    """)
    return


@app.cell
def _(np):
    def swap_matrix(target_order: list[int], d=2) -> np.ndarray:
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

        return P

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


@app.cell
def _(mo):
    mo.md(r"""
    #### A.4
    4. By hand, write the matrix representations of $X \otimes Y$, $I_2 \otimes Y$, $Y \otimes I_2$, and $Y \otimes I_4$, where the Pauli matrices are
       $$ X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} $$
       and $I_k$ is the $k$-dimensional Identity matrix. Implement the tensor product operator in code to produce a picture of the matrix $Z \otimes X \otimes Y \otimes Z \otimes Y + X \otimes I_2 \otimes Y \otimes Z \otimes Y$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$ X \otimes Y = \begin{bmatrix}0&1\\1&0\end{bmatrix}\otimes \begin{bmatrix}0&-i\\i&0\end{bmatrix} = \begin{bmatrix} 0 & 0 & 0 & -i \\ 0 & 0 & i & 0 \\ 0 & -i & 0 & 0 \\ i & 0 & 0 & 0 \end{bmatrix} $$

    $$ I_2 \otimes Y = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \otimes \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} = \begin{bmatrix} 0 & -i & 0 & 0 \\ i & 0 & 0 & 0 \\ 0 & 0 & 0 & -i \\ 0 & 0 & i & 0 \end{bmatrix} $$

    $$ Y \otimes I_2 = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} \otimes \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & 0 & -i & 0 \\ 0 & 0 & 0 & -i \\ i & 0 & 0 & 0 \\ 0 & i & 0 & 0 \end{bmatrix} $$

    $$ Y \otimes I_4 = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} \otimes \begin{bmatrix} 1&0&0&0 \\ 0&1&0&0 \\ 0&0&1&0 \\ 0&0&0&1 \end{bmatrix} = \begin{bmatrix} 0 & 0 & 0 & 0 & -i & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & -i & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & -i & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & -i \\ i & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & i & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & i & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & i & 0 & 0 & 0 & 0 \end{bmatrix} $$
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
       $$ [A \otimes B] v = \text{vec} [B V A^T] $$
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
    $$ |v'_k\rangle \equiv |w_k\rangle - \sum_{j=1}^{k-1} \langle v_j | w_k \rangle |v_j\rangle, \quad |v_k\rangle = \frac{|v'_k\rangle}{\sqrt{\langle v'_k | v'_k \rangle}} $$
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
       $$ |\psi_1\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\0\\0\\-i\end{pmatrix}, \quad |\psi_2\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\0\\0\\1\end{pmatrix}, \quad |\psi_3\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix}0\\1\\-1\\0\end{pmatrix}, \quad |\psi_4\rangle = \frac{1}{2}\begin{pmatrix}1\\1\\1\\1\end{pmatrix} $$
    """)
    return


@app.cell
def _(np):
    _psi_1 = 1 / np.sqrt(2) * np.array([1, 0, 0, -1j])
    _psi_2 = 1 / np.sqrt(2) * np.array([1, 0, 0, 1j])
    _psi_3 = 1 / np.sqrt(2) * np.array([0, 1, -1, 0])
    _psi_4 = 1 / np.sqrt(2) * np.array([0, 1, 1, 0])

    def gram_schmidt(vectors: list[np.ndarray]) -> list[np.ndarray]:
        new_basis = []
        for vector in vectors:
            for basis_vector in new_basis:
                vector = vector - np.dot(vector, basis_vector) * basis_vector
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
    $$ i\hbar \frac{d}{dt}|\psi(t)\rangle = H |\psi(t)\rangle $$

    For a time-independent Hamiltonian $H$:
    $$ \frac{d}{dt}|\psi(t)\rangle = -\frac{i}{\hbar} H |\psi(t)\rangle $$

    This is a first-order linear differential equation of the form $\dot{y} = Ay$. The solution is:
    $$ |\psi(t)\rangle = e^{-iHt/\hbar} |\psi(0)\rangle $$

    The time evolution operator $U(t)$ matches the state at $t=0$ to $t$:
    $$ |\psi(t)\rangle = U(t) |\psi(0)\rangle $$

    Thus:
    $$ U(t) = e^{-iHt/\hbar} $$
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
    $$ U^\dagger = (e^{iA})^\dagger = e^{-iA^\dagger} = e^{-iA} $$

    Note: $(e^M)^\dagger = e^{M^\dagger}$.

    Now computing the product (since $A$ commutes with itself, $e^X e^Y = e^{X+Y}$):
    $$ U^\dagger U = e^{-iA} e^{iA} = e^{-iA + iA} = e^0 = I $$

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
    $$ U = V D V^\dagger $$
    where $D$ is a diagonal matrix containing the eigenvalues of $U$.
    Since $U$ is unitary, its eigenvalues lie on the unit circle in the complex plane. Thus, we can write the diagonal elements as:
    $$ D_{jj} = e^{i\lambda_j} $$
    where $\lambda_j \in \mathbb{R}$.

    We can define a diagonal matrix $\Lambda$ such that $\Lambda_{jj} = \lambda_j$. Then $D = e^{i\Lambda}$.

    Now, substitute back:
    $$ U = V e^{i\Lambda} V^\dagger = e^{i V \Lambda V^\dagger} $$

    Let $H = V \Lambda V^\dagger$.
    Since $\Lambda$ is real and diagonal, $\Lambda^\dagger = \Lambda$.
    $$ H^\dagger = (V \Lambda V^\dagger)^\dagger = (V^\dagger)^\dagger \Lambda^\dagger V^\dagger = V \Lambda V^\dagger = H $$
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
    $$ \frac{d}{dt} \langle A \rangle = \frac{d}{dt} \langle \psi | A | \psi \rangle = \langle \dot{\psi} | A | \psi \rangle + \langle \psi | A | \dot{\psi} \rangle $$

    Using Schrödinger eq: $|\dot{\psi}\rangle = \frac{-i}{\hbar} H |\psi\rangle$ and $\langle \dot{\psi} | = \frac{i}{\hbar} \langle \psi | H^\dagger = \frac{i}{\hbar} \langle \psi | H$.

    $$ \frac{d}{dt} \langle A \rangle = \left( \frac{i}{\hbar} \langle \psi | H \right) A | \psi \rangle + \langle \psi | A \left( \frac{-i}{\hbar} H | \psi \rangle \right) $$
    $$ = \frac{i}{\hbar} \langle \psi | (HA - AH) | \psi \rangle $$
    $$ = \frac{i}{\hbar} \langle [H, A] \rangle $$

    For $\langle A \rangle$ to be constant in time for *any* state $|\psi\rangle$, we require $\frac{d}{dt} \langle A \rangle = 0$, which implies $\langle [H, A] \rangle = 0$. Since this must hold for any state, the operator itself must be zero:
    $$ [H, A] = 0 $$

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
    $$ U e^A U^\dagger = U \left( \sum_{k=0}^\infty \frac{A^k}{k!} \right) U^\dagger = \sum_{k=0}^\infty \frac{U A^k U^\dagger}{k!} $$

    Note that $U A^k U^\dagger = (U A U^\dagger)(U A U^\dagger)...(U A U^\dagger) = (U A U^\dagger)^k$.

    $$ = \sum_{k=0}^\infty \frac{(U A U^\dagger)^k}{k!} = e^{U A U^\dagger} $$

    So this encoding is **always** possible. This means simply that changing the basis of the system rotates the Hamiltonian, and the time evolution operator rotates accordingly in the same way.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.7
    7. Given the Schrödinger equation ($\hbar = 1$) with time-independent Hamiltonian,
       $$ i \frac{\partial}{\partial t} |\psi\rangle = \hat{H} |\psi\rangle $$
       find the equation of motion (time derivative) for the time-evolution of the density matrix,
       $$ \rho(t) = \sum_{j=1}^n p_j |\psi_j(t)\rangle \langle \psi_j(t)| $$
       Does the density matrix evolve like a Heisenberg picture operator?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$ \dot{\rho}(t) = \sum_j p_j \frac{d}{dt} \left( |\psi_j\rangle \langle \psi_j| \right) $$
    $$ = \sum_j p_j \left( |\dot{\psi}_j\rangle \langle \psi_j| + |\psi_j\rangle \langle \dot{\psi}_j| \right) $$

    Substitute $|\dot{\psi}\rangle = -i H |\psi\rangle$ and $\langle \dot{\psi}| = i \langle \psi | H$:

    $$ = \sum_j p_j \left( (-i H |\psi_j\rangle) \langle \psi_j| + |\psi_j\rangle (i \langle \psi_j | H) \right) $$
    $$ = -i H \left( \sum_j p_j |\psi_j\rangle \langle \psi_j| \right) + i \left( \sum_j p_j |\psi_j\rangle \langle \psi_j| \right) H $$
    $$ = -i H \rho + i \rho H $$
    $$ = -i [H, \rho] $$

    So the equation is:
    $$ \frac{\partial \rho}{\partial t} = -i [H, \rho] $$
    (or with $\hbar$: $i \hbar \dot{\rho} = [H, \rho]$).

    **Comparison to Heisenberg Picture:**
    A Heisenberg picture operator $A_H(t)$ evolves as:
    $$ \frac{d A_H}{dt} = \frac{i}{\hbar} [H, A_H] $$
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
       $$ (\sigma_0, \vec{\sigma}) = (\sigma_0, \sigma_1, \sigma_2, \sigma_3) = \left( \begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}, \begin{pmatrix}0 & 1 \\ 1 & 0\end{pmatrix}, \begin{pmatrix}0 & -i \\ i & 0\end{pmatrix}, \begin{pmatrix}1 & 0 \\ 0 & -1\end{pmatrix} \right) $$
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
    2. Using the Kronecker delta and Levi-Civita epsilon in index notation, determine succinct expressions for the commutators, $[\sigma_i, \sigma_j]$, the anticommutators, $\{\sigma_i, \sigma_j\}$, and the operator operator products, $\sigma_i \sigma_j$, for $i, j \in \{1, 2, 3\}$. Use this to determine the maximum number of Pauli operators that can be simultaneously represented as diagonal matrices.
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
       $$ e^{i\theta \vec{v} \cdot \vec{\sigma}} = \cos(\theta)I_2 + i \sin(\theta)\vec{v} \cdot \vec{\sigma} $$
       where $\vec{v} \cdot \vec{\sigma} \equiv \sum_{j=1}^3 v_j \sigma_j$. Generalize this to show that for any function, $f(\cdot)$, that maps complex numbers to complex numbers,
       $$ f(\theta \vec{v} \cdot \vec{\sigma}) = \frac{f(\theta) + f(-\theta)}{2} I + \frac{f(\theta) - f(-\theta)}{2} \vec{v} \cdot \vec{\sigma} $$
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
    $$ f(\theta A) = \frac{f(\theta) + f(-\theta)}{2} I + \frac{f(\theta) - f(-\theta)}{2} A $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.4
    4. Pauli decompositions are often utilized in quantum computing to break computations into pieces that are manageable to implement on quantum devices. Convince yourself that a complete basis for representing generic spin-1/2 Hermitian operators is spanned by the Pauli matrices, i.e., all single-qubit operators can be expressed as a vector of Pauli coefficients. Utilizing the trace properties of the Pauli operators, devise a systematic way of determining the Pauli decomposition of an arbitrary single-qubit operator, $H_2$. Extend the technique to arbitrary $n$-qubit operators, $H_{2^n}$. Implement your scheme to determine the multi-qubit Pauli decomposition of the four-qubit operators:
       $$ \hat{\phi}(JLP) = \frac{1}{15} \text{diag}(-15, -13, \dots, 15) $$
       and
       $$ \hat{\phi}(HO) = \frac{a+a^\dagger}{\sqrt{2}} $$
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
    $$ \hat{H} = \frac{\hbar\omega}{2} (\bar{p}^2 + \bar{x}^2) $$
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

    $$ \frac{1}{\sqrt{2}} \left( \bar{x} + \frac{d}{d\bar{x}} \right) \psi_0(\bar{x}) = 0 $$
    $$ \frac{d\psi_0}{d\bar{x}} = -\bar{x} \psi_0 $$
    $$ \frac{d\psi_0}{\psi_0} = -\bar{x} d\bar{x} $$
    $$ \ln \psi_0 = -\frac{\bar{x}^2}{2} + C $$
    $$ \psi_0(\bar{x}) = A e^{-\bar{x}^2/2} $$

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
    2. Utilizing the multi-qubit local Hadamard transform, analyze the above circuit to find the final state prior to measurement. Describe the logic applied to the measurement results that allows a single implementation of this circuit to determine whether the function is constant, $f(x) = c \forall x$, or balanced. Remark upon the state remaining on the last qubit after measurement.
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

    Thus we measure with $\ket{0}$ with a probability proportional to the agreement of the function. When they are constant (i.e., $f(x) = c$ for all $x$), the amplitude is 1 and we measure $\ket{0}$ with certainty. When they are balanced (i.e., $f(x) = 1$ for half of the inputs and $f(x) = 0$ for the other half), the amplitude is 0 and we measure $\ket{1}$ with certainty. The last qubit remains in the $\ket{-}$ state throughout the algorithm and is not measured, so it does not affect the outcome of the algorithm.
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
    4. For $f(0) = 1$ and $f(1) = 0$ (negation), and $U_{f}\ket{x}\ket{y} = \ket{x}\ket{y \oplus f(x)}$. Therefore have $U_f = CNOT \cdot (I\otimes X)$ (Pauli-X followed by Pauli-Z on the target qubit).
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
    The choice function is built from exponential, so this is asympotically exponential in $n$.
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
       $$ \Delta(\hat{A})\Delta(\hat{B}) \geq \frac{1}{2} \left|\left\langle \left[\hat{A},\hat{B}\right] \right\rangle\right| $$
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

    **N&C Exercise 1.2**: Explain how a device that could perfectly distinguish between two non-orthogonal states could be used to create $U_{xerox}$ to clone them. Or alternatively, show how $U_{xerox}$ could be used to distinguish non-orthogonal states.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    TODO: idk what this is asking
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
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.2
    2. Your experimentalist friend devises a game in which one instance of a quantum state is randomly generated that is promised to be prepared in either $|\psi_1\rangle$ or $|\psi_2\rangle$. The two possible wavefunctions are provided and generated with an even 50/50 probability. You are allowed to projectively measure this instance in a basis of your choice prior to guessing the state prepared.

    - Consider the two possible states of
      $$ |\psi_1\rangle = |0\rangle \qquad |\psi_2\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}} $$
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
    If you choose to measure in $\ket{\psi_1}$ basis, you will get the following probabilities,
    $$
    \mathbb{P}(\text{distinguish}) = \braket{\psi_{1} | \psi_{2}}^2
    $$

    For the given example we have $\ket{\psi_1} = \ket{0}$ and $\ket{\psi_2} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{1})$. The probability that $\ket{psi_2}$ gives a different measurement outcome than $\ket{\psi_1}$ is:

    $$\braket{\psi_{1} | \psi_{2}}^2 = \frac{1}{2}$$

    ---

    If you choose to measure in the orthogonal basis, $\ket{\psi_1^\perp}$, you will get the following probabilities,
    $$
    \mathbb{P}(\text{distinguish}) = \braket{\psi_{1}^\perp | \psi_{2}}^2
    $$

    For the given example we have $\ket{\psi_1^\perp} = \ket{1}$ and $\ket{\psi_2} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{1})$. The probability that $\ket{psi_2}$ gives a different measurement outcome than $\ket{\psi_1^\perp}$ is:
    $$\braket{\psi_{1}^\perp | \psi_{2}}^2 = \frac{1}{2}$$

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
    # Geometric Explination for distinguising non-orthogonal states
    # Note: any two statevectors will be on the same circle, so we don't need to graph the full bloch sphere, just a circle. The optimal measurement basis to distinguis two states is the one that bisects the angle between them, so we can draw the two states as vectors on a circle and then draw the measurement basis as the line that bisects the angle between them. The probability of correctly distinguising the two states is related to the angle between them, with orthogonal states (180 degrees apart) being perfectly distinguishable and identical states (0 degrees apart) being indistinguishable.

    _measurement_basis = (
        1 / np.sqrt(2) * np.array([1, 0, 0, 1])
    )  # Bisects the angle between psi_1 and psi_2

    plt.figure(figsize=(6, 6))

    plt.plot(
        [0, _measurement_basis[0]],
        [0, _measurement_basis[3]],
        label="Measurement Basis",
        linestyle="--",
    )
    plt.legend()
    plt.title("Geometric Interpretation of State Distinguishability")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid()
    plt.axis("equal")
    plt.show()
    # TODO: this is wrong
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
    - Discuss how the odds of successful identification improve if the game provides an ensemble of two instances of the state. Should the same measurement basis be used for both measurements? If not, how would you choose the second measurement basis?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### C.3
    3. If possible, discuss experimentally how to distinguish between the following two mixed state ensembles:
    $$ \rho_1 = \frac{1}{2}|\Phi^+\rangle\langle\Phi^+| + \frac{1}{2}|\Phi^-\rangle\langle\Phi^-| $$
    $$ \rho_2 = \frac{1}{2}|00\rangle\langle00| + \frac{1}{2}|11\rangle\langle11| $$
    where $|\Phi^\pm\rangle = \frac{|00\rangle \pm |11\rangle}{\sqrt{2}}$ are two entangled Bell states. If not possible, explain why.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### D. Quantum Gate Teleportation and Compilation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.1
    1. Analyze the function of the circuit (see PDF Eq 27). Taking $\theta = \pi/4$, discuss the importance of this circuit in the context of $\{H, T\}$ being universal.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.2
    2. Fill in the details of the discussion in Section 5.3 of KLM, and complete associated exercise 5.3.1 establishing the necessary classical logic for implementing a CNOT via a four-qubit entangled state and single-qubit operators classically controlled by measurements in the Bell basis.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.3
    3. Describe the relationship between the Pauli rotation $e^{-iZ^{\otimes n} t}$ and the parity. From this perspective, analyze the circuit (see PDF Eq 28). Describe the modifications that produce circuits capable of implementing any Pauli rotation.
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
    - Tensor product Pauli operators provide a complete basis. Report how many distinct measurement ensembles would be needed to learn the density matrix as a function of $n$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.2
    - For two copies of a single-qubit state, determine and physically describe the relationship between $\langle \Psi^- | \rho \otimes \rho | \Psi^- \rangle$ and the purity.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.3
    - For an arbitrary number of qubits, motivate the relationship $\text{Tr}[\rho^2] = \text{Tr}[(\rho \otimes \rho)\text{SWAP}]$. Show how the purity information may be calculated through the statistics of a single auxiliary qubit (Hadamard test).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### B. Projecting Spatial Parity

    Under the interchange of spatial coordinates, even-parity states are unchanged while odd-parity states accrue a minus sign.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.1
    - Show that under interchange of the two Hilbert spaces, the Bell states ($|\phi^\pm\rangle$, $|\psi^\pm\rangle$) have the symmetry properties as advertised (triplet/singlet).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### B.2
    - What are the projectors of spatial parity, $P_\pm$?
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
def _(mo):
    mo.md(r"""
    #### C.2.b
       (b) Identify an operator (e.g., parity and/or built from Bell basis projectors) whose expectation values distinguish the populated states from those that remain unpopulated.
    """)
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
    #### C.3.c
       (c) Consider an alternate error mitigation technique, *twirling*, that introduces a dynamic randomization of Pauli bases throughout the time evolution to turn coherent noise into incoherent channels, in some cases destructively interfering coherent noise entirely.
       - Viable basis transformations of the noiseless operators leave invariant the logical circuits of time evolution. Find all Pauli operators, within the set of 16 $P_j = \{\mathbb{I}_2, X, Y, Z\} \otimes \{\mathbb{I}_2, X, Y, Z\}$, that satisfy the criteria $U_{\text{noiseless}} = P^\dagger U_{\text{noiseless}} P$.
       - Devise a stochastic strategy of these invariant basis changes that maintains, up to $t = 1.5$, the Bell basis measurement probabilities within $\pm 10\%$. How many layers of basis changes were used? Is the utilization of all basis transformations necessary to achieve this goal? (*Hint: while probabilistic numerical simulation could be used, try utilizing density matrix methods to capture the stochastic features of the procedure*)
    """)
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
       $$ \rho' = \mathcal{E}_{AD}(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger \qquad E_0 = \begin{pmatrix}1 & 0 \\ 0 & \sqrt{1-p}\end{pmatrix} \qquad E_1 = \begin{pmatrix}0 & \sqrt{p} \\ 0 & 0\end{pmatrix} $$
       This *channel* represents the non-unitary interaction seen from the perspective of the qubit system alone. Choose a few pure states on the surface of the Bloch sphere and plot their trajectories as a function of time (or decay time steps) when subject to this type of environment interaction.
    """)
    return


@app.cell
def _(mo, np, plot_bloch_multivector):
    # Choose |1>, |+>, S|+> as our three states (amplitude dampening)
    ad_psi_1 = np.array([0, 1])  # |1>
    ad_psi_2 = 1 / np.sqrt(2) * np.array([1, 1])  # |+>
    ad_psi_3 = 1 / np.sqrt(2) * np.array([1, 1j])  # S|+>

    # First plot each on the bloch sphere
    from qiskit.quantum_info import Statevector
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
def _(ad_psi_1, ad_psi_2, ad_psi_3, mo, np):
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
        rho = np.outer(initial_state, initial_state.conj())  # Convert state to density matrix
        p1_values = []
        for step in range(num_steps):
            p1 = rho[1, 1].real  # P(1) = rho_11
            p1_values.append(p1)
            rho = amplitude_damping_channel(rho, gamma)
        return np.array(p1_values)

    # Compute trajectories
    p1_psi1 = track_p1_trajectory(ad_psi_1, num_steps, gamma)
    p1_psi2 = track_p1_trajectory(ad_psi_2, num_steps, gamma)
    p1_psi3 = track_p1_trajectory(ad_psi_3, num_steps, gamma)

    # Plot trajectories
    import altair as alt
    import pandas as pd

    data_list = []
    for ad_idx, p1 in enumerate(p1_psi1):
        data_list.append({"step": ad_idx, "P(1)": p1, "state": "|1⟩"})
    for ad_idx, p1 in enumerate(p1_psi2):
        data_list.append({"step": ad_idx, "P(1)": p1, "state": "|+⟩"})
    for ad_idx, p1 in enumerate(p1_psi3):
        data_list.append({"step": ad_idx, "P(1)": p1, "state": "S|+⟩"})

    df = pd.DataFrame(data_list)

    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("step", title="Decay Steps"),
        y=alt.Y("P(1)", title="Probability of Measuring |1⟩", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("state", title="Initial State"),
        tooltip=["step", "P(1)", "state"]
    ).properties(
        width=600,
        height=400,
        title="Amplitude Damping: P(1) Trajectories"
    ).interactive()

    mo.ui.chart(chart)
    return alt, pd


@app.cell
def _(mo):
    mo.md(r"""
    #### A.2
    2. The phase damping channel may arise from fluctuations in unitary controls or scattering events that do not shift the state of the computational Hilbert space. Show that in the latter scenario, the qubit observes a channel of the form
       $$ \rho' = \mathcal{E}_{PD}(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger \qquad E_0 = \begin{pmatrix}1 & 0 \\ 0 & \sqrt{1-p}\end{pmatrix} \qquad E_1 = \begin{pmatrix}0 & 0 \\ 0 & \sqrt{p}\end{pmatrix} $$
       Again, choose a few pure states on the surface of the Bloch sphere and plot their trajectories as a function of time (or decay time steps) when subject to this type of environment interaction.
    """)
    return


@app.cell
def _(ad_psi_1, ad_psi_2, ad_psi_3, alt, mo, np, pd):
    # Phase Damping Channel: Plot trajectories for the three states
    def phase_damping_channel(rho, p):
        """Apply phase damping channel to density matrix rho with dephasing probability p."""
        E0 = np.array([[1, 0], [0, np.sqrt(1 - p)]])
        E1 = np.array([[0, 0], [0, np.sqrt(p)]])
        return E0 @ rho @ E0.T.conj() + E1 @ rho @ E1.T.conj()

    # Compute trajectories for phase damping
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

    # Plot coherence trajectories
    data_list_pd = []
    for pd_idx, coh in enumerate(coh_psi1):
        data_list_pd.append({"step": pd_idx, "Coherence |ρ₀₁|": coh, "state": "|1⟩"})
    for pd_idx, coh in enumerate(coh_psi2):
        data_list_pd.append({"step": pd_idx, "Coherence |ρ₀₁|": coh, "state": "|+⟩"})
    for pd_idx, coh in enumerate(coh_psi3):
        data_list_pd.append({"step": pd_idx, "Coherence |ρ₀₁|": coh, "state": "S|+⟩"})

    df_phase = pd.DataFrame(data_list_pd)

    chart_phase = alt.Chart(df_phase).mark_line(point=True).encode(
        x=alt.X("step", title="Dephasing Steps"),
        y=alt.Y("Coherence |ρ₀₁|", title="Coherence Magnitude |ρ₀₁|"),
        color=alt.Color("state", title="Initial State"),
        tooltip=["step", "Coherence |ρ₀₁|", "state"]
    ).properties(
        width=600,
        height=400,
        title="Phase Damping: Coherence Trajectories"
    ).interactive()

    mo.ui.chart(chart_phase)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### A.3
    3. When analyzing the teleportation process from the perspective of the receiver, we recognized the depolarizing channel,
       $$ \mathcal{E}_D(\rho) = p\frac{\mathbb{I}}{2} + (1-p)\rho $$
       One last time, choose a few pure states on the surface of the Bloch sphere and plot their trajectories as a function of time (or decay time steps) when subject to this type of environment interaction. In terms of the vector of Pauli matrices, $\vec{\sigma}$, find the local operators $E_k$ that express the depolarizing channel in the form $\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$.
    """)
    return


@app.cell
def _(ad_psi_1, ad_psi_2, ad_psi_3, alt, mo, np, pd):
    # Depolarizing Channel: Plot trajectories for the three states
    def depolarizing_channel(rho, p):
        """Apply depolarizing channel to density matrix rho with depolarizing probability p."""
        return p * np.eye(2) / 2 + (1 - p) * rho

    # Compute trajectories for depolarizing channel
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

    # Plot purity trajectories
    data_list_depol = []
    for dpol_idx, pur in enumerate(pur_psi1):
        data_list_depol.append({"step": dpol_idx, "Purity Tr(ρ²)": pur, "state": "|1⟩"})
    for dpol_idx, pur in enumerate(pur_psi2):
        data_list_depol.append({"step": dpol_idx, "Purity Tr(ρ²)": pur, "state": "|+⟩"})
    for dpol_idx, pur in enumerate(pur_psi3):
        data_list_depol.append({"step": dpol_idx, "Purity Tr(ρ²)": pur, "state": "S|+⟩"})

    df_depol = pd.DataFrame(data_list_depol)

    chart_depol = alt.Chart(df_depol).mark_line(point=True).encode(
        x=alt.X("step", title="Depolarization Steps"),
        y=alt.Y("Purity Tr(ρ²)", title="Purity Tr(ρ²)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("state", title="Initial State"),
        tooltip=["step", "Purity Tr(ρ²)", "state"]
    ).properties(
        width=600,
        height=400,
        title="Depolarizing Channel: Purity Trajectories"
    ).interactive()

    mo.ui.chart(chart_depol)
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
    #### A.5
    5. Generalize your unitary circuit to implement the completely depolarizing channel to a two qubit system, $\mathcal{E}(\rho) = \frac{\mathbb{I}}{4}$. How would your circuit generalize to an $n$-qubit system, $\mathcal{E}(\rho) = \frac{\mathbb{I}}{2^n}$? How many additional qubits do you anticipate using as a function of this memory capacity of your drive?
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
    #### B.2
    2. Through generalization of your methods, argue that quantum states, pure or mixed, of any qubit number have an associated orthonormal basis in which measurement outcomes are uniform, and thus minimally informative.
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
    #### B.3.a
       - State the minimum number of pure state ensembles that could be used to prepare a mixed state.
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
       $$ E_{\text{yield}} \left(\{p_i, |\psi_i\rangle\langle\psi_i|\}\right) = \sum_i p_i E\left(|\psi_i\rangle\langle\psi_i|\right) $$
       What is the $A$-$B$ entanglement yield for the GHZ and W states upon measurement of $C$ in the Z-basis?
    """)
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
    #### C.3
    3. Based on your experience, describe a $C$-collaborative protocol that begins in the $|W\rangle$ or $|GHZ\rangle$ state and deterministically produces the state $|\psi\rangle_{AB} = \frac{|00\rangle+|11\rangle}{\sqrt{2}}$.

    *Hint:* You may wish to consider a two-step procedure of the form (1) $C$-space measurement and (2) measurement-dependent $A$, $B$-space local unitaries.
    """)
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
    #### D.2
    2. Consider this separable mixture of unentangled states (*Phys. Rev. Lett.* **91**, 037902),
       $$ \rho_{abc} = \frac{1}{6}\sum_{k=0}^{3} |\Psi_k\rangle\langle\Psi_k| \otimes |\Psi_{-k}\rangle\langle\Psi_{-k}| \otimes |0\rangle\langle 0| + \frac{1}{6}\sum_{j=0}^{1} |j\rangle\langle j| \otimes |j\rangle\langle j| \otimes |1\rangle\langle 1| $$
       where the eigenvectors of the $X, Y$ Pauli operators are parameterized as
       $$ |\Psi_k\rangle = \frac{|0\rangle + e^{ik\pi/2}|1\rangle}{\sqrt{2}} $$
       List the six pure states present in the noted pure state decomposition.
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
    #### D.3.a
       - Through application of $CNOT_{A,C}$ on each pure state in the initial decomposition, show that the state after the first entangling operation is
         $$ \rho'_{ABC} = \frac{1}{6}\sum_{k=0}^{3} \rho_{AC}\!\left(\frac{|00\rangle + e^{ik\pi/2}|11\rangle}{\sqrt{2}}\right) \otimes \rho_B\!\left(|\Psi_{-k}\rangle\right) + \frac{1}{6}|0,0,1\rangle\langle 0,0,1| + \frac{1}{6}|1,1,0\rangle\langle 1,1,0| $$
         where $\rho(|\psi\rangle) = |\psi\rangle\langle\psi|$. Are there entangled pure states present in the resulting pure state convex decomposition?

       The terminology of mixed state entanglement is the following: If a convex decomposition (positive real weights $\sum_i p_i = 1$) of the density matrix composed of only tensor-product pure states exists,
       $$ \rho_{\text{separable}} = \sum_i p_i\, \rho^A_i \otimes \rho^B_i $$
       then the mixed state is considered *separable* i.e., its density matrix could be independently created without distributing entanglement. A mixed state is said to be entangled *iff* it is not separable.

       Is the resulting state, $\rho'_{ABC}$, entangled along the $AC|B$ Hilbert space partition? By analyzing the symmetry of the density matrix under the interchange of Hilbert spaces $B$ and $C$, is the state considered entangled along the $AB|C$ Hilbert space partition?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### D.3.b
       - After application of the second entangling operation $CNOT_{B,C}$, show that the final mixed state is
         $$ \rho''_{ABC} = \frac{1}{6}\sum_{k=0}^{3} \rho\!\left(\frac{1}{\sqrt{2}}\left(\frac{|00\rangle+|11\rangle}{\sqrt{2}}\right)|0\rangle_C + \frac{1}{\sqrt{2}}\left(\frac{e^{-ik\pi/2}|01\rangle+e^{ik\pi/2}|10\rangle}{\sqrt{2}}\right)|1\rangle_C\right) + \frac{1}{6}\rho(|0,0,1\rangle) + \frac{1}{6}\rho(|1,1,1\rangle) $$
         Using computational basis state projectors in the $C$ Hilbert space, show that this density matrix is equivalent to one decomposed in the form
         $$ \rho''_{ABC} = p_0\, \rho^{(0)}_{AB} \otimes |0\rangle\langle 0| + p_1\, \rho^{(1)}_{AB} \otimes |1\rangle\langle 1| $$
         and is thus considered separable along the $AB|C$ Hilbert space partition. Write the expressions for $\rho^{(0)}_{AB}$ and $\rho^{(1)}_{AB}$, are either of these entangled mixed states? Write the final pure state decompositions upon projective measurement of $|0\rangle_C$ or $|1\rangle_C$, with what probabilities would these two measurements be made? Are entangled states present across the $A|B$ partition in either measurement outcome?
    """)
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
       - Superpositions
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
       - Entanglement
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 1.d
       - Density matrices
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 1.e
       - Explain why the colloquial definition of superposition as "a phenomenon in which a particle can be in two states at the same time" lacks mathematical clarity.
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
       - Describe the role of mixed states in incorporating noise or environmental interactions into simulations of quantum systems.
    """)
    return


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
       - If quantum systems could be perfectly isolated from their environments and manipulated with exactly unitary operators, describe any residual value provided by the mixed state formalism. What kind of states do local operators in entangled states see?
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
       - Describe an application in which a partial projective measurement may be computationally valuable for quantum information.
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
 
    """)
    return


if __name__ == "__main__":
    app.run()

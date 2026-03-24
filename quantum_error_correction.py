# /// script
# dependencies = [
#     "marimo>=0.20.2",
#     "anywidget>=0.9.0",
#     "traitlets",
#     "matplotlib==3.10.8",
#     "networkx==3.6.1",
#     "numpy==2.4.3",
#     "scipy==1.17.1",
#     "pyzmq>=27.1.0",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import networkx as nx
    from collections import defaultdict
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from typing import Optional
    from scipy.optimize import linear_sum_assignment
    import anywidget
    import traitlets

    return anywidget, linear_sum_assignment, mo, np, nx, plt, traitlets


@app.cell
def _(linear_sum_assignment, np, nx):
    def repetition_parity_matrix(n: int) -> np.ndarray:
        """Parity-check matrix H for repetition code of length n. Rows: adjacent-bit parity."""
        H = np.zeros((n - 1, n), dtype=int)
        for i in range(n - 1):
            H[i, i] = 1
            H[i, i + 1] = 1
        return H

    def repetition_encode(bit: int, n: int) -> np.ndarray:
        return np.full(n, bit, dtype=int)

    def repetition_decode(codeword: np.ndarray) -> int:
        return int(np.round(codeword.sum() / len(codeword)))

    # Hamming [8,4] extended (SECDED)
    HAMMING_H = np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=int)

    def syndrome(H: np.ndarray, error: np.ndarray) -> np.ndarray:
        return (H @ error) % 2

    def build_syndrome_table(H: np.ndarray) -> list[tuple[tuple[int, ...], tuple[int, ...], int]]:
        """List of (syndrome_tuple, correction_indices_tuple, weight)."""
        n_checks, n_bits = H.shape
        table = []
        for err in range(2**n_bits):
            e = np.array([(err >> i) & 1 for i in range(n_bits)], dtype=int)
            s = tuple(syndrome(H, e).tolist())
            table.append((s, tuple(np.where(e)[0].tolist()), int(e.sum())))
        return table

    def build_surface_code_lattice(L: int):
        """Build edge list, face->edges, vertex->edges for L×L surface code. Qubits on edges."""
        edges = []  # list of ((r,c),(r,c')) or ((r,c),(r',c))
        edge_id = 0
        # horizontal edges: (r,c)-(r,c+1) for r in 0..L, c in 0..L-1
        horz = {}
        for r in range(L + 1):
            for c in range(L):
                e = ((r, c), (r, c + 1))
                edges.append(e)
                horz[(r, c), (r, c + 1)] = edge_id
                edge_id += 1
        vert = {}
        for r in range(L):
            for c in range(L + 1):
                e = ((r, c), (r + 1, c))
                edges.append(e)
                vert[(r, c), (r + 1, c)] = edge_id
                edge_id += 1
        n_qubits = len(edges)
        # faces: square (r,c) has 4 edges: top, right, bottom, left
        face_to_edges = []
        for r in range(L):
            for c in range(L):
                top = horz[((r, c), (r, c + 1))]
                bot = horz[((r + 1, c), (r + 1, c + 1))]
                left = vert[((r, c), (r + 1, c))]
                right = vert[((r, c + 1), (r + 1, c + 1))]
                face_to_edges.append([top, right, bot, left])
        # vertices: each (r,c) has incident edges
        vertex_to_edges = []
        for r in range(L + 1):
            for c in range(L + 1):
                inc = []
                if c > 0:
                    inc.append(horz[((r, c - 1), (r, c))])
                if c < L:
                    inc.append(horz[((r, c), (r, c + 1))])
                if r > 0:
                    inc.append(vert[((r - 1, c), (r, c))])
                if r < L:
                    inc.append(vert[((r, c), (r + 1, c))])
                vertex_to_edges.append(inc)
        return edges, face_to_edges, vertex_to_edges, n_qubits

    def surface_H_Z(edges: list, vertex_to_edges: list, n_qubits: int) -> np.ndarray:
        """Z-stabilizer matrix: rows = vertices, cols = qubits."""
        n_v = len(vertex_to_edges)
        H = np.zeros((n_v, n_qubits), dtype=int)
        for v, elist in enumerate(vertex_to_edges):
            for e in elist:
                H[v, e] = 1
        return H

    def surface_defect_graph(L: int, edges: list, vertex_to_edges: list):
        """Graph on vertices (0..n_vertices-1) with edge weights 1; map (va,vb) -> physical edge id."""
        n_v = (L + 1) * (L + 1)
        G = nx.Graph()
        edge_id_by_endpoints = {}
        for eid, (a, b) in enumerate(edges):
            va = a[0] * (L + 1) + a[1]
            vb = b[0] * (L + 1) + b[1]
            G.add_edge(va, vb, weight=1)
            edge_id_by_endpoints[(va, vb)] = eid
            edge_id_by_endpoints[(vb, va)] = eid
        return G, edge_id_by_endpoints

    def mwpm_correction(
        defect_vertices: list[int],
        G: nx.Graph,
        edge_id_by_endpoints: dict,
        n_qubits: int,
    ) -> tuple[list[int], int]:
        """Return (list of edge indices to flip, cost). Pairs defects by minimum total path length."""
        if len(defect_vertices) % 2 != 0:
            return [], 0
        if len(defect_vertices) == 0:
            return [], 0
        n_d = len(defect_vertices)
        # All-pairs shortest path length between defects
        cost_mat = np.zeros((n_d, n_d))
        paths = {}
        for i in range(n_d):
            for j in range(n_d):
                if i == j:
                    cost_mat[i, j] = 0
                else:
                    p = nx.shortest_path(G, defect_vertices[i], defect_vertices[j], weight="weight")
                    paths[(i, j)] = p
                    cost_mat[i, j] = len(p) - 1  # path length in edges
        # Minimum-weight perfect matching
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        correction_edges = set()
        for i, j in zip(row_ind, col_ind):
            if i >= j:
                continue
            p = paths[(i, j)]
            for k in range(len(p) - 1):
                va, vb = p[k], p[k + 1]
                eid = edge_id_by_endpoints.get((va, vb), edge_id_by_endpoints.get((vb, va)))
                if eid is not None:
                    if eid in correction_edges:
                        correction_edges.remove(eid)
                    else:
                        correction_edges.add(eid)
        return list(correction_edges), sum(cost_mat[i, j] for i, j in zip(row_ind, col_ind) if i < j)

    return (
        HAMMING_H,
        build_surface_code_lattice,
        build_syndrome_table,
        mwpm_correction,
        repetition_decode,
        repetition_encode,
        repetition_parity_matrix,
        surface_H_Z,
        surface_defect_graph,
        syndrome,
    )


@app.cell
def _():
    import json
    from data.quantum_error_correction import (
        build_tanner_widget,
        build_widget_data,
        parse_widget_list,
        ShorCodeWidget,
        SurfaceCodeWidget,
    )

    return (
        ShorCodeWidget,
        SurfaceCodeWidget,
        build_tanner_widget,
        build_widget_data,
        parse_widget_list,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quantum Error Correction

    ## What is error correction?

    When we transmit or store information over unreliable media, physical imperfections introduce *noise*: bits (or qubits) can flip or change unexpectedly. We need to *detect* or *correct* these errors without re-sending the message or restarting our calculations. **Error correction** achieves this by adding *redundancy*: we encode the message into a longer string and use *consistency checks* to spot when something went wrong.

    In this context, we distinguish between the raw storage units, called **physical bits**, and the information they represent, which we call the **logical bit**. To **detect** an error means we know something is wrong with our physical bits; to **correct** it means we can identify which physical bits flipped and fix them to recover the intended logical bit. The pattern of "which checks failed" is called a *syndrome*, and it guides us to a *correction*—a set of bits to flip to recover the intended codeword. This notebook builds from such classical ideas to quantum codes, where syndromes and corrections play the same conceptual role.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.1 A simple worked example: Repetition Code

    A simple way to protect against errors is the repetition code: represent one logical bit as $n$ physical bits (e.g., codewords 000 and 111 for $n=3$). If one physical bit flips, we might receive 100, 010, or 001 instead of 000. The decoder takes a *majority vote*: more 0s than 1s → decode to logical 0; more 1s → decode to logical 1.

    Try it yourself below. Encode a bit, manually flip some physical bits, and see how the majority vote decoder handles it. Notice that if *two* physical bits flip (e.g., receiving 110 from 000), the decoder wrongly outputs logical 1.
    """)
    return


@app.cell
def _(mo):
    simple_rep_n = mo.ui.slider(3, 11, step=2, value=3, label="Number of physical bits (n)")
    simple_encode = mo.ui.radio(options={"0": 0, "1": 1}, value="0", label="Encode logical bit", inline=True)
    return simple_encode, simple_rep_n


@app.cell
def _(mo, simple_rep_n):
    # Create checkboxes for injecting errors
    _n = simple_rep_n.value
    simple_errors = mo.ui.array([mo.ui.checkbox(label=f"Flip bit {_i}") for _i in range(_n)])
    return (simple_errors,)


@app.cell
def _(mo, simple_encode, simple_errors, simple_rep_n):
    _n = simple_rep_n.value
    _logical = simple_encode.value
    _encoded = [_logical] * _n
    _received = [ (1 - _val) if simple_errors.value[_i] else _val for _i, _val in enumerate(_encoded) ]

    _ones = sum(_received)
    _zeros = _n - _ones
    _decoded = 1 if _ones > _n / 2 else 0
    _success = _decoded == _logical

    _status = "✅ Success" if _success else "❌ Failed"
    _zeros_str = f"{_zeros} zeros ⭐" if _zeros > _ones else f"{_zeros} zeros"
    _ones_str = f"{_ones} ones ⭐" if _ones > _zeros else f"{_ones} ones"

    mo.vstack([
        mo.hstack([simple_rep_n, simple_encode]),
        mo.md("**Inject Errors:**"),
        simple_errors,
        mo.md(f"**Encoded:** `{_encoded}` → **Received:** `{_received}`"),
        mo.md(f"**Counts:** {_zeros_str}, {_ones_str}"),
        mo.md(f"**Decoded (majority vote):** `{_decoded}` ({_status})")
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Classical error correction: Redundancy and parity

    > **Goal:** Understand codewords, parity checks, and encode/decode before we introduce syndromes.

    We send *more* physical bits than the raw logical message so we can check *consistency*: more bits can flip, but we gain the ability to detect and correct. Only certain bit strings are **valid**—the **codewords**; the rest are codewords with errors applied (or invalid). A **parity check** is a linear constraint (sum of certain bits mod 2 = 0). Codewords are exactly those physical bit patterns satisfying *all* checks, i.e. the kernel of the check system. The **parity-check matrix** $H$ has one row per check and one column per bit; $H_{j,i}=1$ if check $j$ involves bit $i$. Then $H x = 0 \pmod{2}$ iff $x$ is a codeword. Given a received word, we compute which checks fail—that pattern is the **syndrome**—and use it to choose a **correction** (which physical bits to flip) so the result is a codeword; Section 3 makes decoding precise.
    """)
    return


@app.cell
def _(plt):
    import itertools as _it
    _fig = plt.figure(figsize=(3.5, 3.5))
    _ax = _fig.add_subplot(111, projection='3d')
    _ax.set_title('3-bit codewords & distance', fontsize=9)

    # Cube edges
    for _v in _it.product([0,1], repeat=3):
        for _d in range(3):
            _w = list(_v); _w[_d] ^= 1; _w = tuple(_w)
            if sum(_v) < sum(_w):
                _ax.plot([_v[0],_w[0]], [_v[1],_w[1]], [_v[2],_w[2]], color='#aaaaaa', lw=1)

    # Corners
    for _v in _it.product([0,1], repeat=3):
        _cw = _v in ((0,0,0),(1,1,1))
        _ax.scatter(*_v, color='#2ecc71' if _cw else '#cccccc', s=100 if _cw else 30, zorder=5)
        _ax.text(_v[0]+0.06, _v[1]+0.06, _v[2]+0.06, ''.join(map(str,_v)), fontsize=7, color='#222' if _cw else '#888')

    # Distance-3 path between codewords
    _ax.plot([0,1],[0,1],[0,1], color='#27ae60', lw=2, ls='--')
    _ax.text(0.5, 0.5, 0.6, 'd=3', fontsize=8, color='#27ae60', ha='center')
    _ax.set_axis_off()
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    code_type = mo.ui.radio(
        options={"Repetition code": "repetition", "Hamming [8,4]": "hamming"},
        value="Repetition code",
        label="Code type",
        inline=True,
    )
    rep_n = mo.ui.slider(3, 7, step=1, value=5, label="n")
    code_type
    return code_type, rep_n


@app.cell
def _(code_type, mo):
    code_type_desc = None
    if code_type.value == "repetition":
        code_type_desc = mo.md(r"""
    **Repetition code** (running example): encode 1 logical bit as $n$ physical copies (codewords 00…0 and 11…1). Parity checks are "adjacent bits equal" ($n-1$ checks). So $H$ has $n-1$ rows and $n$ columns. Decoding = majority vote; the code corrects up to $\lfloor (n-1)/2 \rfloor$ errors. 
    """)
    elif code_type.value == "hamming":
        code_type_desc = mo.md(r"""
    **Hamming [8,4]** uses 8 physical bits (4 data, 4 check). It is an extended Hamming code that provides SECDED (Single Error Correction, Double Error Detection). Below, choose
    """)
    else:
        code_type_desc = mo.md("IDK this code type")
    code_type_desc
    return


@app.cell
def _(code_type, np, rep_n, repetition_encode, repetition_parity_matrix):
    # Encoder: for repetition, input is one bit (we'll use a dropdown); for Hamming we could add 4-bit input
    if code_type.value == "repetition":
        n_enc = rep_n.value
        H_parity = repetition_parity_matrix(n_enc)
        codeword_0 = repetition_encode(0, n_enc)
        codeword_1 = repetition_encode(1, n_enc)
    else:
        n_enc = 8
        H_parity = np.array([
            [1, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=int)
        codeword_0 = np.zeros(8, dtype=int)
        codeword_1 = np.zeros(8, dtype=int)
    return H_parity, n_enc


@app.cell
def _(mo):
    data_bit = mo.ui.radio(options={"Encode 0": 0, "Encode 1": 1}, value="Encode 0", inline=True)
    return (data_bit,)


@app.cell
def _(anywidget, traitlets):
    class ParityCheckWidget(anywidget.AnyWidget):
        _esm = """
        function computeSyndrome(H, errors) {
          return H.map(row => row.reduce((s, h, j) => (s + h * (errors[j] || 0)) % 2, 0));
        }

        function render({ model, el }) {
          function draw() {
            const H = model.get("h_matrix");
            const errors = model.get("errors");
            const nCols = H[0]?.length || 0;
            const syn = computeSyndrome(H, errors);

            const table = document.createElement("table");

            const thead = document.createElement("thead");
            const headerRow = document.createElement("tr");
            const corner = document.createElement("th");
            corner.textContent = "H";
            headerRow.appendChild(corner);
            for (let j = 0; j < nCols; j++) {
              const th = document.createElement("th");
              th.textContent = `q${j}`;
              th.className = "bit-header" + (errors[j] ? " error-bit" : "");
              th.title = errors[j] ? "Click to clear error" : "Click to set error";
              th.addEventListener("click", () => {
                const newErrors = [...model.get("errors")];
                newErrors[j] = newErrors[j] ? 0 : 1;
                model.set("errors", newErrors);
                model.save_changes();
              });
              headerRow.appendChild(th);
            }
            thead.appendChild(headerRow);
            table.appendChild(thead);

            const tbody = document.createElement("tbody");
            for (let i = 0; i < H.length; i++) {
              const tr = document.createElement("tr");
              if (syn[i]) tr.classList.add("failed-check");
              const label = document.createElement("td");
              label.textContent = `Check ${i}`;
              label.className = "row-label";
              tr.appendChild(label);
              for (let j = 0; j < nCols; j++) {
                const td = document.createElement("td");
                td.textContent = H[i][j];
                if (H[i][j]) td.classList.add("active-cell");
                tr.appendChild(td);
              }
              tbody.appendChild(tr);
            }
            table.appendChild(tbody);

            el.innerHTML = "";
            const title = document.createElement("div");
            title.className = "pcw-title";
            title.textContent = "Parity-Check Matrix H — click a column header to toggle an error bit";
            el.appendChild(title);
            el.appendChild(table);
          }

          draw();
          model.on("change:h_matrix", draw);
          model.on("change:errors", draw);
        }

        export default { render };
        """
        _css = """
        .pcw-title { font-size: 12px; font-weight: 600; margin-bottom: 6px; color: #666; }
        table { border-collapse: collapse; font-family: monospace; font-size: 14px; }
        th, td { border: 1px solid #ccc; padding: 5px 10px; text-align: center; }
        th.bit-header { cursor: pointer; background: #e8f0fe; user-select: none; }
        th.bit-header:hover { background: #c5d8fd; }
        th.bit-header.error-bit { background: #fb8c00; color: #fff; }
        td.row-label { font-weight: 600; text-align: left; padding-right: 14px; color: #555; }
        td.active-cell { background: #bbdefb; }
        tr.failed-check td { background: #ffcdd2 !important; color: #b71c1c; }
        tr.failed-check td.active-cell { background: #e57373 !important; }
        @media (prefers-color-scheme: dark) {
          .pcw-title { color: #aaa; }
          th, td { border-color: #555; color: #ddd; background: #1e1e1e; }
          th.bit-header { background: #1a237e; color: #e8eaf6; }
          th.bit-header:hover { background: #283593; }
          th.bit-header.error-bit { background: #e65100; color: #fff; }
          td.row-label { color: #bbb; background: #1e1e1e; }
          td.active-cell { background: #0d47a1 !important; color: #fff; }
          tr.failed-check td { background: #7f0000 !important; color: #ffcdd2; }
          tr.failed-check td.active-cell { background: #c62828 !important; color: #fff; }
        }
        """
        h_matrix = traitlets.List([]).tag(sync=True)
        errors = traitlets.List([]).tag(sync=True)

    return (ParityCheckWidget,)


@app.cell
def _(H_parity, ParityCheckWidget, mo, n_enc):
    parity_widget = mo.ui.anywidget(
        ParityCheckWidget(h_matrix=H_parity.tolist(), errors=[0] * n_enc)
    )
    return (parity_widget,)


@app.cell
def _(H_parity, build_tanner_widget, mo, n_enc, parity_widget):
    _raw_fwd = build_tanner_widget(H_parity.tolist(), n_enc, mode="forward", locked=True)
    tanner_forward_widget = mo.ui.anywidget(_raw_fwd)
    bit_flip_tabs = mo.ui.tabs({
        "Parity Grid": parity_widget,
        "Check Adjacency Graph": mo.vstack([
            mo.md("Another common representation of a linear code is called a *tanner graph* which is graph with qubit nodes (click these!) and check node that react to flips"), 
            tanner_forward_widget]),
    })
    return bit_flip_tabs, tanner_forward_widget


@app.cell
def _(bit_flip_tabs):
    bit_flip_tabs
    return


@app.cell
def _(bit_flip_tabs, n_enc, parity_widget, tanner_forward_widget):
    import json as _json_bft
    if bit_flip_tabs.value == "Tanner Graph":
        _raw = tanner_forward_widget.value.get("bit_errors_json", "[]")
        try:
            _errs = _json_bft.loads(_raw) if isinstance(_raw, str) else _raw
        except (TypeError, _json_bft.JSONDecodeError):
            _errs = []
    else:
        _errs = parity_widget.value.get("errors", [])
    error_bit_values = tuple(bool(_errs[_i]) if _i < len(_errs) else False for _i in range(n_enc))
    return (error_bit_values,)


@app.cell
def _(
    H_parity,
    code_type,
    data_bit,
    error_bit_values,
    n_enc,
    np,
    repetition_decode,
    repetition_encode,
    syndrome,
):
    _error_bits = error_bit_values
    if code_type.value == "repetition":
        bit = data_bit.value
        cw = repetition_encode(bit, n_enc)
        err_enc = np.zeros(n_enc, dtype=int)
        _indices = [_i for _i in range(n_enc) if _error_bits[_i]]
        for _idx in _indices:
            err_enc[_idx] = 1
        received = (cw + err_enc) % 2
        decoded = repetition_decode(received)
        s_enc = syndrome(H_parity, err_enc)
        is_codeword = (s_enc == 0).all()
    else:
        cw = np.zeros(n_enc, dtype=int)
        err_enc = np.array([1 if _i < len(_error_bits) and _error_bits[_i] else 0 for _i in range(n_enc)], dtype=int)
        received = (cw + err_enc) % 2
        s_enc = (H_parity @ err_enc) % 2
        is_codeword = (s_enc == 0).all()
        decoded = 0
    return cw, decoded, err_enc, is_codeword, received, s_enc


@app.cell
def _(H_parity, cw, err_enc, mo, received, s_enc):
    def _to_bmatrix(arr):
        if arr.ndim == 1:
            rows = [" & ".join(str(int(x)) for x in arr)]
        else:
            rows = [" & ".join(str(int(arr[i, j])) for j in range(arr.shape[1])) for i in range(arr.shape[0])]
        return r"\begin{bmatrix} " + " \\\\ ".join(rows) + r" \end{bmatrix}"

    _H = _to_bmatrix(H_parity)
    _e_col = _to_bmatrix(err_enc.reshape(-1, 1))
    _s_col = _to_bmatrix(s_enc.reshape(-1, 1))
    _n = H_parity.shape[1]
    _m = H_parity.shape[0]
    _x_row = _to_bmatrix(cw)
    _e_row = _to_bmatrix(err_enc)
    _y_row = _to_bmatrix(received)
    _s_row = _to_bmatrix(s_enc)
    _block_h = r"$$H = " + _H + r"\quad (" + str(_m) + r"\text{ checks} \times " + str(_n) + r"\text{ bits})$$"
    _block_vec = (
        r"$$\mathbf{x}^\top = "
        + _x_row
        + r",\quad \mathbf{e}^\top = "
        + _e_row
        + r",\quad \mathbf{y}^\top = "
        + _y_row
        + r",\quad \mathbf{s}^\top = "
        + _s_row
        + r"$$"
    )
    _block_eq = r"$$\mathbf{s} = H\mathbf{e} = " + _H + r" \, " + _e_col + r" = " + _s_col + r" \pmod{2}$$"
    _intro = (
        r"**Math (concrete):** Codeword **x**, error **e**, received **y** = **x** + **e** (mod 2), "
        r"syndrome **s** = H**e** (mod 2)."
    )
    mo.vstack([
        mo.md(_block_eq), mo.md(_intro)
    ])
    return


@app.cell
def _(cw, decoded, is_codeword, mo, received, s_enc):
    mo.md(f"""
    **Encoded (codeword):** `{cw.tolist()}` → **After error:** `{received.tolist()}`  \n
    **Decoded (majority):** `{decoded}`  \n
    **Received word is a codeword?** {'Yes' if is_codeword else 'No'}  \n
    **Syndrome (which checks failed):** `{s_enc.tolist()}`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Error Identification**:

    After we receive a string, we *run all parity checks*: each row of the parity-check matrix $H$ defines a check, and we compute whether that constraint is satisfied (0) or not (1). The vector of pass/fail outcomes is the **syndrome** $s$.

    Many different errors can produce the *same* syndrome. For example, in 3-bit repetition with checks "b0+b1" and "b1+b2", an error on bit 0 gives syndrome (1,0) (first check fails); an error on bit 2 gives (0,1); an error on bit 1 gives (1,1) (both fail). So each single-bit error has a distinct syndrome.

    **Error Correction/Decoding**

    **Decoding** means: given the syndrome $s$, choose a correction $c$ (which bits to flip back), then apply $c$ to recover a codeword.

    Formally, if the received word is $y = x + e$ (codeword $x$ plus error $e$), we want $c$ such that $y + c$ is a codeword; one natural choice is $c = e$ (undo the error). We don't know $e$, but we know $s = H e$, so we choose a $c$ with $H c = s$ and use that as our correction.

    We assume that *fewer* bit flips are more likely than many (low physical error rate). Under that assumption, the most likely error that could have produced a given syndrome $s$ is one of *minimum weight* (fewest 1s) among all errors with $H e = s$. For a syndrome we compute the **canonical correction** (e.g. a minimum-weight vector $c$ with $H c = s$). This approach, known as **lowest weight decoding**, is the direct generalization of "majority voting" in the repetition code—in both cases, we choose the outcome that assumes the fewest physical errors occurred.
    """)
    return


@app.cell
def _(H_parity, build_syndrome_table):
    _table_rep = build_syndrome_table(H_parity)
    _seen = {}
    for _st, _co, _w in _table_rep:
        _key = tuple(_st)
        if _key not in _seen or _w < _seen[_key][1]:
            _seen[_key] = (_co, _w)
    _rows = [("Syndrome (binary)", "Correction (flip bits)", "Cost")]
    syndrome_tuple_to_idx = {}
    for _sy, (_co, _w) in _seen.items():
        _idx_row = len(_rows)
        _rows.append((str(_sy), str(_co), str(_w)))
        syndrome_tuple_to_idx[_sy] = _idx_row
    syndrome_table_data = _rows
    return syndrome_table_data, syndrome_tuple_to_idx


@app.cell
def _(H_parity, build_tanner_widget, mo, n_enc):
    _raw = build_tanner_widget(H_parity.tolist(), n_enc)
    tanner_widget = mo.ui.anywidget(_raw)
    tanner_raw_widget = _raw
    return (tanner_widget,)


@app.cell
def _(mo, tanner_widget):
    mo.vstack([
        mo.md("**Tanner graph:** **Forward mode** — click a bit (circle) to flip it; failed checks turn red. **Decode mode** — click a check (square) or the C0, C1, … buttons to set the syndrome; corrected bits turn orange."),
        tanner_widget,
    ])
    return


@app.cell
def _(H_parity, syndrome_table_data, syndrome_tuple_to_idx, tanner_widget):
    import json as _json
    _n_checks = H_parity.shape[0]
    _raw_val = tanner_widget.value.get("syndrome_json", "[]")
    try:
        _vals = _json.loads(_raw_val) if isinstance(_raw_val, str) else _raw_val
    except (TypeError, _json.JSONDecodeError):
        _vals = [0] * _n_checks
    _vals = list(_vals)[:_n_checks]
    _vals.extend([0] * (_n_checks - len(_vals)))
    selected_syndrome_tuple = tuple(1 if (_i < _n_checks and _vals[_i]) else 0 for _i in range(_n_checks))
    selected_row_idx = syndrome_tuple_to_idx.get(selected_syndrome_tuple, 1)
    if selected_row_idx >= len(syndrome_table_data):
        selected_row_idx = 1
    selected_syndrome_row = syndrome_table_data[selected_row_idx]
    return (selected_syndrome_row,)


@app.cell
def _(mo, selected_syndrome_row):
    _row = selected_syndrome_row
    mo.vstack([
        mo.md("**Selected syndrome → correction**"),
        mo.md(f"Syndrome = {_row[0]}, Correction = {_row[1]}, Cost = {_row[2]}."),
    ])
    return


@app.cell
def _(
    H_parity,
    build_syndrome_table,
    error_bit_values,
    mo,
    n_enc,
    np,
    syndrome,
):
    _err_vec = np.zeros(n_enc, dtype=int)
    _indices = [_i for _i in range(n_enc) if _i < len(error_bit_values) and error_bit_values[_i]]
    for _idx in _indices:
        _err_vec[_idx] = 1
    _s_tuple = tuple(syndrome(H_parity, _err_vec).tolist())
    _tbl = build_syndrome_table(H_parity)
    _match = [_t for _t in _tbl if tuple(_t[0]) == _s_tuple]
    # Filter for minimum weight if multiple entries found (though build_syndrome_table doesn't filter,
    # we filter by selecting the first matching entry with min weight)
    _res = None
    if _match:
        # Find entry in _match with minimum weight
        min_w = min(m[2] for m in _match)
        _m_best = [m for m in _match if m[2] == min_w][0]
        _, _corr, _cost = _m_best
        _res = mo.md(f"**Error injector + Decode:** Syndrome = `{_s_tuple}` → **Recommended correction:** flip bits {_corr}, **Cost** = {_cost}.")
    else:
        _res = mo.md(f"**Syndrome:** `{_s_tuple}`.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why is quantum harder?

    Just as we used multiple **physical bits** to protect one **logical bit**, we now aim to protect a **logical qubit** using many **physical qubits**. However, quantum mechanics introduces new challenges.

    We cannot simply copy a qubit (the *no-cloning* theorem), so we cannot repeat the classical "send many copies" trick in the obvious way. Worse, a qubit can suffer *two* kinds of errors: **bit-flip** (Pauli $X$, like a classical flip) and **phase-flip** (Pauli $Z$), which changes the relative phase. A single classical-style check cannot see both. If we only check "adjacent qubits equal" in the standard 0 and 1 states, a phase error can leave all those checks satisfied while corrupting the state. So we need a *second* family of checks sensitive to phase.

    Furthermore, *measuring* the data qubits directly would destroy the encoded quantum state. We must infer errors from *indirect* measurements—we measure *stabilizers* (certain invariants in the relations between qubits), not the logical qubit. Each measurement gives 1 (pass) or 0 (fail); the pattern of outcomes is our syndrome. The consequence is that protecting *one* logical qubit requires *two* syndromes: one for X-like errors and one for Z-like errors. Codes such as the **Shor code** and the **surface code** are built to provide both, as we will see.

    ## 4. From bits to qubits

    > **Goal:** One kind of check (bit-flip) is not enough for a qubit; we need two syndromes (X and Z).

    A qubit has both amplitude and phase. Errors can flip the *bit* (Pauli $X$, like a classical 0↔1 flip) or the *phase* (Pauli $Z$). Classical repetition only protects against bit-flip: we check "adjacent bits equal" in the standard 0 and 1 states. A *phase* error leaves this pattern unchanged—all those checks can still pass—while corrupting the relative phase of the state. So we are blind to Z-errors if we only use bit-flip checks. We need a *second* kind of check that is sensitive to phase. That leads to two families: **Bit-flip checks** (Z-stabilizers, detect X-errors) and **Phase-flip checks** (X-stabilizers, detect Z-errors). Both are required to protect one logical qubit.
    """)
    return


@app.cell
def _(plt):
    # Diagram comparing classical bit to quantum bit (Bloch circle)
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Classical Bit
    _ax1.plot([0, 0], [1, -1], 'k-', lw=2)
    _ax1.plot([0], [1], 'o', color='black', markersize=10)
    _ax1.plot([0], [-1], 'o', color='black', markersize=10)
    _ax1.text(0, 1.2, '1', ha='center', va='center', fontsize=16, fontweight='bold')
    _ax1.text(0, -1.2, '0', ha='center', va='center', fontsize=16, fontweight='bold')
    _ax1.annotate('', xy=(0.2, -0.8), xytext=(0.2, 0.8), arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8))
    _ax1.annotate('', xy=(-0.2, 0.8), xytext=(-0.2, -0.8), arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8))
    _ax1.text(0.4, 0, 'Bit Flip (X)', color='red', va='center', fontsize=12)
    _ax1.set_xlim(-1.5, 1.5)
    _ax1.set_ylim(-1.5, 1.5)
    _ax1.axis('off')
    _ax1.set_title('Classical Bit\n(1D line)')

    # Qubit (2D Phase space)
    _circle = plt.Circle((0, 0), 1, color='blue', fill=False, lw=2, linestyle='--')
    _ax2.add_patch(_circle)
    _ax2.plot([0, 0], [1, -1], 'o', color='black', markersize=10)
    _ax2.plot([1, -1], [0, 0], 'o', color='black', markersize=10)
    _ax2.text(0, 1.2, '|1⟩', ha='center', va='center', fontsize=14)
    _ax2.text(0, -1.2, '|0⟩', ha='center', va='center', fontsize=14)
    _ax2.text(1.2, 0, '|+⟩', ha='center', va='center', fontsize=14)
    _ax2.text(-1.2, 0, '|-⟩', ha='center', va='center', fontsize=14)

    # X flip arrow
    _ax2.annotate('', xy=(0.0, -0.8), xytext=(0.0, 0.8),
                 arrowprops=dict(edgecolor='red', facecolor='red', shrink=0.05, width=1.5, headwidth=6))
    _ax2.text(0.1, -0.4, 'Bit Flip\n(Pauli X)', color='red', va='center', fontsize=10)

    # Z flip arrow
    _ax2.annotate('', xy=(-0.8, 0.0), xytext=(0.8, 0.0),
                 arrowprops=dict(edgecolor='green', facecolor='green', shrink=0.05, width=1.5, headwidth=6))
    _ax2.text(-.4, 0.2, 'Phase Flip\n(Pauli Z)', color='green', ha='center', fontsize=10)

    _ax2.set_xlim(-1.5, 1.5)
    _ax2.set_ylim(-1.5, 1.5)
    _ax2.axis('off')
    _ax2.set_title('Quantum Bit\n(2D phase space)')

    plt.tight_layout()
    qubit_diagram = _fig
    qubit_diagram
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **Shor code** uses 9 qubits in 3 blocks of 3. Each block is a repetition code for *bit-flip* (bit-flip checks within the block). A *phase* check ties the three blocks together, so a single Z-error on any qubit flips that phase check and is detected. So we get both bit-flip and phase-flip protection. **CSS** (Calderbank–Shor–Steane) codes are the general construction: take two classical codes (one for X, one for Z) and build a quantum code. The **surface code** and **colour code** we meet later are CSS codes with a nice lattice structure. The **interactive Shor tool** below lets you choose Phase-flip checks only vs Both checks and place a Z error on any of the 9 qubits; the figure updates to show the syndrome (invisible with only one type of check, visible with both).
    """)
    return


@app.cell
def _(mo):
    section4_code_choice = mo.ui.radio(
        options={"Bit-flip checks only": "Z_only", "Both checks": "X_and_Z"},
        value="Bit-flip checks only",
        label="Code type",
        inline=True,
    )
    return (section4_code_choice,)


@app.cell
def _(ShorCodeWidget, mo, section4_code_choice):
    shor_widget = mo.ui.anywidget(ShorCodeWidget(code_type=section4_code_choice.value))
    return (shor_widget,)


@app.cell
def _(shor_widget):
    _v = shor_widget.value
    shor_error_qubit_id = _v.get("error_qubit", -1)
    shor_error_type = _v.get("error_type", "Z")
    shor_error_qubit_id = shor_error_qubit_id if shor_error_qubit_id is not None and -1 <= shor_error_qubit_id < 9 else None
    if shor_error_qubit_id == -1:
        shor_error_qubit_id = None
    return shor_error_qubit_id, shor_error_type


@app.cell
def _(
    mo,
    section4_code_choice,
    shor_error_qubit_id,
    shor_error_type,
    shor_widget,
):
    _err_i = shor_error_qubit_id
    _code = section4_code_choice.value
    if _err_i is not None:
        _type_label = "Z" if shor_error_type == "Z" else "X"
        if _code == "Z_only":
            if _type_label == "Z":
                _msg = "Z error on q{} → syndrome all 0s. Z errors are **invisible** to bit-flip checks alone.".format(_err_i)
            else:
                _msg = "X error on q{} → Bit-flip checks in the block detect it (but no phase-flip checks are available to see Z errors).".format(_err_i)
        else:
            if shor_error_type == "Z":
                _msg = "Z error on q{} → Phase-flip check fires (red). Bit-flip checks unaffected.".format(_err_i)
            else:
                _msg = "X error on q{} → Bit-flip checks fire (red). Phase-flip checks unaffected.".format(_err_i)
        syndrome_text = mo.md(_msg)
    else:
        syndrome_text = mo.md("Click a qubit to apply an error. Use the Z/X buttons to choose error type.")
    mo.vstack([
        mo.md("**Interactive Shor tool:** Choose **Code type**. Click a qubit to inject an error."),
        section4_code_choice,
        shor_widget,
        syndrome_text,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Surface code: Structure
    """)
    return


@app.cell
def _(mo):
    lattice_L = mo.ui.slider(2, 5, step=1, value=3, label="Lattice size L")
    lattice_L
    return (lattice_L,)


@app.cell
def _(build_surface_code_lattice, lattice_L):
    L = lattice_L.value
    edges, face_to_edges, vertex_to_edges, n_qubits = build_surface_code_lattice(L)
    n_faces = len(face_to_edges)
    n_vertices = len(vertex_to_edges)
    distance = L  # minimum weight logical = L for this patch
    return (
        L,
        distance,
        edges,
        face_to_edges,
        n_faces,
        n_qubits,
        n_vertices,
        vertex_to_edges,
    )


@app.cell
def _(
    L,
    SurfaceCodeWidget,
    build_widget_data,
    distance,
    edges,
    face_to_edges,
    mo,
    n_faces,
    n_qubits,
    n_vertices,
    vertex_to_edges,
):
    _edges_json, _vertices_json, _faces_json = build_widget_data(L, edges, vertex_to_edges, face_to_edges)
    preview_widget = SurfaceCodeWidget(
        L=L,
        edges_json=_edges_json,
        vertices_json=_vertices_json,
        faces_json=_faces_json,
        defects=[],
        error_edges=[],
        correction_edges=[],
        view_mode='display',
    )
    _param_str = f"**Parameters:** $n$ = {n_qubits} physical qubits, {n_faces} X-stabilizers (faces), {n_vertices} Z-stabilizers (vertices). **Distance** $d$ = {distance}."
    mo.vstack([
        mo.ui.anywidget(preview_widget),
        mo.md("**Legend:** ⚫ = **qubits** (edge), 🟦 = **bit-flip** checks (vertex), 🟥 = **phase-flip** check (faces)"),
        mo.md(_param_str),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Goal:** Get comfortable with the lattice (qubits on edges, faces, vertices), X and Z stabilizers, and distance.

    Physical qubits live on the **edges** of a 2D square grid (aka lattice). For an $L \times L$ patch there are $(L+1)L + L(L+1)$ edges (horizontal and vertical). Each **face** (square) has 4 edges and gets one **X-stabilizer**—the product of Pauli $X$ on those 4 edges. Each **vertex** (corner) has 2–4 incident edges and gets one **Z-stabilizer**—the product of Pauli $Z$ on those edges. So we have two interleaved structures: X-stabilizers are face-centered (red in the figure), Z-stabilizers are vertex-centered (blue). Use the **Lattice Preview** above to observe the layout as you adjust the lattice size.

    They appear as *chains*: e.g. a string of $Z$ along a row of edges, or $X$ along a column. The figure below shows one minimum-weight logical X (dark blue) and one logical Z (green) on the lattice.

    The **distance** $d$ is the minimum number of physical qubits you must touch starting from a valid codeword to flip the logical qubit to another valid codeword—i.e., the minimum weight of a logical operator. Larger $d$ means the code can correct more errors (ideally up to $\lfloor (d-1)/2 \rfloor$). For this $L \times L$ patch, $d \approx L$. The **Parameters** line below gives $n$ (physical qubits), the number of X- and Z-stabilizers, and $d$.
    """)
    return


@app.cell
def _(
    L,
    SurfaceCodeWidget,
    build_widget_data,
    edges,
    face_to_edges,
    vertex_to_edges,
):
    _horz_count = (L + 1) * L
    logical_Z_edges = list(range(L))
    logical_X_edges = [_horz_count + r * (L + 1) for r in range(L)]
    _edges_json, _vertices_json, _faces_json = build_widget_data(L, edges, vertex_to_edges, face_to_edges)
    logical_chains_widget = SurfaceCodeWidget(
        L=L,
        edges_json=_edges_json,
        vertices_json=_vertices_json,
        faces_json=_faces_json,
        defects=[],
        error_edges=logical_Z_edges,
        correction_edges=logical_X_edges,
        # view_mode='forward',
    )
    return (logical_chains_widget,)


@app.cell
def _(logical_chains_widget, mo):
    _desc = r"""
    **Logical operators** act on the *encoded* qubit (the logical qubit). They are intentional sequences of operations that change the underlying logical state. If the environment accidentally creates a matching sequence of physical errors, it is called a **logical error** because it passes all checks while corrupting the state.

    On the surface code they appear as **chains** of Pauli operators along paths on the lattice:
    - **Logical Z** (red edges, horizontal): a string of $Z$ along a row of edges from one boundary (side) to the opposite one. It satisfies all X-stabilizers (faces) and Z-stabilizers (vertices) and has the minimum length $L$—so the **distance** (minimum number of physical errors needed to cause a logical error) is $L$.
    - **Logical X** (cyan edges, vertical): a string of $X$ along a column of edges from one boundary to the other. Same idea: minimum-weight chain that flips the logical qubit.

    The figure below shows the lattice. One representative of each logical operator is highlighted: the horizontal chain is logical Z (orange), the vertical chain is logical X (cyan). Any chain that is **homologically equivalent** (differs only by closed loops that represent passing checks) does the same logical action; the ones drawn are the shortest. Larger $L$ gives larger distance and better error correction.

    > **CS intuition — homological equivalence as path rerouting:** Imagine two routes through a city grid between the same start and end point. If you can reroute one path into the other by swapping a detour through any block (a closed loop), they are *equivalent routes* — they go "around the same holes." In the surface code, two logical operator chains are **homologically equivalent** if one can be obtained from the other by XOR-ing in any combination of stabilizer loops. Because the stabilizers are precisely the checks the decoder passes, adding them is undetectable — so both chains flip the same logical qubit. The "holes" that can't be filled in are exactly the logical degrees of freedom the code protects.
    """
    mo.vstack([
        mo.md(_desc),
        mo.ui.anywidget(logical_chains_widget),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Surface code: Syndromes and defects

    We do *not* measure the data qubits directly—that would destroy the encoded state. Instead we measure the **stabilizers**, e.g. using ancilla qubits and gates. Each stabilizer measurement returns 0 or 1. 0 means the code state satisfies that check (meaning the bordering qubits agree, though they could still both be flipped). 1 means a check failure (a "defect").

    A **correction** identifies the most likely physical errors and undoes them—it applies the same Pauli operator to the affected qubits to cancel the error. Concretely, a correction is a set of edges to flip; it is valid when it removes all defects: $H_Z c = s \pmod{2}$.

    The syndrome structure is analogous to the puzzle *Lights Out*: each physical error (edge flip) toggles the syndrome bits at its two endpoints (adjacent vertices). Given the lit-up pattern of defects, you must infer which edges were pressed. Decoding algorithms (e.g. MWPM in Section 7) *pair* the defects and choose paths between them; the edges along those paths form the correction.

    When two adjacent errors occur (on edges sharing a vertex), they both flip the same syndrome bit—which cancels to 0. So defects only appear at the *endpoints* of an error chain, not its interior. This is why we search for *paths* connecting defect pairs: a path of edges whose endpoints are defects is the minimum explanation. Edges in the interior of the path share their syndrome contributions and cancel out.

    ### Correction: cost and validity

    The correction is **valid** if $H_Z c = s \pmod{2}$: flipping those edges removes exactly the defects we observed. Not every set of edges is valid: a random choice usually leaves some vertices still defective or creates new ones.

    The **cost** (weight) of a correction is the number of edges in it. We prefer the *minimum-weight* valid correction: under the assumption that few errors occurred, the lightest explanation is most likely.

    Because neighboring errors cancel interior defects, a correction forms a *path*: flip the edges along a path from one defect to its pair, and the interior vertices see two flips (which cancel), leaving only the endpoint vertices cleared. This is why decoders search for minimum-weight paths between paired defects rather than arbitrary edge sets.

    Use **Your correction (edges)** to select which edges to flip. The notebook checks validity and shows your cost. Use **Suggest correction** to run the decoder and see its edges (cyan on the lattice; yours are orange). Compare your cost to the optimal.
    """)
    return


@app.cell
def _(
    L,
    SurfaceCodeWidget,
    build_widget_data,
    edges,
    face_to_edges,
    mo,
    vertex_to_edges,
):
    _edges_json, _vertices_json, _faces_json = build_widget_data(L, edges, vertex_to_edges, face_to_edges)
    _w = SurfaceCodeWidget(
        L=L,
        edges_json=_edges_json,
        vertices_json=_vertices_json,
        faces_json=_faces_json,
        defects=[],
        error_edges=[],
        correction_edges=[],
        view_mode="forward",
    )
    surface_lattice_widget = mo.ui.anywidget(_w)
    _ui = mo.vstack([
        mo.md("**Interactive lattice:** Select error type (Z = phase-flip, X = bit-flip) then click qubits to inject errors. Checks light up automatically."),
        surface_lattice_widget
    ])
    _ui
    return (surface_lattice_widget,)


@app.cell
def _(parse_widget_list, surface_lattice_widget):
    defects = parse_widget_list(surface_lattice_widget.value.get("defects", "[]"))
    return (defects,)


@app.cell
def _(
    edges,
    n_qubits,
    np,
    parse_widget_list,
    surface_H_Z,
    surface_lattice_widget,
    vertex_to_edges,
):
    H_Z = surface_H_Z(edges, vertex_to_edges, n_qubits)
    your_c = np.zeros(n_qubits, dtype=int)
    for _eid in parse_widget_list(surface_lattice_widget.value.get("correction_edges", "[]")):
        if 0 <= _eid < n_qubits:
            your_c[_eid] = 1
    # valid = np.array_equal((H_Z @ your_c) % 2, defect_vec)
    # your_cost = int(your_c.sum())
    return H_Z, your_c


@app.cell
def _(mo):
    show_decoder_correction = mo.ui.checkbox(value=True, label="Show decoder correction on lattice")
    suggest_correction_btn = mo.ui.run_button(label="Suggest correction (run decoder)")
    return show_decoder_correction, suggest_correction_btn


@app.cell
def _(
    L,
    SurfaceCodeWidget,
    build_widget_data,
    defects,
    edges,
    face_to_edges,
    opt_edges,
    show_decoder_correction,
    vertex_to_edges,
    your_c,
):
    _edges_json, _vertices_json, _faces_json = build_widget_data(L, edges, vertex_to_edges, face_to_edges)
    _your_list = [i for i in range(len(your_c)) if your_c[i] == 1]
    _opt_list = opt_edges if (opt_edges is not None and show_decoder_correction.value) else []
    decoder_comparison_widget = SurfaceCodeWidget(
        L=L,
        edges_json=_edges_json,
        vertices_json=_vertices_json,
        faces_json=_faces_json,
        defects=defects,
        error_edges=_your_list,
        correction_edges=_opt_list,
    )
    return (decoder_comparison_widget,)


@app.cell
def _(H_Z, defect_vec, mo, valid, your_c):
    if False and not valid and H_Z is not None:
        _residual = (H_Z @ your_c) % 2
        _still_wrong = [str(_i) for _i in range(len(defect_vec)) if _residual[_i] != defect_vec[_i]]
        if _still_wrong:
            mo.md(f"**Invalid:** defects still present at vertices: **{', '.join(_still_wrong)}**.")
        else:
            mo.md("**Invalid:** your correction does not match the syndrome.")
    return


@app.cell
def _(
    L,
    defects,
    edges,
    mwpm_correction,
    n_qubits,
    surface_defect_graph,
    vertex_to_edges,
):
    _G_surf, _edge_id_map = surface_defect_graph(L, edges, vertex_to_edges)
    opt_edges, opt_cost = mwpm_correction(defects, _G_surf, _edge_id_map, n_qubits)
    return opt_cost, opt_edges


@app.cell
def _(mo, opt_cost, opt_edges):
    mo.md(f"""
    **Optimal correction (MWPM):** flip edges {opt_edges}, Cost = **{opt_cost}**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Decoding: Algorithms

    Four major approaches are used in practice — Lookup, MWPM, Union-Find, and Belief Propagation.
    For CSS codes (including the surface code), bit-flip (X) and phase-flip (Z) errors are decoded
    **independently**: the X-stabilisers detect Z errors and vice versa, so two separate decoder
    passes are run — one on each syndrome — and their corrections are applied together.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Lookup.** For very small codes we can precompute a table: every syndrome → one chosen
    correction (e.g. minimum weight). Decoding is: measure syndrome, look up correction, apply it.
    This notebook uses lookup for the repetition and Hamming codes in Section 3.
    It does not scale: the table size is exponential in the number of checks.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **MWPM (Minimum-Weight Perfect Matching).** For the surface code we treat defects as
    *vertices* in a graph whose edges are the physical qubit edges (weight 1). We find a
    *perfect matching* of the defect vertices that minimizes the total path length between
    matched pairs. The correction is the set of physical edges along those paths.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ```
    MWPM(defects, graph):
        G = complete_graph(defects, weighted by shortest_path)
        matching = min_weight_perfect_matching(G)
        correction = union of shortest paths for each matched pair
        return correction
    ```
    """)
    return


@app.cell
def _(mo):
    mwpm_step = mo.ui.slider(0, 3, label="MWPM step")
    return (mwpm_step,)


@app.cell
def _(mwpm_step, plt):
    import math as _math

    def draw_mwpm_step(step: int):
        """Visualise MWPM decoding on a fixed L=3 example with 2 defect pairs."""
        _L = 3
        # Fixed defect vertices (row-major index in (L+1)x(L+1) grid)
        # Use vertices at positions: (0,0)=0, (0,2)=2, (2,1)=9, (2,3)=11
        _defect_ids = [0, 2, 9, 11]
        _vpos = {}  # vid -> (x, y)
        for _r in range(_L + 1):
            for _c in range(_L + 1):
                _vid = _r * (_L + 1) + _c
                _vpos[_vid] = (_c, _L - _r)

        _fig, _ax = plt.subplots(figsize=(5, 5))
        _ax.set_xlim(-0.5, _L + 0.5)
        _ax.set_ylim(-0.5, _L + 0.5)
        _ax.set_aspect("equal")
        _ax.axis("off")
        _ax.set_title(
            ["Step 0: Defects on lattice",
             "Step 1: Complete graph on defects",
             "Step 2: Minimum-weight matching",
             "Step 3: Correction paths on lattice"][step]
        )

        # Draw lattice edges
        for _r in range(_L + 1):
            for _c in range(_L):
                _x0, _y0 = _c, _L - _r
                _ax.plot([_x0, _x0 + 1], [_y0, _y0], color="#cccccc", lw=1.5, zorder=1)
        for _r in range(_L):
            for _c in range(_L + 1):
                _x0, _y0 = _c, _L - _r
                _ax.plot([_x0, _x0], [_y0, _y0 - 1], color="#cccccc", lw=1.5, zorder=1)

        # Step 1+: complete graph overlay on defects (gray edges)
        if step >= 1:
            _pairs_all = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            for _i, _j in _pairs_all:
                _x0, _y0 = _vpos[_defect_ids[_i]]
                _x1, _y1 = _vpos[_defect_ids[_j]]
                _ax.plot([_x0, _x1], [_y0, _y1], color="#aaaaaa", lw=1, ls="--", zorder=2)

        # Step 2+: minimum-weight matching (pair 0↔2 and 1↔3)
        _matching = [(0, 2), (1, 3)]
        _match_colors = ["#e74c3c", "#3498db"]
        if step >= 2:
            for (_i, _j), _col in zip(_matching, _match_colors):
                _x0, _y0 = _vpos[_defect_ids[_i]]
                _x1, _y1 = _vpos[_defect_ids[_j]]
                _ax.plot([_x0, _x1], [_y0, _y1], color=_col, lw=3, zorder=3)

        # Step 3: correction paths on lattice (horizontal paths between matched pairs)
        if step >= 3:
            # Correction paths traced as thick segments on the lattice
            _corr_segs = [
                ((0, _L), (1, _L)), ((1, _L), (2, _L)),  # top row path
                ((1, _L - 2), (2, _L - 2)),                # lower path
            ]
            for _seg in _corr_segs:
                (_sx0, _sy0), (_sx1, _sy1) = _seg
                _ax.plot([_sx0, _sx1], [_sy0, _sy1], color="#00bcd4", lw=5, zorder=4, solid_capstyle="round")

        # Draw defect stars
        for _vid in _defect_ids:
            _x, _y = _vpos[_vid]
            _pts = []
            for _k in range(10):
                _ang = _k * _math.pi / 5 - _math.pi / 2
                _rr = 0.18 if _k % 2 == 0 else 0.08
                _pts.append((_x + _rr * _math.cos(_ang), _y + _rr * _math.sin(_ang)))
            _star = plt.Polygon(_pts, facecolor="#e74c3c", edgecolor="#b71c1c", lw=1.5, zorder=5)
            _ax.add_patch(_star)

        # Draw all vertices as small dots
        for _vid, (_vx, _vy) in _vpos.items():
            _col = "#3498db" if _vid not in _defect_ids else "#e74c3c"
            _ax.plot(_vx, _vy, "s", color=_col, markersize=6, zorder=6)

        plt.tight_layout()
        return _fig

    _mwpm_fig = draw_mwpm_step(mwpm_step.value)
    return (draw_mwpm_step,)


@app.cell
def _(draw_mwpm_step, mo, mwpm_step):
    mo.vstack([mwpm_step, draw_mwpm_step(mwpm_step.value)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Union-Find.** Grows clusters from defect vertices and merges them when paths connect
    defects, then "peels" to decide which edges to flip. Faster in practice than MWPM for
    large codes; shares the same high-level idea of connecting defects with a minimal edge set.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ```
    UNION_FIND(defects, graph):
        clusters = {d: {d} for d in defects}
        while any cluster has odd size:
            grow each cluster by 1 hop
            merge overlapping clusters
        peel each cluster to find correction edges
        return correction
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Belief propagation.** Message-passing on the Tanner graph (qubits and checks as nodes)
    is used for LDPC and other codes. It belongs to a different decoder family; see further
    reading for references.
    """)
    return


@app.cell
def _(mo):
    decoder_choice = mo.ui.radio(
        ["MWPM", "Lookup"],
        value="MWPM",
        label="Decoder",
        inline=True,
    )
    return (decoder_choice,)


@app.cell
def _(decoder_choice, defects, mo, opt_cost, opt_edges, your_cost):
    _dec = decoder_choice.value
    mo.vstack([
        mo.md(f"**Decoder:** {_dec} (both use MWPM for this surface patch)."),
        mo.md(f"**Result:** For defects = {defects}, correction edges = **{opt_edges}**, Cost = **{opt_cost}**. Your cost = **{your_cost}**."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Syndrome playground

    **Capstone.** This section ties together Sections 6 and 7. Set defects using Section 6 (presets or vertex multiselect). Choose your correction with the edge selector. Run the decoder with **Suggest correction**. Compare validity and cost, and see both your correction (orange) and the decoder's (cyan) on the lattice.

    **Learning goal.** Syndrome → validity (does your correction explain the defects?) → cost (how many edges?) → decoders (how does the optimal correction look?). The controls here are: defect presets and the interactive lattice (Section 6) to set defects and your correction, **Show decoder correction** checkbox, and **Suggest correction** button. Tweak defects and corrections until the story is clear.
    """)
    return


@app.cell
def _(
    decoder_comparison_widget,
    mo,
    opt_cost,
    opt_edges,
    show_decoder_correction,
    suggest_correction_btn,
    valid,
    your_cost,
):
    mo.vstack([
        mo.md("**Playground:** Set defects and correction on the interactive lattice (Section 6), then compare with the decoder below."),
        mo.hstack([show_decoder_correction, suggest_correction_btn]),
        mo.md(f"**Summary:** Your correction valid = **{valid}**, your cost = **{your_cost}**, optimal cost = **{opt_cost}** (decoder edges: {opt_edges})."),
        mo.ui.anywidget(decoder_comparison_widget),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Threshold: Logical vs Physical Error Rate

    Now that we understand decoding, we can ask: *how well does it work?* The answer depends on the physical error rate $p$. We plot $p$ (probability a single qubit/bit flips) against the **logical error rate** (probability the decoded logical qubit/bit is wrong after correction).

    **Threshold.** The **threshold** is the $p$ below which *increasing* code size (larger $n$ or $L$) *reduces* the logical error rate. Below threshold, bigger codes help; above threshold, bigger codes can perform worse because the extra qubits introduce more places for errors.

    **Below vs above threshold.** Below threshold, at a fixed $p$, the curve for a larger code lies below that for a smaller code—we can suppress logical errors by scaling up. Above threshold, the curves can cross: "more redundancy" is not always better.

    **Number of nines.** Engineers often express error rates using "nines": $10^{-3}$ is "three nines" ($99.9\%$ fidelity), $10^{-6}$ is "six nines". A good fault-tolerant computer needs nine or more nines of logical fidelity per gate—achievable only well below threshold.

    **Repetition (classical)** and **Surface code.** For repetition, at small $p$ longer codes suppress logical errors (majority decoding); at large $p$ longer codes can be worse. The surface code has an estimated threshold around 1% for depolarizing noise with ideal syndrome extraction. The plot here uses a simplified model (Z-errors only, perfect measurement), so the numerical threshold is illustrative rather than a precise prediction.
    """)
    return


@app.cell
def _(mo):
    ler_code_type = mo.ui.radio(
        options={"Repetition": "repetition", "Surface code": "surface"},
        value="Repetition",
        label="Code",
        inline=True,
    )
    ler_N_trials = mo.ui.slider(200, 1500, step=100, value=500, label="Trials per point")
    return ler_N_trials, ler_code_type


@app.cell
def _(
    build_surface_code_lattice,
    ler_N_trials,
    ler_code_type,
    mwpm_correction,
    np,
    repetition_decode,
    repetition_encode,
    surface_H_Z,
    surface_defect_graph,
):
    _rng = np.random.default_rng(42)
    _p_vals = np.logspace(np.log10(10e-3), np.log10(0.35), 15)
    _results = {}
    if ler_code_type.value == "repetition":
        for _n in [5, 7, 9]:
            _ler_list = []
            for _p in _p_vals:
                _errs = 0
                for _ in range(ler_N_trials.value):
                    _cw = repetition_encode(0, _n)
                    _noise = (_rng.random(_n) < _p).astype(int)
                    _rec = (_cw + _noise) % 2
                    if repetition_decode(_rec) != 0:
                        _errs += 1
                _p_hat = _errs / ler_N_trials.value
                _ler_list.append((_p_hat, np.sqrt(_p_hat * (1 - _p_hat) / ler_N_trials.value)))
            _results[f"d={_n}"] = (_p_vals, _ler_list)
    else:
        _L_vals = [2, 3, 4]
        for _L in _L_vals:
            _edges_l, _face_to_edges_l, _vertex_to_edges_l, _n_q = build_surface_code_lattice(_L)
            _H_Z_l = surface_H_Z(_edges_l, _vertex_to_edges_l, _n_q)
            _logical_Z_edges = set(range(_L))
            _G_l, _edge_id_map_l = surface_defect_graph(_L, _edges_l, _vertex_to_edges_l)
            _ler_list = []
            for _p in _p_vals:
                _errs = 0
                for _ in range(ler_N_trials.value):
                    _err_vec = (_rng.random(_n_q) < _p).astype(int)
                    _synd = (_H_Z_l @ _err_vec) % 2
                    _defects = list(np.where(_synd)[0])
                    _opt_edges, _ = mwpm_correction(_defects, _G_l, _edge_id_map_l, _n_q)
                    _residual = set(np.where(_err_vec)[0]) ^ set(_opt_edges)
                    _logical_err = (len(_residual & _logical_Z_edges) % 2) == 1
                    if _logical_err:
                        _errs += 1
                _p_hat = _errs / ler_N_trials.value
                _ler_list.append((_p_hat, np.sqrt(_p_hat * (1 - _p_hat) / ler_N_trials.value)))
            _results[f"d={_L}"] = (_p_vals, _ler_list)
    ler_plot_data = _results
    return (ler_plot_data,)


@app.cell
def _(ler_plot_data, plt):
    _fig_ler, _ax_ler = plt.subplots(figsize=(7, 4))
    _ax_ler.set_yscale("log")
    _ax_ler.set_xscale("log")
    for _label, (_p_vals, _ler_list) in ler_plot_data.items():
        _nonzero = [(p, v[0], v[1]) for p, v in zip(_p_vals, _ler_list) if v[0] > 0]
        if _nonzero:
            _px, _means, _errs = zip(*_nonzero)
            _ax_ler.errorbar(_px, _means, yerr=_errs, fmt="o-", capsize=3, label=_label)
    _ax_ler.set_xlabel("Physical error rate p (log scale)")
    _ax_ler.set_ylabel("Logical error rate (log scale)")
    _ax_ler.legend()
    _ax_ler.grid(True, alpha=0.3, which="both")
    _ax_ler.set_title("Logical error rate vs physical error rate")
    _ax_ler.axvline(x=0.1, color="gray", ls=":", lw=1.5, label="approx. threshold")
    _ax_ler.annotate("≈ threshold", xy=(0.1, 0.5), xycoords=("data", "axes fraction"),
                     fontsize=8, color="gray", ha="left", va="center")
    ler_fig = _fig_ler
    return (ler_fig,)


@app.cell
def _(ler_N_trials, ler_code_type, ler_fig, mo):
    mo.vstack([
        mo.hstack([mo.md("**Code:** "), ler_code_type, mo.md(" **Trials:** "), ler_N_trials]),
        ler_fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Generalizations

    **Parity-check matrix as universal language.** Any linear code (classical or quantum) can be described by a parity-check matrix $H$: rows = checks, columns = qubits; $H_{j,i}=1$ means check $j$ involves qubit $i$. For CSS quantum codes we have two matrices (one for X-stabilizers, one for Z). The same formalism covers repetition, Hamming, surface, and colour codes—and everything beyond.

    ### Surface code

    The surface code encodes 1 logical qubit in an $L \times L$ square lattice with distance $d = L$. Its qubits sit on edges; Z-stabilizers at vertices and X-stabilizers at face centers give a 2D nearest-neighbor connectivity, making it well-suited to superconducting qubit hardware where qubits are arranged in a grid and can only interact with their immediate neighbors.

    ### Colour code

    A **colour code** is built from a **three-colourable** tiling of the plane—e.g. hexagons coloured red, green, and blue so that no two adjacent faces share a colour. Each **face** carries *both* an X-stabilizer and a Z-stabilizer. A key advantage: colour codes support **transversal** implementation of the full Clifford group (logical gates done qubit-by-qubit without spreading errors), making them attractive for trapped-ion hardware that offers all-to-all connectivity. **Tool A — Tiling viewer** below shows a simplified R/G/B honeycomb; each coloured face corresponds to a stabilizer (X and Z on that face).
    """)
    return


@app.cell
def _(mo):
    colour_rings = mo.ui.slider(1, 3, step=1, value=2, label="Colour code rings")
    return (colour_rings,)


@app.cell
def _(plt):
    import math
    _r = 0.5
    _dx = math.sqrt(3) * _r
    _dy = _r
    _fig_cc, _ax_cc = plt.subplots(figsize=(6, 5))
    _colors_cc = ["#e74c3c", "#2ecc71", "#3498db"]
    _centers = [
        (0, 0),
        (_dx, 0),
        (_dx / 2, _dy),
        (_dx * 1.5, _dy),
        (0, 2 * _dy),
        (_dx, 2 * _dy),
        (_dx * 2, 2 * _dy),
    ]
    for _i, (_cx, _cy) in enumerate(_centers):
        _hex = plt.Polygon(
            [
                (_cx + _r * math.cos(math.pi / 6 + _j * math.pi / 3), _cy + _r * math.sin(math.pi / 6 + _j * math.pi / 3))
                for _j in range(6)
            ],
            facecolor=_colors_cc[_i % 3], edgecolor="black", linewidth=1.5,
        )
        _ax_cc.add_patch(_hex)
    _ax_cc.set_xlim(-0.6, _dx * 2.2)
    _ax_cc.set_ylim(-0.6, 2 * _dy + 0.6)
    _ax_cc.set_aspect("equal")
    _ax_cc.axis("off")
    _ax_cc.set_title("Colour code tiling (R/G/B faces); each face = X- and Z-stabilizer")
    colour_tiling_fig = _fig_cc
    return (colour_tiling_fig,)


@app.cell
def _(colour_rings, colour_tiling_fig, mo):
    mo.vstack([
        mo.md("**Tool A — Tiling viewer:** 3-coloured faces (simplified honeycomb)."),
        colour_rings,
        colour_tiling_fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Hypergraph / LDPC codes

    Beyond surface and colour codes, **arbitrary hypergraph codes** (also called quantum LDPC codes) use any sparse $H$ matrix. They offer better *encoding rate* (more logical qubits per physical qubit) and are an active research frontier. Their Tanner graphs need not come from a regular lattice.

    **Tanner graph.** The **Tanner graph** is bipartite: one part is qubit nodes, one part is check nodes; there is an edge between check $j$ and qubit $i$ if $H_{j,i}=1$. Decoding algorithms (e.g. belief propagation) run on this graph. Surface and colour codes are special cases whose Tanner graphs happen to be planar lattices.

    ### Tradeoffs

    | Code type | Connectivity | Distance | Transversal gates | Hardware fit |
    |-----------|-------------|----------|-------------------|--------------|
    | Surface | 2D nearest-neighbor | $d = L$ | T gate needs magic state distillation | Superconducting qubits |
    | Colour | All-to-all or 2D | $d \sim L$ | Full Clifford group | Trapped ions |
    | Hypergraph LDPC | Varies | Can exceed $\sqrt{n}$ | Depends on code | Near-term research |

    ### Tanner graph explorer

    The **Preset explorer** and **H → Tanner graph** tool below let you switch between repetition, Hamming, surface, and colour and see the same "code = $H$ = graph" story interactively.
    """)
    return


@app.cell
def _(mo):
    code_preset = mo.ui.radio(
        options={
            "Repetition": "repetition",
            "Hamming [8,4]": "hamming",
            "Surface code": "surface",
            "Colour code": "colour",
        },
        value="Repetition",
        label="Preset code",
        inline=True,
    )
    return (code_preset,)


@app.cell
def _(code_preset, mo, rep_n):
    _p = code_preset.value
    if _p == "repetition":
        _n, _k = rep_n.value, 1
        _txt = f"**Repetition:** n = {_n}, 1 logical bit, {_n - 1} checks. Corrects up to ⌊(n-1)/2⌋ bit-flip errors."
    elif _p == "hamming":
        _txt = "**Hamming [8,4]:** n = 8, k = 4, 4 checks. SECDED (Single Error Correction, Double Error Detection)."
    elif _p == "surface":
        _txt = "**Surface code:** n ≈ 2L²+2L (Section 5), 1 logical qubit. X- and Z-stabilizers on faces/vertices. Distance d = L."
    else:
        _txt = "**Colour code:** 3-coloured tiling (Section 10). Faces carry X and Z. Similar distance to surface for comparable size."
    mo.md("**Preset explorer:** " + _txt)
    return


@app.cell
def _(HAMMING_H, code_preset, nx, plt, rep_n, repetition_parity_matrix):
    if code_preset.value == "repetition":
        n_tan = rep_n.value
        H_tan = repetition_parity_matrix(n_tan)
    else:
        H_tan = HAMMING_H
        n_tan = H_tan.shape[1]
    m_tan = H_tan.shape[0]
    G_tan = nx.Graph()
    for _j in range(m_tan):
        G_tan.add_node(f"C{_j}", bipartite=1)
    for _i in range(n_tan):
        G_tan.add_node(f"Q{_i}", bipartite=0)
    for _j in range(m_tan):
        for _i in range(n_tan):
            if H_tan[_j, _i] == 1:
                G_tan.add_edge(f"C{_j}", f"Q{_i}")
    _fig_tan, _ax_tan = plt.subplots(figsize=(6, 4))
    _pos_tan = nx.bipartite_layout(G_tan, [f"Q{_i}" for _i in range(n_tan)])
    nx.draw(G_tan, _pos_tan, with_labels=True, node_color="lightblue", ax=_ax_tan, font_size=8)
    _ax_tan.set_title("Tanner graph (Q = qubits, C = checks)")
    tanner_fig = _fig_tan
    return (tanner_fig,)


@app.cell
def _(mo, tanner_fig):
    mo.vstack([mo.md("**Tool — H → Tanner graph**"), tanner_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Further topics

    **Stabilizer formalism.** The code space is the joint +1 eigenspace of a set of **stabilizers** (Pauli operators that commute with each other). The stabilizer group defines the code: logical operators are those that commute with all stabilizers but are not in the group. This framework unifies all the codes in this notebook.

    **Threshold and scaling.** Below threshold, the logical error rate decreases as we increase code size at fixed physical error rate $p$; scaling laws and fault-tolerant gate constructions determine how many physical qubits are needed for a target logical error rate.

    **Fault tolerance.** In real devices, errors occur during syndrome extraction too. We need repeated rounds of measurement and fault-tolerant gate constructions so that logical gates do not spread errors uncontrollably. See Nielsen & Chuang (QEC chapter), surface code reviews, and tools like **stim** for simulations.
    """)
    return


if __name__ == "__main__":
    app.run()

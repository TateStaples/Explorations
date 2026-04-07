# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.0",
#     "pyzx>=0.10.0",
#     "matplotlib>=3.7.0",
# ]
# ///
# Run: uv run marimo edit pyzx_clifford_reduce.py

import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyZX: random Clifford+T circuit and full reduce

    Generate a random Clifford+T diagram, draw it, run `zx.simplify.full_reduce`, then draw the simplified graph. Matplotlib is selected so diagrams render in marimo.
    """)
    return


@app.cell
def _():
    import pyzx as zx

    # Marimo is not always detected as a Jupyter kernel; matplotlib figures display reliably.
    # zx.settings.drawing_backend = "matplotlib"

    qubit_amount = 5
    gate_count = 10
    return gate_count, qubit_amount, zx


@app.cell
def _(gate_count, qubit_amount, zx):
    # Generate random circuit of Clifford+T gates
    circuit = zx.generate.cliffordT(qubit_amount, gate_count)
    zx.draw(circuit)
    return (circuit,)


@app.cell
def _(circuit, zx):
    zx.simplify.full_reduce(circuit)
    zx.draw(circuit)
    return


if __name__ == "__main__":
    app.run()

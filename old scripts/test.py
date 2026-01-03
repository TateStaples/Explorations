# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pyzmq",
# ]
# ///
# import marimo

import marimo
mo = marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    x = 2 * 2
    x
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

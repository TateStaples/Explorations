# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.17.0",
#     "pyzmq",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    marimo knows how your cells are related, and can automatically update
    outputs like a spreadsheet. This eliminates hidden state and hidden bugs, accelerates data exploration,
    and makes it possible for marimo to run your notebooks as scripts and web apps.
    For expensive notebooks, you can [turn this
    behavior off](https://docs.marimo.io/guides/expensive_notebooks/) via the notebook footer.

    Try updating the values of variables below and see what happens! You can also try deleting a cell.
    """)
    return


@app.cell
def _():
    x = 0
    return (x,)


@app.cell
def _():
    y = 1
    return


@app.cell
def _(x):
    x
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Tests
    this is a markdown cell

    ```python
    print("this is a code cell from Markdown")
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()

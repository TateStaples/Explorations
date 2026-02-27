# Marimo Notebook Skill

Use these patterns whenever writing or editing Marimo notebooks (`.py` files that contain `import marimo`).

## Cell Structure

Every notebook starts with a module-level `marimo.App()` instance. Each logical unit of work is a decorated function:

```python
import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)
```

### Rules for cells

- **Return every variable** that another cell will use — Marimo tracks dependencies through return values.
- **Never mutate variables** from another cell; assign to a new name instead.
- **One concern per cell**: imports, data loading, transformation, and display should each live in separate cells.
- Use `hide_code=True` on cells that only produce UI elements (markdown, charts, widgets).

```python
@app.cell(hide_code=True)
def _(mo):
    mo.md("## Section Title")
    return
```

## Markdown Cells

Use `mo.md()` with a raw string for all prose, headings, and LaTeX:

```python
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section Heading

    Explain the concept here. Math inline: $E = mc^2$

    Display math:
    $$
    \hat{\beta} = (X^T X)^{-1} X^T y
    $$
    """)
    return
```

## UI Elements & Reactivity

```python
@app.cell
def _(mo):
    slider = mo.ui.slider(start=1, stop=100, value=10, label="Sample size")
    dropdown = mo.ui.dropdown(["option A", "option B"], value="option A", label="Method")
    return slider, dropdown


@app.cell
def _(slider, dropdown):
    # React automatically when slider or dropdown change
    n = slider.value
    method = dropdown.value
    return n, method
```

Common `mo.ui` elements: `slider`, `number`, `dropdown`, `multiselect`, `checkbox`,
`radio`, `text`, `text_area`, `date`, `switch`, `button`, `file`.

## Layout

Compose outputs with layout helpers — never use `print()` for structured display:

```python
@app.cell
def _(mo, chart, table):
    mo.vstack([
        mo.md("### Results"),
        mo.hstack([chart, table], justify="start"),
    ])
    return
```

Use `mo.tabs()` for multi-panel views and `mo.accordion()` for collapsible sections:

```python
@app.cell
def _(mo, chart1, chart2):
    mo.tabs({
        "Overview": chart1,
        "Detail": chart2,
    })
    return
```

## Displaying Altair Charts

Return the chart object from the cell — Marimo renders it automatically:

```python
@app.cell
def _(alt, df):
    chart = (
        alt.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="category:N")
        .interactive()
    )
    return (chart,)
```

## Type Hints

Add type annotations to every helper function defined inside a cell:

```python
@app.cell
def _(pd):
    def load_data(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    df = load_data("data.csv")
    return (df,)
```

## Notebook Entry Point

Always end the file with:

```python
if __name__ == "__main__":
    app.run()
```

This lets the notebook run as a plain Python script (`python notebook.py`) as well as a
Marimo app (`marimo edit notebook.py` or `marimo run notebook.py`).

## Common Pitfalls

| ❌ Avoid | ✅ Prefer |
|---------|---------|
| `global x` inside a cell | Return `x` and accept it as a parameter |
| Mutating a list/dict from another cell | Create a new object |
| `print()` for display | `mo.md()`, `mo.vstack()`, return value |
| Jupyter-style `%%` magic | Plain Python inside `@app.cell` |
| Forgetting to return shared variables | Always list every exported name in `return` |

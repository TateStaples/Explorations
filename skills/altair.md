# Altair Plotting Skill

Use these patterns whenever creating data visualizations with Altair (`import altair as alt`).

## Basic Chart Anatomy

Every Altair chart follows this layered grammar:

```python
import altair as alt
import pandas as pd

chart = (
    alt.Chart(df)           # 1. Data source (DataFrame or URL)
    .mark_point()           # 2. Mark type (visual shape)
    .encode(                # 3. Encoding channels (data → visual property)
        x="column_name:Q",  # Quantitative
        y="other_col:Q",
        color="category:N", # Nominal
        size="value:Q",
        tooltip=["col1", "col2"],
    )
    .properties(title="Chart Title", width=500, height=300)
    .interactive()          # Enable pan/zoom
)
```

## Encoding Types

Always include a type suffix to avoid ambiguity:

| Suffix | Type | Example |
|--------|------|---------|
| `:Q` | Quantitative (continuous number) | `"temperature:Q"` |
| `:N` | Nominal (unordered category) | `"species:N"` |
| `:O` | Ordinal (ordered category) | `"grade:O"` |
| `:T` | Temporal (date/time) | `"date:T"` |

## Mark Types

```python
.mark_point()       # Scatter plots
.mark_line()        # Line charts
.mark_bar()         # Bar charts
.mark_area()        # Area charts
.mark_rect()        # Heatmaps / 2-D binned charts
.mark_rule()        # Reference lines
.mark_text()        # Text annotations
.mark_tick()        # Strip plots
.mark_arc()         # Pie / donut charts
.mark_boxplot()     # Box-and-whisker
```

## Common Chart Patterns

### Scatter plot with regression line

```python
base = alt.Chart(df)

scatter = base.mark_point(opacity=0.6).encode(
    x=alt.X("x:Q", title="X axis"),
    y=alt.Y("y:Q", title="Y axis"),
    color="group:N",
    tooltip=["x", "y", "group"],
)

regression = base.transform_regression("x", "y").mark_line(strokeDash=[5, 5])

chart = (scatter + regression).properties(width=500, height=350).interactive()
```

### Bar chart with sorted axis

```python
chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("count:Q", title="Count"),
        y=alt.Y("category:N", sort="-x", title=None),
        color=alt.Color("category:N", legend=None),
        tooltip=["category:N", "count:Q"],
    )
    .properties(title="Category Distribution", width=400)
)
```

### Heatmap

```python
chart = (
    alt.Chart(df)
    .mark_rect()
    .encode(
        x=alt.X("x_col:O", title="X"),
        y=alt.Y("y_col:O", title="Y"),
        color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=["x_col", "y_col", "value"],
    )
    .properties(width=400, height=300)
)
```

### Time series

```python
chart = (
    alt.Chart(df)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Value"),
        color="series:N",
        tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), "value:Q", "series:N"],
    )
    .properties(width=600, height=300)
    .interactive()
)
```

## Layering, Faceting & Concatenation

```python
# Layer two marks on the same axes
layered = scatter + regression_line

# Small multiples (facet)
faceted = (
    alt.Chart(df)
    .mark_point()
    .encode(x="x:Q", y="y:Q")
    .facet(column="category:N")
)

# Side-by-side
side_by_side = chart1 | chart2          # alt.hconcat(chart1, chart2)

# Stacked vertically
stacked = chart1 & chart2               # alt.vconcat(chart1, chart2)
```

## Interactivity & Selections

```python
# Brush selection for linked views
brush = alt.selection_interval()

points = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x="x:Q", y="y:Q",
        color=alt.condition(brush, "category:N", alt.value("lightgray")),
    )
    .add_params(brush)
)

bars = (
    alt.Chart(df)
    .mark_bar()
    .encode(x="category:N", y="count():Q")
    .transform_filter(brush)
)

dashboard = points | bars
```

## Large Datasets

When working with more than 5,000 rows, disable the row limit **once** at the top of the
notebook (or the relevant cell):

```python
alt.data_transformers.disable_max_rows()
```

## Themes & Styling

```python
# Apply a built-in theme globally
alt.themes.enable("fivethirtyeight")   # or "dark", "latimes", "urbaninstitute"

# Custom color scale
color_scale = alt.Scale(scheme="tableau10")   # or "viridis", "plasma", "blues"

# Axis and title formatting
x_enc = alt.X("value:Q", axis=alt.Axis(format=".0%", title="Percentage"))
```

## Using Altair in Marimo

Return the chart object from a cell — Marimo renders it automatically:

```python
@app.cell
def _(alt, df):
    chart = alt.Chart(df).mark_point().encode(x="x:Q", y="y:Q").interactive()
    return (chart,)
```

For a dashboard with multiple charts, compose them and return the composed object:

```python
@app.cell
def _(alt, df):
    c1 = alt.Chart(df).mark_bar().encode(x="cat:N", y="count():Q")
    c2 = alt.Chart(df).mark_point().encode(x="x:Q", y="y:Q")
    dashboard = c1 | c2
    return (dashboard,)
```

## Common Pitfalls

| ❌ Avoid | ✅ Prefer |
|---------|---------|
| Omitting type suffix (`"col"`) | Always add `:Q`, `:N`, `:O`, or `:T` |
| Using Matplotlib inside Marimo cells | Use Altair for interactive charts |
| Large DataFrames without `disable_max_rows()` | Call `alt.data_transformers.disable_max_rows()` |
| Hard-coding colors | Use `alt.Scale(scheme=...)` |
| Forgetting `.interactive()` | Add `.interactive()` for pan/zoom |

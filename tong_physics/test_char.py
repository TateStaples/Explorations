import sys

prompt = """Keep fixing, I see  in the text as mistaken formulas. Also markdown blocks in code won't render unless its at the end. To render plots do something like 

mo.vstack([
    _chart,
    mo.md("### Figure 42: Phase Diagram (p vs v)")
])"""

print(repr(prompt))

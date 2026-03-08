import json

figures_code = {
    "34": """    mo.md("### Figure 34: van der Waals Isotherms")
    _v = np.linspace(0.4, 5.0, 500)
    _T_vals = [0.85, 1.0, 1.15]
    _dfs = []
    for _T in _T_vals:
        _p = 8 * _T / (3 * _v - 1) - 3 / (_v**2)
        _df = pd.DataFrame({"v": _v, "p": _p, "T": f"T={_T}T_c"})
        _dfs.append(_df)
    _df = pd.concat(_dfs)
    _df = _df[(_df["p"] < 3.0) & (_df["p"] > -0.5)]
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x=alt.X("v:Q", title="Volume (v_r)", scale=alt.Scale(domain=[0.4, 5.0])),
        y=alt.Y("p:Q", title="Pressure (p_r)", scale=alt.Scale(domain=[-0.5, 3.0])),
        color=alt.Color("T:N", title="Isotherms", scale=alt.Scale(scheme="set1")),
        tooltip=["v", "p", "T"]
    ).properties(width=700, height=400, title="Van der Waals equation of state: Isotherms").interactive()
    return _chart,""",
    "35": """    mo.md("### Figure 35: Unstable Region at T < T_c")
    _v = np.linspace(0.4, 5.0, 500)
    _T = 0.85
    _p = 8 * _T / (3 * _v - 1) - 3 / (_v**2)
    _df = pd.DataFrame({"v": _v, "p": _p})
    _df = _df[(_df["p"] < 1.5) & (_df["p"] > -0.5)]
    _dp_dv = -24 * _T / (3 * _v - 1) ** 2 + 6 / (_v**3)
    _df["stable"] = _dp_dv <= 0
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x=alt.X("v:Q", title="Volume (v_r)"),
        y=alt.Y("p:Q", title="Pressure (p_r)"),
        color=alt.Color("stable:N", scale=alt.Scale(domain=[True, False], range=["blue", "red"]), legend=alt.Legend(title="Thermodynamically Stable")),
        tooltip=["v", "p"]
    ).properties(width=700, height=400, title="Van der Waals Isotherm at T < T_c").interactive()
    return _chart,""",
    "36": """    mo.md("### Figure 36: Maxwell Construction")
    _v = np.linspace(0.4, 5.0, 500)
    _T = 0.85
    _p = 8*_T/(3*_v - 1) - 3/(_v**2)
    _df = pd.DataFrame({'v': _v, 'p': _p})
    _df = _df[(_df['p'] < 1.5) & (_df['p'] > -0.5)]
    _p_maxwell = 0.504
    _base = alt.Chart(_df).mark_line(size=3).encode(
        x=alt.X('v:Q', title='Volume'),
        y=alt.Y('p:Q', title='Pressure'),
        tooltip=['v', 'p']
    )
    _rule = alt.Chart(pd.DataFrame({'p': [_p_maxwell]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='p:Q')
    _chart = (_base + _rule).properties(width=700, height=400, title="Maxwell Construction").interactive()
    return _chart,""",
    "37": """    mo.md("### Figure 37: Gibbs Free Energy (T > Tc)")
    _p = np.linspace(0.1, 2.0, 100)
    _g = np.log(_p)
    _df = pd.DataFrame({'p': _p, 'g': _g})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x='p:Q', y='g:Q', tooltip=['p', 'g']
    ).properties(width=700, height=400, title="Gibbs Free Energy vs Pressure (T > Tc)").interactive()
    return _chart,""",
    "38": """    mo.md("### Figure 38: Gibbs Free Energy (T < Tc)")
    _p = np.linspace(0.1, 2.0, 500)
    _g = -0.5 * (_p - 1)**3 + 0.5 * _p
    _df = pd.DataFrame({'p': _p, 'g': _g})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x='p:Q', y='g:Q', tooltip=['p', 'g']
    ).properties(width=700, height=400, title="Gibbs Free Energy vs Pressure (T < Tc)").interactive()
    return _chart,""",
    "39": """    mo.md("### Figure 39: Gibbs Free Energy - Lowest Branch")
    _p = np.linspace(0.1, 2.0, 100)
    _g = np.minimum(_p*0.5, _p*1.5 - 1)
    _df = pd.DataFrame({'p': _p, 'g': _g})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x='p:Q', y='g:Q', tooltip=['p', 'g']
    ).properties(width=700, height=400, title="Gibbs Free Energy (Lowest branch preferred)").interactive()
    return _chart,""",
    "40": """    mo.md("### Figure 40: Phase Diagram (p vs T)")
    _T = np.linspace(0.5, 1.0, 100)
    _p = np.exp(-1/_T)
    _df = pd.DataFrame({'T': _T, 'p': _p})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x='T:Q', y='p:Q', tooltip=['T', 'p']
    ).properties(width=700, height=400, title="Liquid-Gas Coexistence Curve").interactive()
    return _chart,""",
    "41": """    mo.md("### Figure 41: Critical Point Phase Diagram (p vs T)")
    _T = np.linspace(0.5, 1.0, 100)
    _p = np.exp(-1/_T)
    _df = pd.DataFrame({'T': _T, 'p': _p})
    _line = alt.Chart(_df).mark_line(size=3).encode(x=alt.X('T:Q', title="Temperature"), y=alt.Y('p:Q', title="Pressure"))
    _point = alt.Chart(pd.DataFrame({'T': [1.0], 'p': [np.exp(-1.0)]})).mark_point(color='red', size=100, filled=True).encode(x='T:Q', y='p:Q')
    _chart = (_line + _point).properties(width=700, height=400, title="Phase diagram with critical point").interactive()
    return _chart,""",
    "42": """    mo.md("### Figure 42: Phase Diagram (p vs v)")
    _v_liquid = np.linspace(0.4, 1.0, 100)
    _v_gas = np.linspace(1.0, 5.0, 100)
    _p_liquid = 1 - (_v_liquid - 1)**2
    _p_gas = 1 - 0.2*(_v_gas - 1)**2
    _df_l = pd.DataFrame({'v': _v_liquid, 'p': _p_liquid, 'Phase': 'Liquid'})
    _df_g = pd.DataFrame({'v': _v_gas, 'p': _p_gas, 'Phase': 'Gas'})
    _df = pd.concat([_df_l, _df_g])
    _df = _df[_df['p'] > 0]
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x=alt.X('v:Q', title='Volume'), y=alt.Y('p:Q', title='Pressure'), color='Phase:N'
    ).properties(width=700, height=400, title="Co-existence Region in p-v plane").interactive()
    return _chart,""",
    "43": """    mo.md("### Figure 43: 1D Ising Model Diagram")
    _spins = [1, -1, 1, 1, -1, -1, 1, -1]
    _positions = np.arange(len(_spins))
    _df = pd.DataFrame({'x': _positions, 'spin': _spins, 'label': ["↑" if s > 0 else "↓" for s in _spins]})
    _chart = alt.Chart(_df).mark_text(size=40).encode(
        x=alt.X('x:O', axis=alt.Axis(labels=False, ticks=False), title='Chain Position'),
        text='label:N', color=alt.condition(alt.datum.spin > 0, alt.value('blue'), alt.value('red'))
    ).properties(width=700, height=200, title="1D Ising Model Spin Chain").interactive()
    return _chart,""",
    "44": """    mo.md("### Figure 44: 2D Ising Model Heatmap Placeholder")
    _x, _y = np.meshgrid(np.arange(10), np.arange(10))
    _spins = np.random.choice([-1, 1], size=(10, 10))
    _df = pd.DataFrame({'x': _x.flatten(), 'y': _y.flatten(), 'spin': _spins.flatten()})
    _chart = alt.Chart(_df).mark_rect().encode(
        x='x:O', y='y:O', color=alt.Color('spin:Q', scale=alt.Scale(range=['red', 'blue'])),
        tooltip=['x', 'y', 'spin']
    ).properties(width=400, height=400, title="2D Ising Model").interactive()
    return _chart,""",
    "45": """    mo.md("### Figure 45: Mean Field Free Energy (T > Tc)")
    _m = np.linspace(-2, 2, 100)
    _f = 0.5 * _m**2 + 0.1 * _m**4
    _df = pd.DataFrame({'m': _m, 'f': _f})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x=alt.X('m:Q', title='Magnetization m'), y=alt.Y('f:Q', title='Free Energy F')
    ).properties(width=700, height=400, title="Free Energy vs Magnetization (T > Tc)").interactive()
    return _chart,""",
    "46": """    mo.md("### Figure 46: Mean Field Free Energy (T < Tc)")
    _m = np.linspace(-2, 2, 100)
    _f = -0.5 * _m**2 + 0.1 * _m**4
    _df = pd.DataFrame({'m': _m, 'f': _f})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x=alt.X('m:Q', title='Magnetization m'), y=alt.Y('f:Q', title='Free Energy F')
    ).properties(width=700, height=400, title="Free Energy vs Magnetization (T < Tc)").interactive()
    return _chart,""",
    "47": """    mo.md("### Figure 47: Mean Field Free Energy with Finite B")
    _m = np.linspace(-2, 2, 100)
    _f = -0.5 * _m**2 + 0.1 * _m**4 - 0.3 * _m
    _df = pd.DataFrame({'m': _m, 'f': _f})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x=alt.X('m:Q', title='Magnetization m'), y=alt.Y('f:Q', title='Free Energy F')
    ).properties(width=700, height=400, title="Free Energy vs Magnetization (T < Tc, B > 0)").interactive()
    return _chart,""",
    "48": """    mo.md("### Figure 48: Magnetization vs Temperature")
    _T = np.linspace(0, 2, 200)
    _m = np.where(_T < 1, np.sqrt(3*(1-_T)), 0)
    _df = pd.DataFrame({'T': _T, 'm': _m})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x=alt.X('T:Q', title='Temperature (T/Tc)'), y=alt.Y('m:Q', title='Spontaneous Magnetization')
    ).properties(width=700, height=400, title="Magnetization as a Function of Temperature").interactive()
    return _chart,""",
}


def create_marimo_notebook(json_path, out_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    with open(out_path, "w") as out:
        out.write("import marimo\n\n")
        out.write('__generated_with = "0.19.4"\n')
        out.write("app = marimo.App()\n\n")

        out.write("@app.cell\n")
        out.write("def imports():\n")
        out.write("    import marimo as mo\n")
        out.write("    import numpy as np\n")
        out.write("    import altair as alt\n")
        out.write("    import pandas as pd\n")
        out.write("    return mo, np, alt, pd\n\n")

        for page in data:
            page_text = page["text"]
            lines = page_text.split("\n")
            text_chunk = []

            for line in lines:
                if line.startswith("Figure "):
                    if text_chunk:
                        chunk_str = (
                            "\n".join(text_chunk)
                            .replace("\\", "\\\\")
                            .replace('"', '\\"')
                        )
                        out.write("@app.cell\n")
                        out.write("def _(mo):\n")
                        out.write(f'    mo.md("""{chunk_str}""")\n')
                        out.write("    return\n\n")
                        text_chunk = []

                    safe_caption = line.replace("\\", "\\\\").replace('"', '\\"')
                    import re

                    match = re.search(r"Figure (\d+)", line)
                    fig_num_str = match.group(1) if match else "0"

                    out.write("@app.cell\n")
                    out.write("def _(mo, alt, pd, np):\n")

                    if fig_num_str in figures_code:
                        out.write(figures_code[fig_num_str] + "\n\n")
                    else:
                        out.write(
                            f'    _mo_caption = mo.md("### Figure {fig_num_str}: Interactive Plot Placeholder")\n'
                        )
                        out.write(f"    _x = np.linspace(0, 10, 100)\n")
                        out.write(f"    _y = np.sin(_x + int({fig_num_str}))\n")
                        out.write(f"    _df = pd.DataFrame({{'x': _x, 'y': _y}})\n")
                        out.write(
                            f"    _chart = alt.Chart(_df).mark_line(size=3).encode(x='x:Q', y='y:Q', tooltip=['x', 'y']).properties(width=700, height=400, title='Interactive Placeholder {fig_num_str}').interactive()\n"
                        )
                        out.write(f"    return _mo_caption, _chart,\n\n")
                else:
                    text_chunk.append(line)

            if text_chunk:
                chunk_str = (
                    "\n".join(text_chunk).replace("\\", "\\\\").replace('"', '\\"')
                )
                out.write("@app.cell\n")
                out.write("def _(mo):\n")
                out.write(f'    mo.md("""{chunk_str}""")\n')
                out.write("    return\n\n")

    print(f"Created {out_path} with completely correct private variables")


create_marimo_notebook("extracted_text.json", "stat_mech.py")

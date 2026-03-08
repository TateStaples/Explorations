import re

file_path = "stat_mech.py"
with open(file_path, "r") as f:
    content = f.read()

figures = {
    "36": """    mo.md("### Figure 36: Maxwell Construction")
    v = np.linspace(0.4, 5.0, 500)
    T = 0.85
    p = 8*T/(3*v - 1) - 3/(v**2)
    df = pd.DataFrame({'v': v, 'p': p})
    df = df[(df['p'] < 1.5) & (df['p'] > -0.5)]
    p_maxwell = 0.504 # Approximate equal area pressure for T=0.85
    base = alt.Chart(df).mark_line(size=3).encode(
        x=alt.X('v:Q', title='Volume'),
        y=alt.Y('p:Q', title='Pressure'),
        tooltip=['v', 'p']
    )
    rule = alt.Chart(pd.DataFrame({'p': [p_maxwell]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='p:Q')
    chart = (base + rule).properties(width=700, height=400, title="Maxwell Construction").interactive()
    return chart,""",
    "37": """    mo.md("### Figure 37: Gibbs Free Energy (T > Tc)")
    p = np.linspace(0.1, 2.0, 100)
    g = np.log(p)
    df = pd.DataFrame({'p': p, 'g': g})
    chart = alt.Chart(df).mark_line(size=3).encode(
        x='p:Q', y='g:Q', tooltip=['p', 'g']
    ).properties(width=700, height=400, title="Gibbs Free Energy vs Pressure (T > Tc)").interactive()
    return chart,""",
    "38": """    mo.md("### Figure 38: Gibbs Free Energy (T < Tc)")
    p = np.linspace(0.1, 2.0, 500)
    g = -0.5 * (p - 1)**3 + 0.5 * p  # Example swallowtail curve
    df = pd.DataFrame({'p': p, 'g': g})
    chart = alt.Chart(df).mark_line(size=3).encode(
        x='p:Q', y='g:Q', tooltip=['p', 'g']
    ).properties(width=700, height=400, title="Gibbs Free Energy vs Pressure (T < Tc)").interactive()
    return chart,""",
    "39": """    mo.md("### Figure 39: Gibbs Free Energy - Lowest Branch")
    p = np.linspace(0.1, 2.0, 100)
    g = np.minimum(p*0.5, p*1.5 - 1)
    df = pd.DataFrame({'p': p, 'g': g})
    chart = alt.Chart(df).mark_line(size=3).encode(
        x='p:Q', y='g:Q', tooltip=['p', 'g']
    ).properties(width=700, height=400, title="Gibbs Free Energy (Lowest branch preferred)").interactive()
    return chart,""",
    "40": """    mo.md("### Figure 40: Phase Diagram (p vs T)")
    T = np.linspace(0.5, 1.0, 100)
    p = np.exp(-1/T)
    df = pd.DataFrame({'T': T, 'p': p})
    chart = alt.Chart(df).mark_line(size=3).encode(
        x='T:Q', y='p:Q', tooltip=['T', 'p']
    ).properties(width=700, height=400, title="Liquid-Gas Coexistence Curve").interactive()
    return chart,""",
    "41": """    mo.md("### Figure 41: Critical Point Phase Diagram (p vs T)")
    T = np.linspace(0.5, 1.0, 100)
    p = np.exp(-1/T)
    df = pd.DataFrame({'T': T, 'p': p})
    line = alt.Chart(df).mark_line(size=3).encode(x=alt.X('T:Q', title="Temperature"), y=alt.Y('p:Q', title="Pressure"))
    point = alt.Chart(pd.DataFrame({'T': [1.0], 'p': [np.exp(-1.0)]})).mark_point(color='red', size=100, filled=True).encode(x='T:Q', y='p:Q')
    chart = (line + point).properties(width=700, height=400, title="Phase diagram with critical point").interactive()
    return chart,""",
    "42": """    mo.md("### Figure 42: Phase Diagram (p vs v)")
    v_liquid = np.linspace(0.4, 1.0, 100)
    v_gas = np.linspace(1.0, 5.0, 100)
    p_liquid = 1 - (v_liquid - 1)**2
    p_gas = 1 - 0.2*(v_gas - 1)**2
    df_l = pd.DataFrame({'v': v_liquid, 'p': p_liquid, 'Phase': 'Liquid'})
    df_g = pd.DataFrame({'v': v_gas, 'p': p_gas, 'Phase': 'Gas'})
    df = pd.concat([df_l, df_g])
    df = df[df['p'] > 0]
    chart = alt.Chart(df).mark_line(size=3).encode(
        x=alt.X('v:Q', title='Volume'), y=alt.Y('p:Q', title='Pressure'), color='Phase:N'
    ).properties(width=700, height=400, title="Co-existence Region in p-v plane").interactive()
    return chart,""",
    "43": """    mo.md("### Figure 43: 1D Ising Model Diagram")
    spins = [1, -1, 1, 1, -1, -1, 1, -1]
    positions = np.arange(len(spins))
    df = pd.DataFrame({'x': positions, 'spin': spins, 'label': ["↑" if s > 0 else "↓" for s in spins]})
    chart = alt.Chart(df).mark_text(size=40).encode(
        x=alt.X('x:O', axis=alt.Axis(labels=False, ticks=False), title='Chain Position'),
        text='label:N', color=alt.condition(alt.datum.spin > 0, alt.value('blue'), alt.value('red'))
    ).properties(width=700, height=200, title="1D Ising Model Spin Chain").interactive()
    return chart,""",
    "45": """    mo.md("### Figure 45: Mean Field Free Energy (T > Tc)")
    m = np.linspace(-2, 2, 100)
    f = 0.5 * m**2 + 0.1 * m**4
    df = pd.DataFrame({'m': m, 'f': f})
    chart = alt.Chart(df).mark_line(size=3).encode(
        x=alt.X('m:Q', title='Magnetization m'), y=alt.Y('f:Q', title='Free Energy F')
    ).properties(width=700, height=400, title="Free Energy vs Magnetization (T > Tc)").interactive()
    return chart,""",
    "46": """    mo.md("### Figure 46: Mean Field Free Energy (T < Tc)")
    m = np.linspace(-2, 2, 100)
    f = -0.5 * m**2 + 0.1 * m**4
    df = pd.DataFrame({'m': m, 'f': f})
    chart = alt.Chart(df).mark_line(size=3).encode(
        x=alt.X('m:Q', title='Magnetization m'), y=alt.Y('f:Q', title='Free Energy F')
    ).properties(width=700, height=400, title="Free Energy vs Magnetization (T < Tc)").interactive()
    return chart,""",
    "47": """    mo.md("### Figure 47: Mean Field Free Energy with Finite B")
    m = np.linspace(-2, 2, 100)
    f = -0.5 * m**2 + 0.1 * m**4 - 0.3 * m
    df = pd.DataFrame({'m': m, 'f': f})
    chart = alt.Chart(df).mark_line(size=3).encode(
        x=alt.X('m:Q', title='Magnetization m'), y=alt.Y('f:Q', title='Free Energy F')
    ).properties(width=700, height=400, title="Free Energy vs Magnetization (T < Tc, B > 0)").interactive()
    return chart,""",
    "48": """    mo.md("### Figure 48: Magnetization vs Temperature")
    T = np.linspace(0, 2, 200)
    m = np.where(T < 1, np.sqrt(3*(1-T)), 0)
    df = pd.DataFrame({'T': T, 'm': m})
    chart = alt.Chart(df).mark_line(size=3).encode(
        x=alt.X('T:Q', title='Temperature (T/Tc)'), y=alt.Y('m:Q', title='Spontaneous Magnetization')
    ).properties(width=700, height=400, title="Magnetization as a Function of Temperature").interactive()
    return chart,""",
}


def generate_default(match):
    num = match.group(1)
    if num in figures:
        return figures[num]

    # Generate generic placeholders for the rest
    return f"""    mo.md("### Figure {num}: Interactive Plot Placeholder")
    x = np.linspace(0, 10, 100)
    y = np.sin(x + int({num}))
    df = pd.DataFrame({{'x': x, 'y': y}})
    chart = alt.Chart(df).mark_line(size=3).encode(
        x='x:Q', y='y:Q', tooltip=['x', 'y']
    ).properties(width=700, height=400, title="Figure {num} Interactive Placeholder").interactive()
    return chart,"""


new_content = re.sub(
    r'    mo\.md\("Interactive Figure (\d+)"\)\n    # Figure \1:\n    # TODO: Implement interactive plot\n    return',
    generate_default,
    content,
)

with open(file_path, "w") as f:
    f.write(new_content)

print("Applied interactive figures to stat_mech.py")

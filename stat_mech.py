import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd
    return alt, mo, np, pd


@app.cell
def _(mo):
    mo.md("""
    5. Phase Transitions
    A phase transition is an abrupt, discontinuous change in the properties of a system.
    We’ve already seen one example of a phase transition in our discussion of Bose-Einstein
    condensation. In that case, we had to look fairly closely to see the discontinuity: it was
    lurking in the derivative of the heat capacity. In other phase transitions — many of
    them already familiar — the discontinuity is more manifest. Examples include steam
    condensing to water and water freezing to ice.
    In this section we’ll explore a couple of phase transitions in some detail and extract
    some lessons that are common to all transitions.
    5.1 Liquid-Gas Transition
    Recall that we derived the van der Waals equation of state for a gas (2.31) in Section
    2.5. We can write the van der Waals equation as
    p = kBT
    v −b −a
    v2
    (5.1)
    where v = V/N is the volume per particle. In the literature, you will also see this
    equation written in terms of the particle density ⇢= 1/v.
    On the right we ﬁx T at di↵erent values
    b
    p
    v
    T>T
    T=T
    T<T
    c
    c
    c
    c
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 34: van der Waals Isotherms")
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
    return


@app.cell
def _(mo):
    mo.md("""
    and sketch the graph of p vs. V determined
    by the van der Waals equation. These curves
    are isotherms — lines of constant tempera-
    ture. As we can see from the diagram, the
    isotherms take three di↵erent shapes depend-
    ing on the value of T. The top curve shows
    the isotherm for large values of T. Here we
    can e↵ectively ignore the −a/v2 term. (Recall
    that v cannot take values smaller than b, re-
    ﬂecting the fact that atoms cannot approach
    to arbitrarily closely.) The result is a monotonically decreasing function, essentially the
    same as we would get for an ideal gas. In contrast, when T is low enough, the second
    term in (5.1) can compete with the ﬁrst term. Roughly speaking, this happens when
    kBT ⇠a/v is in the allowed region v > b. For these low value of the temperature, the
    isotherm has a wiggle.
    – 135 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    At some intermediate temperature, the wiggle must ﬂatten out so that the bottom
    curve looks like the top one. This happens when the maximum and minimum meet
    to form an inﬂection point. Mathematically, we are looking for a solution to dp/dv =
    d2p/dv2 = 0. It is simple to check that these two equations only have a solution at the
    critical temperature T = Tc given by
    kBTc = 8a
    27b
    (5.2)
    Let’s look in more detail at the T < Tc curve. For a range of pressures, the system can
    have three di↵erent choices of volume. A typical, albeit somewhat exagerated, example
    of this curve is shown in the ﬁgure below. What’s going on? How should we interpret
    the fact that the system can seemingly live at three di↵erent densities ⇢= 1/v?
    First look at the middle solution. This has
    b
    p
    T<Tc
    v
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 35: Unstable Region at T < T_c")
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
    return


@app.cell
def _(mo):
    mo.md("""
    some fairly weird properties. We can see from the
    graph that the gradient is positive: dp/dv|T > 0.
    This means that if we apply a force to the con-
    tainer to squeeze the gas, the pressure decreases.
    The gas doesn’t push back; it just relents. But if
    we expand the gas, the pressure increases and the
    gas pushes harder. Both of these properties are
    telling us that the gas in that state is unstable.
    If we were able to create such a state, it wouldn’t hang around for long because any
    tiny perturbation would lead to a rapid, explosive change in its density. If we want to
    ﬁnd states which we are likely to observe in Nature then we should look at the other
    two solutions.
    The solution to the left on the graph has v slightly bigger than b. But, recall from
    our discussion of Section 2.5 that b is the closest that the atoms can get. If we have
    v ⇠b, then the atoms are very densely packed. Moreover, we can also see from the
    graph that |dp/dv| is very large for this solution which means that the state is very
    diﬃcult to compress: we need to add a great deal of pressure to change the volume
    only slightly. We have a name for this state: it is a liquid.
    You may recall that our original derivation of the van der Waals equation was valid
    only for densities much lower than the liquid state. This means that we don’t really
    trust (5.1) on this solution. Nonetheless, it is interesting that the equation predicts
    the existence of liquids and our plan is to gratefully accept this gift and push ahead
    to explore what the van der Waals tells us about the liquid-gas transition. We will see
    that it captures many of the qualitative features of the phase transition.
    – 136 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    The last of the three solutions is the one on the right in the ﬁgure. This solution has
    v ≫b and small |dp/dv|. It is the gas state. Our goal is to understand what happens
    in between the liquid and gas state. We know that the naive, middle, solution given to
    us by the van der Waals equation is unstable. What replaces it?
    5.1.1 Phase Equilibrium
    Throughout our derivation of the van der Waals equation in Section 2.5, we assumed
    that the system was at a ﬁxed density. But the presence of two solutions — the liquid
    and gas state — allows us to consider more general conﬁgurations: part of the system
    could be a liquid and part could be a gas.
    How do we ﬁgure out if this indeed happens? Just because both liquid and gas states
    can exist, doesn’t mean that they can cohabit. It might be that one is preferred over
    the other. We already saw some conditions that must be satisﬁed in order for two
    systems to sit in equilibrium back in Section 1. Mechanical and thermal equilibrium
    are guaranteed if two systems have the same pressure and temperature respectively.
    But both of these are already guaranteed by construction for our two liquid and gas
    solutions: the two solutions sit on the same isotherm and at the same value of p. We’re
    left with only one further requirement that we must satisfy which arises because the
    two systems can exchange particles. This is the requirement of chemical equilibrium,
    µliquid = µgas
    (5.3)
    Because of the relationship (4.22) between the chemical potential and the Gibbs free
    energy, this is often expressed as
    gliquid = ggas
    (5.4)
    where g = G/N is the Gibbs free energy per particle.
    Notice that all the equilibrium conditions involve only intensive quantities: p, T and
    µ. This means that if we have a situation where liquid and gas are in equilibrium, then
    we can have any number Nliquid of atoms in the liquid state and any number Ngas in
    the gas state. But how can we make sure that chemical equilibrium (5.3) is satisﬁed?
    Maxwell Construction
    We want to solve µliquid = µgas. We will think of the chemical potential as a function of
    p and T: µ = µ(p, T). Importantly, we won’t assume that µ(p, T) is single valued since
    that would be assuming the result we’re trying to prove! Instead we will show that if
    we ﬁx T, the condition (5.3) can only be solved for a very particular value of pressure
    – 137 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    p. To see this, start in the liquid state at some ﬁxed value of p and T and travel along
    the isotherm. The inﬁnitesimal change in the chemical potential is
    dµ = @µ
    @p
    !!!!
    T
    dp
    However, we can get an expression for @µ/@p by recalling that arguments involving
    extensive and intensive variables tell us that the chemical potential is proportional to
    the Gibbs free energy: G(p, T, N) = µ(p, T)N (4.22). Looking back at the variation of
    the Gibbs free energy (4.21) then tells us that
    @G
    @p
    !!!!
    N,T
    = @µ
    @p
    !!!!
    T
    N = V
    (5.5)
    Integrating along the isotherm then tells us the chem-
    b
    p
    T<Tc
    v
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 36: Maxwell Construction")
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
    return


@app.cell
def _(mo):
    mo.md("""
    ical potential of any point on the curve,
    µ(p, T) = µliquid +
    Z p
    pliquid
    dp0 V (p0, T)
    N
    When we get to gas state at the same pressure p =
    pliquid that we started from, the condition for equi-
    librium is µ = µliquid. Which means that the integral
    has to vanish. Graphically this is very simple to de-
    scribe: the two shaded areas in the graph must have equal area. This condition, known
    as the Maxwell construction, tells us the pressure at which gas and liquid can co-exist.
    I should confess that there’s something slightly dodgy about the Maxwell construc-
    tion.
    We already argued that the part of the isotherm with dp/dv > 0 su↵ers an
    instability and is unphysical. But we needed to trek along that part of the curve to
    derive our result. There are more rigorous arguments that give the same answer.
    For each isotherm, we can determine the pressure at which the liquid and gas states
    are in equilibrium. The gives us the co-existence curve, shown by the dotted line in
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 37: Gibbs Free Energy (T > Tc)")
    _p = np.linspace(0.1, 2.0, 100)
    _g = np.log(_p)
    _df = pd.DataFrame({'p': _p, 'g': _g})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x='p:Q', y='g:Q', tooltip=['p', 'g']
    ).properties(width=700, height=400, title="Gibbs Free Energy vs Pressure (T > Tc)").interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    and pressure. But there is nothing that tells us how much gas there should be and how
    much liquid: atoms can happily move from the liquid state to the gas state. This means
    that while the density of gas and liquid is ﬁxed, the average density of the system is
    not. It can vary between the gas density and the liquid density simply by changing the
    amount of liquid. The upshot of this argument is that inside the co-existence curves,
    the isotherms simply become ﬂat lines, reﬂecting the fact that the density can take any
    value. This is shown in the graph on the right of Figure 37.
    – 138 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    p
    v
    c
    T=T
    p
    v
    c
    T=T
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 37: Gibbs Free Energy (T > Tc)")
    _p = np.linspace(0.1, 2.0, 100)
    _g = np.log(_p)
    _df = pd.DataFrame({'p': _p, 'g': _g})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x='p:Q', y='g:Q', tooltip=['p', 'g']
    ).properties(width=700, height=400, title="Gibbs Free Energy vs Pressure (T > Tc)").interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    of a harmonious mixture of vapour and liquid.
    To illustrate the physics of this situation, suppose that we sit
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 38: Gibbs Free Energy (T < Tc)")
    _p = np.linspace(0.1, 2.0, 500)
    _g = -0.5 * (_p - 1)**3 + 0.5 * _p
    _df = pd.DataFrame({'p': _p, 'g': _g})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x='p:Q', y='g:Q', tooltip=['p', 'g']
    ).properties(width=700, height=400, title="Gibbs Free Energy vs Pressure (T < Tc)").interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    at some ﬁxed density ⇢= 1/v and cool the system down from a
    high temperature to T < Tc at a point inside the co-existence curve
    so that we’re now sitting on one of the ﬂat lines. Here, the system
    is neither entirely liquid, nor entirely gas. Instead it will split into
    gas, with density 1/vgas, and liquid, with density 1/vliquid so that the
    average density remains 1/v. The system undergoes phase separation.
    The minimum energy conﬁguration will typically be a single phase of
    liquid and one of gas because the interface between the two costs energy. (We will derive
    an expression for this energy in Section 5.5.) The end result is shown on the right. In
    the presence of gravity, the higher density liquid will indeed sink to the bottom.
    Meta-Stable States
    We’ve understood what replaces the unstable region
    v
    p
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 39: Gibbs Free Energy - Lowest Branch")
    _p = np.linspace(0.1, 2.0, 100)
    _g = np.minimum(_p*0.5, _p*1.5 - 1)
    _df = pd.DataFrame({'p': _p, 'g': _g})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x='p:Q', y='g:Q', tooltip=['p', 'g']
    ).properties(width=700, height=400, title="Gibbs Free Energy (Lowest branch preferred)").interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    of the van der Waals phase diagram. But we seem to have
    removed more states than anticipated: parts of the Van
    der Waals isotherm that had dp/dv < 0 are contained in
    the co-existence region and replaced by the ﬂat pressure
    lines. This is the region of the p-V phase diagram that
    is contained between the two dotted lines in the ﬁgure to
    the right. The outer dotted line is the co-existence curve.
    The inner dotted curve is constructed to pass through the
    stationary points of the van der Waals isotherms. It is
    called the spinodal curve.
    – 139 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    The van der Waals states which lie between the spinodal curve and the co-existence
    curve are good states. But they are meta-stable. One can show that their Gibbs free
    energy is higher than that of the liquid-gas equilibrium at the same p and T. However,
    if we compress the gas very slowly we can coax the system into this state. It is known
    as a supercooled vapour. It is delicate. Any small disturbance will cause some amount
    of the gas to condense into the liquid. Similarly, expanding a liquid beyond the co-
    existence curve results in a meta-stable, superheated liquid.
    5.1.2 The Clausius-Clapeyron Equation
    We can also choose to plot the liquid-gas phase dia-
    Tc
    liquid
    gas
    critical point
    p
    T
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 40: Phase Diagram (p vs T)")
    _T = np.linspace(0.5, 1.0, 100)
    _p = np.exp(-1/_T)
    _df = pd.DataFrame({'T': _T, 'p': _p})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x='T:Q', y='p:Q', tooltip=['T', 'p']
    ).properties(width=700, height=400, title="Liquid-Gas Coexistence Curve").interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    gram on the p −T plane. Here the co-existence region is
    squeezed into a line: if we’re sitting in the gas phase and
    increase the pressure just a little bit at ﬁxed T < Tc then
    we jump immediately to the liquid phase. This appears
    as a discontinuity in the volume. Such discontinuities are
    the sign of a phase transition. The end result is sketched
    in the ﬁgure to the right; the thick solid line denotes the
    presence of a phase transition.
    Either side of the line, all particles are either in the gas or liquid phase. We know
    from (5.4) that the Gibbs free energies (per particle) of these two states are equal,
    gliquid = ggas
    So G is continuous as we move across the line of phase transitions. Suppose that we
    sit on the line itself and move along it. How does g change? We can easily compute
    this from (4.21),
    dGliquid = −SliquiddT + Vliquiddp
    )
    dgliquid = −sliquiddT + vliquiddp
    where, just as g = G/N an v = V/N, the entropy density is s = S/N. Equating this
    with the free energy in the gaseous phase gives
    dgliquid = −sliquiddT + vliquiddp = dggas = −sgasdT + vgasdp
    This can be rearranged to gives us a nice expression for the slope of the line of phase
    transitions in the p −T plane. It is
    dp
    dT = sgas −sliquid
    vgas −vliquid
    – 140 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    We usually deﬁne the speciﬁc latent heat
    L = T(sgas −sliquid)
    This is the energy released per particle as we pass through the phase transition. We
    see that the slope of the line in the p−T plane is determined by the ratio of latent heat
    released in the phase transition and the discontinuity in volume. The result is known
    as the Clausius-Clapeyron equation,
    dp
    dT =
    L
    T(vgas −vliquid)
    (5.6)
    There is a classiﬁcation of phase transitions, due originally to Ehrenfest. When the
    nth derivative of a thermodynamic potential (either F or G usually) is discontinuous,
    we say we have an nth order phase transition. In practice, we nearly always deal with
    ﬁrst, second and (very rarely) third order transitions. The liquid-gas transition releases
    latent heat, which means that S = −@F/@T is discontinuous. Alternatively, we can
    say that V = @G/@p is discontinuous. Either way, it is a ﬁrst order phase transition.
    The Clausius-Clapeyron equation (5.6) applies to any ﬁrst order transition.
    As we approach T ! Tc, the discontinuity dimin-
    p
    T
    Tc
    solid
    gas
    liquid
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 41: Critical Point Phase Diagram (p vs T)")
    _T = np.linspace(0.5, 1.0, 100)
    _p = np.exp(-1/_T)
    _df = pd.DataFrame({'T': _T, 'p': _p})
    _line = alt.Chart(_df).mark_line(size=3).encode(x=alt.X('T:Q', title="Temperature"), y=alt.Y('p:Q', title="Pressure"))
    _point = alt.Chart(pd.DataFrame({'T': [1.0], 'p': [np.exp(-1.0)]})).mark_point(color='red', size=100, filled=True).encode(x='T:Q', y='p:Q')
    _chart = (_line + _point).properties(width=700, height=400, title="Phase diagram with critical point").interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    ishes and Sliquid ! Sgas. At the critical point T = Tc we
    have a second order phase transition. Above the critical
    point, there is no sharp distinction between the gas phase
    and liquid phase.
    For most simple materials, the phase diagram above is
    part of a larger phase diagram which includes solids at
    smaller temperatures or higher pressures. A generic ver-
    sion of such a phase diagram is shown to the right. The
    van der Waals equation is missing the physics of solidiﬁca-
    tion and includes only the liquid-gas line.
    An Approximate Solution to the Clausius-Clapeyron Equation
    We can solve the Clausius-Clapeyron solution if we make the following assumptions:
    • The latent heat L is constant.
    • vgas ≫vliquid, so vgas −vliquid ⇡vgas. For water, this is an error of less than 0.1%.
    • Although we derived the phase transition using the van der Waals equation, now
    we’ve got equation (5.6) we’ll pretend the gas obeys the ideal gas law pv = kBT.
    – 141 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    With these assumptions, it is simple to solve (5.6). It reduces to
    dp
    dT =
    Lp
    kBT 2
    )
    p = p0e−L/kBT
    5.1.3 The Critical Point
    Let’s now return to discuss some aspects of life at the critical point. We previously
    worked out the critical temperature (5.2) by looking for solutions to simultaneous equa-
    tions @p/@v = @2p/@v2 = 0. There’s a slightly more elegant way to ﬁnd the critical
    point which also quickly gives us pc and vc as well. We rearrange the van der Waals
    equation (5.1) to get a cubic,
    pv3 −(pb + kBT)v2 + av −ab = 0
    For T < Tc, this equation has three real roots. For T > Tc there is just one. Precisely at
    T = Tc, the three roots must therefore coincide (before two move o↵onto the complex
    plane). At the critical point, this curve can be written as
    pc(v −vc)3 = 0
    Comparing the coeﬃcients tells us the values at the critical point,
    kBTc = 8a
    27b
    ,
    vc = 3b
    ,
    pc =
    a
    27b2
    (5.7)
    The Law of Corresponding States
    We can invert the relations (5.7) to express the parameters a and b in terms of the
    critical values, which we then substitute back into the van der Waals equation. To this
    end, we deﬁne the reduced variables,
    ¯T = T
    Tc
    ,
    ¯v = v
    vc
    ¯p = p
    pc
    The advantage of working with ¯T, ¯v and ¯p is that it allows us to write the van der
    Waals equation (5.1) in a form that is universal to all gases, usually referred to as the
    law of corresponding states
    ¯p = 8
    3
    ¯T
    ¯v −1/3 −3
    ¯v2
    Moreover, because the three variables Tc, pc and vc at the critical point are expressed
    in terms of just two variables, a and b (5.7), we can construct a combination of them
    – 142 –
    """)
    return


@app.cell
def _(alt, mo, np, pd):
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

    mo.vstack([
        _chart,
        mo.md("### Figure 42: Phase Diagram (p vs v)")
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    and CH4.
    which is independent of a and b and therefore supposedly the same for all gases. This
    is the universal compressibility ratio,
    pcvc
    kBTc
    = 3
    8 = 0.375
    (5.8)
    Comparing to real gases, this number is a little high. Values range from around
    0.28 to 0.3. We shouldn’t be too discouraged by this; after all, we knew from the
    beginning that the van der Waals equation is unlikely to be accurate in the liquid
    regime. Moreover, the fact that gases have a critical point (deﬁned by three variables
    Tc, pc and vc) guarantees that a similar relationship would hold for any equation of
    state which includes just two parameters (such as a and b) but would most likely fail
    to hold for equations of state that included more than two parameters.
    Dubious as its theoretical foundation is, the law of corresponding states is the ﬁrst
    suggestion that something remarkable happens if we describe a gas in terms of its
    reduced variables. More importantly, there is striking experimental evidence to back
    this up! Figure 42 shows the Guggenheim plot, constructed in 1945. The co-existence
    curve for 8 di↵erent gases in plotted in reduced variables: ¯T along the vertical axis;
    ¯⇢= 1/¯v along the horizontal. The gases vary in complexity from the simple monatomic
    gas Ne to the molecule CH4. As you can see, the co-existence curve for all gases is
    essentially the same, with the chemical make-up largely forgotten. There is clearly
    something interesting going on. How to understand it?
    Critical Exponents
    We will focus attention on physics close to the critical point. It is not immediately
    – 143 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    obvious what are the right questions to ask. It turns out that the questions which
    have the most interesting answers are concerned with how various quantities change
    as we approach the critical point. There are lots of ways to ask questions of this type
    since there are many quantities of interest and, for each of them, we could approach
    the critical point from di↵erent directions. Here we’ll look at the behaviour of three
    quantities to get a feel for what happens.
    First, we can ask what happens to the di↵erence in (inverse) densities vgas −vliquid as
    we approach the critical point along the co-existence curve. For T < Tc, or equivalently
    ¯T < 1, the reduced van der Waals equation (5.8) has two stable solutions,
    ¯p =
    8 ¯T
    3¯vliquid −1 −
    3
    ¯v2
    liquid
    =
    8 ¯T
    3¯vgas −1 −
    3
    ¯v2
    gas
    If we solve this for ¯T, we have
    ¯T = (3¯vliquid −1)(3¯vgas −1)(¯vliquid + ¯vgas)
    8¯v2
    gas¯v2
    liquid
    Notice that as we approach the critical point, ¯vgas, ¯vliquid ! 1 and the equation above
    tells us that ¯T ! 1 as expected. We can see exactly how we approach ¯T = 1 by
    expanding the right-hand side for small ✏⌘¯vgas −¯vliquid. To do this quickly, it’s best
    to notice that the equation is symmetric in ¯vgas and ¯vliquid, so close to the critical point
    we can write ¯vgas = 1 + ✏/2 and ¯vliquid = 1 −✏/2. Substituting this into the equation
    above and keeping just the leading order term, we ﬁnd
    ¯T ⇡1 −1
    16(¯vgas −¯vliquid)2
    Or, re-arranging, as we approach Tc along the co-existence curve,
    vgas −vliquid ⇠(Tc −T)1/2
    (5.9)
    This is the answer to our ﬁrst question.
    Our second variant of the question is: how does the volume change with pressure
    as we move along the critical isotherm. It turns out that we can answer this question
    without doing any work. Notice that at T = Tc, there is a unique pressure for a given
    volume p(v, Tc). But we know that @p/@v = @2p/@v2 = 0 at the critical point. So a
    Taylor expansion around the critical point must start with the cubic term,
    p −pc ⇠(v −vc)3
    (5.10)
    This is the answer to our second question.
    – 144 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Our third and ﬁnal variant of the question concerns the compressibility, deﬁned as
    = −1
    v
    @v
    @p
    !!!!
    T
    (5.11)
    We want to understand how changes as we approach T ! Tc from above. In fact, we
    met the compressibility before: it was the feature that ﬁrst made us nervous about the
    van der Waals equation since is negative in the unstable region. We already know
    that at the critical point @p/@v|Tc = 0. So expanding for temperatures close to Tc, we
    expect
    @p
    @v
    !!!!
    T;v=vc
    = −a(T −Tc) + . . .
    This tells us that the compressibility should diverge at the critical point, scaling as
    ⇠(T −Tc)−1
    (5.12)
    We now have three answers to three questions: (5.9), (5.10) and (5.12).
    Are they
    right?! By which I mean: do they agree with experiment? Remember that we’re not
    sure that we can trust the van der Waals equation at the critical point so we should be
    nervous. However, there is also reason for some conﬁdence. Notice, in particular, that
    in order to compute (5.10) and (5.12), we didn’t actually need any details of the van
    der Waals equation. We simply needed to assume the existence of the critical point
    and an analytic Taylor expansion of various quantities in the neighbourhood. Given
    that the answers follow from such general grounds, one may hope that they provide
    the correct answers for a gas in the neighbourhood of the critical point even though we
    know that the approximations that went into the van der Waals equation aren’t valid
    there. Fortunately, that isn’t the case: physics is much more interesting than that!
    The experimental results for a gas in the neighbourhood of the critical point do share
    one feature in common with the discussion above: they are completely independent of
    the atomic make-up of the gas. However, the scaling that we computed using the van
    der Waals equation is not fully accurate. The correct results are as follows. As we
    approach the critical point along the co-existence curve, the densities scale as
    vgas −vliquid ⇠(Tc −T)β
    with
    β ⇡0.32
    (Note that the exponent β has nothing to do with inverse temperature. We’re just near
    the end of the course and running out of letters and β is the canonical name for this
    exponent.) As we approach along an isotherm,
    p −pc ⇠(v −vc)δ
    with
    δ ⇡4.8
    – 145 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Finally, as we approach Tc from above, the compressibility scales as
    ⇠(T −Tc)−γ
    with
    γ ⇡1.2
    The quantities β, γ and δ are examples of critical exponents. We will see more of them
    shortly. The van der Waals equation provides only a crude ﬁrst approximation to the
    critical exponents.
    Fluctuations
    We see that the van der Waals equation didn’t do too badly in capturing the dynamics
    of an interacting gas.
    It gets the qualitative behaviour right, but fails on precise
    quantitative tests. So what went wrong? We mentioned during the derivation of the
    van der Waals equation that we made certain approximations that are valid only at
    low density. So perhaps it is not surprising that it fails to get the numbers right near
    the critical point v = 3b. But there’s actually a deeper reason that the van der Waals
    equation fails: ﬂuctuations.
    This is simplest to see in the grand canonical ensemble. Recall that back in Section
    1 that we argued that ∆N/N ⇠1/
    p
    N, which allowed us to happily work in the
    grand canonical ensemble even when we actually had ﬁxed particle number. In the
    context of the liquid-gas transition, ﬂuctuating particle number is the same thing as
    ﬂuctuating density ⇢= N/V . Let’s revisit the calculation of ∆N near the critical
    point. Using (1.45) and (1.48), the grand canonical partition function can be written
    as log Z = βV p(T, µ), so the average particle number (1.42) is
    hNi = V @p
    @µ
    !!!!
    T,V
    We already have an expression for the variance in the particle number in (1.43),
    ∆N 2 = 1
    β
    @hNi
    @µ
    !!!!
    T,V
    Dividing these two expressions, we have
    ∆N 2
    N
    =
    1
    V β
    @hNi
    @µ
    !!!!
    T,V
    @µ
    @p
    !!!!
    T,V
    =
    1
    V β
    @hNi
    @p
    !!!!
    T,V
    But we can re-write this expression using the general relationship between partial
    derivatives @x/@y|z @y/@z|x @z/@x|y = −1. We then have
    ∆N 2
    N
    = −1
    β
    @hNi
    @V
    !!!!
    p,T
    1
    V
    @V
    @p
    !!!!
    N,T
    – 146 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    This ﬁnal expression relates the ﬂuctuations in the particle number to the compress-
    ibility (5.11). But the compressibility is diverging at the critical point and this means
    that there are large ﬂuctuations in the density of the ﬂuid at this point. The result is
    that any simple equation of state, like the van der Waals equation, which works only
    with the average volume, pressure and density will miss this key aspect of the physics.
    Understanding how to correctly account for these ﬂuctuations is the subject of critical
    phenomena. It has close links with the renormalization group and conformal ﬁeld theory
    which also arise in particle physics and string theory. You will meet some of these ideas
    in next year’s Statistical Field Theory course. Here we will turn to a di↵erent phase
    transition which will allow us to highlight some of the key ideas.
    5.2 The Ising Model
    The Ising model is one of the touchstones of modern physics; a simple system that
    exhibits non-trivial and interesting behaviour.
    The Ising model consists of N sites in a d-dimensional lattice. On each lattice site
    lives a quantum spin that can sit in one of two states: spin up or spin down. We’ll call
    the eigenvalue of the spin on the ith lattice site si. If the spin is up, si = +1; if the spin
    is down, si = −1.
    The spins sit in a magnetic ﬁeld that endows an energy advantage to those which
    point up,
    EB = −B
    N
    X
    i=1
    si
    (A comment on notation: B should be properly denoted H. We’re sticking with B to
    avoid confusion with the Hamiltonian. There is also a factor of the magnetic moment
    which has been absorbed into the deﬁnition of B.) The lattice system with energy EB
    is equivalent to the two-state system that we ﬁrst met when learning the techniques of
    statistical mechanics back in Section 1.2.3. However, the Ising model contains an addi-
    tional complication that makes the system much more interesting: this is an interaction
    between neighbouring spins. The full energy of the system is therefore,
    E = −J
    X
    hiji
    sisj −B
    X
    i
    si
    (5.13)
    The notation hiji means that we sum over all “nearest neighbour” pairs in the lattice.
    The number of such pairs depends both on the dimension d and the type of lattice.
    – 147 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    We’ll denote the number of nearest neighbours as q. For example, in d = 1 a lattice
    has q = 2; in d = 2, a square lattice has q = 4. A square lattice in d dimensions has
    q = 2d.
    If J > 0, neighbouring spins prefer to be aligned ("\" or ##).
    In the context of
    magnetism, such a system is called a ferromagnet. If J < 0, the spins want to anti-
    align ("#). This is an anti-ferromagnet. In the following, we’ll choose J > 0 although
    for the level of discussion needed for this course, the di↵erences are minor.
    We work in the canonical ensemble and introduce the partition function
    Z =
    X
    {si}
    e−βE[si]
    (5.14)
    While the e↵ect of both J > 0 and B 6= 0 is to make it energetically preferable for the
    spins to align, the e↵ect of temperature will be to randomize the spins, with entropy
    winning over energy. Our interest is in the average spin, or average magnetization
    m = 1
    N
    X
    i
    hsii =
    1
    Nβ
    @log Z
    @B
    (5.15)
    The Ising Model as a Lattice Gas
    Before we develop techniques to compute the partition function (5.14), it’s worth point-
    ing out that we can drape slightly di↵erent words around the mathematics of the Ising
    model. It need not be interpreted as a system of spins; it can also be thought of as a
    lattice description of a gas.
    To see this, consider the same d-dimensional lattice as before, but now with particles
    hopping between lattice sites. These particles have hard cores, so no more than one
    can sit on a single lattice site. We introduce the variable ni 2 {0, 1} to specify whether
    a given lattice site, labelled by i, is empty (ni = 0) or ﬁlled (ni = 1). We can also
    introduce an attractive force between atoms by o↵ering them an energetic reward if
    they sit on neighbouring sites. The Hamiltonian of such a lattice gas is given by
    E = −4J
    X
    hiji
    ninj −µ
    X
    i
    ni
    where µ is the chemical potential which determines the overall particle number. But this
    Hamiltonian is trivially the same as the Ising model (5.13) if we make the identiﬁcation
    si = 2ni −1 2 {−1, 1}
    The chemical potential µ in the lattice gas plays the role of magnetic ﬁeld in the spin
    system while the magnetization of the system (5.15) measures the average density of
    particles away from half-ﬁlling.
    – 148 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    5.2.1 Mean Field Theory
    For general lattices, in arbitrary dimension d, the sum (5.14) cannot be performed. An
    exact solution exists in d = 1 and, when B = 0, in d = 2. (The d = 2 solution is
    originally due to Onsager and is famously complicated! Simpler solutions using more
    modern techniques have since been discovered.)
    Here we’ll develop an approximate method to evaluate Z known as mean ﬁeld theory.
    We write the interactions between neighbouring spins in term of their deviation from
    the average spin m,
    sisj = [(si −m) + m][(sj −m) + m]
    = (si −m)(sj −m) + m(sj −m) + m(si −m) + m2
    The mean ﬁeld approximation means that we assume that the ﬂuctuations of spins away
    from the average are small which allows us to neglect the ﬁrst term above. Notice that
    this isn’t the statement that the variance of an individual spin is small; that can never
    be true because si takes values +1 or −1 so hs2
    i i = 1 and the variance h(si −m)2i is
    always large. Instead, the mean ﬁeld approximation is a statement about ﬂuctuations
    between spins on neighbouring sites, so the ﬁrst term above can be neglected when
    summing over P
    hiji. We can then write the energy (5.13) as
    Emf = −J
    X
    hiji
    [m(si + sj) −m2] −B
    X
    i
    si
    = 1
    2JNqm2 −(Jqm + B)
    X
    i
    si
    (5.16)
    where the factor of Nq/2 in the ﬁrst term is simply the number of nearest neighbour
    pairs P
    hiji. The factor of 1/2 is there because P
    hiji is a sum over pairs rather than a
    sum of individual sites. (If you’re worried about this formula, you should check it for
    a linear lattice in d = 1 and a simple square lattice in d = 2 dimensions.) A similar
    factor in the second term cancelled the factor of 2 due to (si + sj).
    We see that the mean ﬁeld approximation has removed the interactions. The Ising
    model reduces to the two state system that we saw way back in Section 1. The result
    of the interactions is that a given spin feels the average e↵ect of its neighbour’s spins
    through a contribution to the e↵ective magnetic ﬁeld,
    Be↵= B + Jqm
    Once we’ve taken into account this extra contribution to Be↵, each spin acts indepen-
    – 149 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    tanh
    m
    m0
    −m
    tanh
    m
    0
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 43: 1D Ising Model Diagram")
    _spins = [1, -1, 1, 1, -1, -1, 1, -1]
    _positions = np.arange(len(_spins))
    _df = pd.DataFrame({'x': _positions, 'spin': _spins, 'label': ["↑" if s > 0 else "↓" for s in _spins]})
    _chart = alt.Chart(_df).mark_text(size=40).encode(
        x=alt.X('x:O', axis=alt.Axis(labels=False, ticks=False), title='Chain Position'),
        text='label:N', color=alt.condition(alt.datum.spin > 0, alt.value('blue'), alt.value('red'))
    ).properties(width=700, height=200, title="1D Ising Model Spin Chain").interactive()
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 44: 2D Ising Model Heatmap Placeholder")
    _x, _y = np.meshgrid(np.arange(10), np.arange(10))
    _spins = np.random.choice([-1, 1], size=(10, 10))
    _df = pd.DataFrame({'x': _x.flatten(), 'y': _y.flatten(), 'spin': _spins.flatten()})
    _chart = alt.Chart(_df).mark_rect().encode(
        x='x:O', y='y:O', color=alt.Color('spin:Q', scale=alt.Scale(range=['red', 'blue'])),
        tooltip=['x', 'y', 'spin']
    ).properties(width=400, height=400, title="2D Ising Model").interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    dently and it is easy to write the partition function. It is
    Z = e−1
    2 βJNqm2 %
    e−βBe↵+ eβBe↵&N
    = e−1
    2 βJNqm22N coshN βBe↵
    (5.17)
    However, we’re not quite done. Our result for the partition function Z depends on Be↵
    which depends on m which we don’t yet know. However, we can use our expression for
    Z to self-consistently determine the magnetization (5.15). We ﬁnd,
    m = tanh(βB + βJqm)
    (5.18)
    We can now solve this equation to ﬁnd the magnetization for various values of T and B:
    m = m(T, B). It is simple to see the nature of the solutions using graphical methods.
    B=0
    Let’s ﬁrst consider the situation with vanishing magnetic ﬁeld, B = 0. The ﬁgures
    above show the graph linear in m compared with the tanh function. Since tanh x ⇡
    x−1
    3x3+. . ., the slope of the graph near the origin is given by βJq. This then determines
    the nature of the solution.
    • The ﬁrst graph depicts the situation for βJq < 1. The only solution is m = 0.
    This means that at high temperatures kBT > Jq, there is no average magne-
    tization of the system. The entropy associated to the random temperature ﬂu-
    cutations wins over the energetically preferred ordered state in which the spins
    align.
    • The second graph depicts the situation for βJq > 1. Now there are three solu-
    tions: m = ±m0 and m = 0. It will turn out that the middle solution, m = 0, is
    unstable. (This solution is entirely analogous to the unstable solution of the van
    der Waals equation. We will see this below when we compute the free energy.)
    – 150 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    For the other two possible solutions, m = ±m0, the magnetization is non-zero.
    Here we see the e↵ects of the interactions begin to win over temperature. Notice
    that in the limit of vanishing temperature, β ! 1, m0 ! 1. This means that all
    the spins are pointing in the same direction (either up or down) as expected.
    • The critical temperature separating these two cases is
    kBTc = Jq
    (5.19)
    The results described above are perhaps rather surprising. Based on the intuition that
    things in physics always happen smoothly, one might have thought that the magneti-
    zation would drop slowly to zero as T ! 1. But that doesn’t happen. Instead the
    magnetization turns o↵abruptly at some ﬁnite value of the temperature T = Tc, with
    no magnetization at all for higher temperatures. This is the characteristic behaviour
    of a phase transition.
    B 6= 0
    For B 6= 0, we can solve the consistency equation (5.18) in a similar fashion. There are
    a couple of key di↵erences to the B 6= 0 case. Firstly, there is now no phase transition
    at ﬁxed B as we vary temperature T. Instead, for very large temperatures kBT ≫Jq,
    the magnetization goes smoothly to zero as
    m !
    B
    kBT
    as T ! 1
    At low temperatures, the magnetization again asymptotes to the state m ! ±1 which
    minimizes the energy. Except this time, there is no ambiguity as to whether the system
    chooses m = +1 or m = −1. This is entirely determined by the sign of the magnetic
    ﬁeld B.
    In fact, the low temperature behaviour requires slightly more explanation. For small
    values of B and T, there are again three solutions to (5.18). This follows simply from
    continuity: there are three solutions for T < Tc and B = 0 shown in Figure 44 and these
    must survive in some neighbourhood of B = 0. One of these solutions is again unstable.
    However, of the remaining two only one is now stable: that with sign(m) = sign(B).
    The other is meta-stable. We will see why this is the case shortly when we come to
    discuss the free energy.
    – 151 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    T
    m
    +1
    −1
    m
    T
    B<0
    B>0
    +1
    −1
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 45: Mean Field Free Energy (T > Tc)")
    _m = np.linspace(-2, 2, 100)
    _f = 0.5 * _m**2 + 0.1 * _m**4
    _df = pd.DataFrame({'m': _m, 'f': _f})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x=alt.X('m:Q', title='Magnetization m'), y=alt.Y('f:Q', title='Free Energy F')
    ).properties(width=700, height=400, title="Free Energy vs Magnetization (T > Tc)").interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    and the phase transtion
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 46: Mean Field Free Energy (T < Tc)")
    _m = np.linspace(-2, 2, 100)
    _f = -0.5 * _m**2 + 0.1 * _m**4
    _df = pd.DataFrame({'m': _m, 'f': _f})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x=alt.X('m:Q', title='Magnetization m'), y=alt.Y('f:Q', title='Free Energy F')
    ).properties(width=700, height=400, title="Free Energy vs Magnetization (T < Tc)").interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    The net result of our discussion is depicted in the ﬁgures above.
    When B = 0
    there is a phase transition at T = Tc. For T < Tc, the system can sit in one of two
    magnetized states with m = ±m0. In contrast, for B 6= 0, there is no phase transition
    as we vary temperature and the system has at all times a preferred magnetization
    whose sign is determined by that of B. Notice however, we do have a phase transition
    if we ﬁx temperature at T < Tc and vary B from negative to positive.
    Then the
    magnetization jumps discontinuously from a negative value to a positive value. Since
    the magnetization is a ﬁrst derivative of the free energy (5.15), this is a ﬁrst order
    phase transition. In contrast, moving along the temperature axis at B = 0 results in a
    second order phase transition at T = Tc.
    5.2.2 Critical Exponents
    It is interesting to compare the phase transition of the Ising model with that of the
    liquid-gas phase transition. The two are sketched in Figure 47. In both cases, we have
    a ﬁrst order phase transition and a quantity jumps discontinuously at T < Tc. In the
    case of the liquid-gas, it is the density ⇢= 1/v that jumps as we vary pressure; in the
    case of the Ising model it is the magnetization m that jumps as we vary the magnetic
    ﬁeld. Moreover, in both cases the discontinuity disappears as we approach T = Tc.
    T
    p
    Tc
    Tc
    liquid
    gas
    critical point
    T
    B
    critical point
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 47: Mean Field Free Energy with Finite B")
    _m = np.linspace(-2, 2, 100)
    _f = -0.5 * _m**2 + 0.1 * _m**4 - 0.3 * _m
    _df = pd.DataFrame({'m': _m, 'f': _f})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x=alt.X('m:Q', title='Magnetization m'), y=alt.Y('f:Q', title='Free Energy F')
    ).properties(width=700, height=400, title="Free Energy vs Magnetization (T < Tc, B > 0)").interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    – 152 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    We can calculate critical exponents for the Ising model. To compare with our discus-
    sion for the liquid-gas critical point, we will compute three quantities. First, consider
    the magnetization at B = 0. We can ask how this magnetization decreases as we tend
    towards the critical point. Just below T = Tc, m is small and we can Taylor expand
    (5.18) to get
    m ⇡βJqm −1
    3(βJqm)3 + . . .
    The magnetization therefore scales as
    m0 ⇠±(Tc −T)1/2
    (5.20)
    This is to be compared with the analogous result (5.9) from the van der Waals equation.
    We see that the values of the exponents are the same in both cases. Notice that the
    derivative dm/dT becomes inﬁnite as we approach the critical point. In fact, we had
    already anticipated this when we drew the plot of the magnetization in Figure 45.
    Secondly, we can sit at T = Tc and ask how the magnetization changes as we approach
    B = 0.
    We can read this o↵from (5.18).
    At T = Tc we have βJq = 1 and the
    consistency condition becomes m = tanh(B/Jq + m). Expanding for small B gives
    m ⇡B
    Jq + m −1
    3
    ✓B
    Jq + m
    ◆3
    + . . . ⇡B
    Jq + m −1
    3m3 + O(B2)
    So we ﬁnd that the magnetization scales as
    m ⇠B1/3
    (5.21)
    Notice that this power of 1/3 is again familiar from the liquid-gas transition (5.10)
    where the van der Waals equation gave vgas −vliquid ⇠(p −pc)1/3.
    Finally, we can look at the magnetic susceptibility χ, deﬁned as
    χ = N @m
    @B
    !!!!
    T
    This is analogous to the compressibility of the gas. We will ask how χ changes as we
    approach T ! Tc from above at B = 0. We di↵erentiate (5.18) with respect to B to
    get
    χ =
    Nβ
    cosh2 βJqm
    ✓
    1 + Jq
    N χ
    ◆
    – 153 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    We now evaluate this at B = 0. Since we want to approach T ! Tc from above, we can
    also set m = 0 in the above expression. Evaluating this at B = 0 gives us the scaling
    χ =
    Nβ
    1 −Jqβ ⇠(T −Tc)−1
    (5.22)
    Once again, we see that same critical exponent that the van der Waals equation gave
    us for the gas (5.12).
    5.2.3 Validity of Mean Field Theory
    The phase diagram and critical exponents above were all derived using the mean ﬁeld
    approximation. But this was an unjustiﬁed approximation. Just as for the van der
    Waals equation, we can ask the all-important question: are our results right?
    There is actually a version of the Ising model for which the mean ﬁeld theory is
    exact: it is the d = 1 dimensional lattice.
    This is unphysical (even for a string
    theorist). Roughly speaking, mean ﬁeld theory works for large d because each spin has
    a large number of neighbours and so indeed sees something close to the average spin.
    But what about dimensions of interest? Mean ﬁeld theory gets things most dramati-
    cally wrong in d = 1. In that case, no phase transition occurs. We will derive this result
    below where we brieﬂy describe the exact solution to the d = 1 Ising model. There is
    a general lesson here: in low dimensions, both thermal and quantum ﬂuctuations are
    more important and invariably stop systems forming ordered phases.
    In higher dimensions, d ≥2, the crude features of the phase diagram, including
    the existence of a phase transition, given by mean ﬁeld theory are essentially correct.
    In fact, the very existence of a phase transition is already worthy of comment. The
    deﬁning feature of a phase transition is behaviour that jumps discontinuously as we
    vary β or B. Mathematically, the functions must be non-analytic. Yet all properties of
    the theory can be extracted from the partition function Z which is a sum of smooth,
    analytic functions (5.14). How can we get a phase transition? The loophole is that Z
    is only necessarily analytic if the sum is ﬁnite. But there is no such guarantee when
    the number of lattice sites N ! 1. We reach a similar conclusion to that of Bose-
    Einstein condensation: phase transitions only strictly happen in the thermodynamic
    limit. There are no phase transitions in ﬁnite systems.
    What about the critical exponents that we computed in (5.20), (5.21) and (5.22)? It
    turns out that these are correct for the Ising model deﬁned in d ≥4. (We will brieﬂy
    sketch why this is true at the end of this Chapter.) But for d = 2 and d = 3, the
    critical exponents predicted by mean ﬁeld theory are only ﬁrst approximations to the
    true answers.
    – 154 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    For d = 2, the exact solution (which goes quite substantially past this course) gives
    the critical exponents to be,
    m0 ⇠(Tc −T)β
    with
    β = 1
    8
    m ⇠B1/δ
    with
    δ = 15
    χ ⇠(T −Tc)−γ
    with
    γ = 7
    4
    The biggest surprise is in d = 3 dimensions. Here the critical exponents are not known
    exactly. However, there has been a great deal of numerical work to determine them.
    They are given by
    β ⇡0.32
    ,
    δ ⇡4.8
    ,
    γ ⇡1.2
    But these are exactly the same critical exponents that are seen in the liquid-gas phase
    transition. That’s remarkable! We saw above that the mean ﬁeld approach to the Ising
    model gave the same critical exponents as the van der Waals equation. But they are
    both wrong. And they are both wrong in the same, complicated, way! Why on earth
    would a system of spins on a lattice have anything to do with the phase transition
    between a liquid and gas? It is as if all memory of the microscopic physics — the type
    of particles, the nature of the interactions — has been lost at the critical point. And
    that’s exactly what happens.
    What we’re seeing here is evidence for universality. There is a single theory which
    describes the physics at the critical point of the liquid gas transition, the 3d Ising model
    and many other systems. This is a theoretical physicist’s dream! We spend a great
    deal of time trying to throw away the messy details of a system to focus on the elegant
    essentials. But, at a critical point, Nature does this for us! Although critical points
    in two dimensions are well understood, there is still much that we don’t know about
    critical points in three dimensions. This, however, is a story that will have to wait for
    another day.
    5.3 Some Exact Results for the Ising Model
    This subsection is something of a diversion from our main interest. In later subsec-
    tions, we will develop the idea of mean ﬁeld theory. But ﬁrst we pause to describe
    some exact results for the Ising model using techniques that do not rely on the mean
    ﬁeld approximation. Many of the results that we derive have broader implications for
    systems beyond the Ising model.
    – 155 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    As we mentioned above, there is an exact solution for the Ising model in d = 1
    dimension and, when B = 0, in d = 2 dimensions. Here we will describe the d = 1
    solution but not the full d = 2 solution. We will, however, derive a number of results
    for the d = 2 Ising model which, while falling short of the full solution, nonetheless
    provide important insights into the physics.
    5.3.1 The Ising Model in d = 1 Dimensions
    We start with the Ising chain, the Ising model on a one dimensional line. Here we will
    see that the mean ﬁeld approximation fails miserably, giving qualitatively incorrect
    results: the exact results show that there are no phase transitions in the Ising chain.
    The energy of the system (5.13) can be trivially rewritten as
    E = −J
    N
    X
    i=1
    sisi+1 −B
    2
    N
    X
    i=1
    (si + si+1)
    We will impose periodic boundary conditions, so the spins live on a circular lattice with
    sN+1 ⌘s1. The partition function is then
    Z =
    X
    s1=±1
    . . .
    X
    sN=±1
    N
    Y
    i=1
    exp
    ✓
    βJsisi+1 + βB
    2 (si + si+1)
    ◆
    (5.23)
    The crucial observation that allows us to solve the problem is that this partition function
    can be written as a product of matrices. We adopt notation from quantum mechanics
    and deﬁne the 2 ⇥2 matrix,
    hsi|T|si+1i ⌘exp
    ✓
    βJsisi+1 + βB
    2 (si + si+1)
    ◆
    (5.24)
    The row of the matrix is speciﬁed by the value of si = ±1 and the column by si+1 = ±1.
    T is known as the transfer matrix and, in more conventional notation, is given by
    T =

    eβJ+βB
    e−βJ
    e−βJ
    eβJ−βB
    !
    The sums over the spins P
    si and product over lattice sites Q
    i in (5.23) simply tell us
    to multiply the matrices deﬁned in (5.24) and the partition function becomes
    Z = Tr (hs1|T|s2ihs2|T|s3i . . . hsN|T|s1i) = Tr T N
    (5.25)
    – 156 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    where the trace arises because we have imposed periodic boundary conditions. To com-
    plete the story, we need only compute the eigenvalues of T to determine the partition
    function. A quick calculation shows that the two eigenvalues of T are
    λ± = eβJ cosh βB ±
    q
    e2βJ cosh2 βB −2 sinh 2βJ
    (5.26)
    where, clearly, λ−< λ+. The partition function is then
    Z = λN
    + + λN
    −= λN
    +
    ✓
    1 + λN
    −
    λN
    +
    ◆
    ⇡λN
    +
    (5.27)
    where, in the last step, we’ve used the simple fact that if λ+ is the largest eigenvalue
    then λN
    −/λN
    + ⇡0 for very large N.
    The partition function Z contains many quantities of interest. In particular, we can
    use it to compute the magnetisation as a function of temperature when B = 0. This,
    recall, is the quantity which is predicted to undergo a phase transition in the mean
    ﬁeld approximation, going abruptly to zero at some critical temperature. In the d = 1
    Ising model, the magnetisation is given by
    m =
    1
    Nβ
    @ log Z
    @B
    !!!!
    B=0
    =
    1
    λ+β
    @λ+
    @B
    !!!!
    B=0
    = 0
    We see that the true physics for d = 1 is very di↵erent than that suggested by the
    mean ﬁeld approximation. When B = 0, there is no magnetisation! While the J term
    in the energy encourages the spins to align, this is completely overwhelmed by thermal
    ﬂuctuations for any value of the temperature.
    There is a general lesson in this calculation: thermal ﬂuctuations always win in one
    dimensional systems. They never exhibit ordered phases and, for this reason, never
    exhibit phase transitions. The mean ﬁeld approximation is bad in one dimension.
    5.3.2 2d Ising Model: Low Temperatures and Peierls Droplets
    Let’s now turn to the Ising model in d = 2 dimensions. We’ll work on a square lattice
    and set B = 0. Rather than trying to solve the model exactly, we’ll have more modest
    goals. We will compute the partition function in two di↵erent limits: high temperature
    and low temperature. We start here with the low temperature expansion.
    The partition function is given by the sum over all states, weighted by e−βE. At low
    temperatures, this is always dominated by the lowest lying states. For the Ising model,
    – 157 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    we have
    Z =
    X
    {si}
    exp
    0
    @βJ
    X
    hiji
    sisj
    1
    A
    The low temperature limit is βJ ! 1, where the partition function can be approxi-
    mated by the sum over the ﬁrst few lowest energy states. All we need to do is list these
    states.
    The ground states are easy. There are two of them: spins all up or spins all down.
    For example, the ground state with spins all up looks like
    Each of these ground states has energy E = E0 = −2NJ.
    The ﬁrst excited states arise by ﬂipping a single spin. Each spin has q = 4 nearest
    neighbours – denoted by red lines in the example below – each of which leads to an
    energy cost of 2J. The energy of each ﬁrst excited state is therefore E1 = E0 + 8J.
    There are, of course, N di↵erent spins that we we can ﬂip and, correspondingly, the
    ﬁrst energy level has a degeneracy of N.
    To proceed, we introduce a diagrammatic method to list the di↵erent states. We
    draw only the “broken” bonds which connect two spins with opposite orientation and,
    as in the diagram above, denote these by red lines. We further draw the ﬂipped spins
    as red dots, the unﬂipped spins as blue dots. The energy of the state is determined
    simply by the number of red lines in the diagram. Pictorially, we write the ﬁrst excited
    state as
    E1 = E0 + 8J
    Degeneracy = N
    – 158 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    The next lowest state has six broken bonds. It takes the form
    E2 = E0 + 12J
    Degeneracy = 2N
    where the extra factor of 2 in the degeneracy comes from the two possible orientations
    (vertical and horizontal) of the graph.
    Things are more interesting for the states which sit at the third excited level. These
    have 8 broken bonds. The simplest conﬁguration consists of two, disconnected, ﬂipped
    spins
    E3 = E0 + 16J
    Degeneracy = 1
    2N(N −5)
    (5.28)
    The factor of N in the degeneracy comes from placing the ﬁrst graph; the factor of
    N −5 arises because the ﬂipped spin in the second graph can sit anywhere apart from
    on the ﬁve vertices used in the ﬁrst graph. Finally, the factor of 1/2 arises from the
    interchange of the two graphs.
    There are also three further graphs with the same energy E3. These are
    E3 = E0 + 16J
    Degeneracy = N
    and
    E3 = E0 + 16J
    Degeneracy = 2N
    – 159 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    where the degeneracy comes from the two orientations (vertical and horizontal). And,
    ﬁnally,
    E3 = E0 + 16J
    Degeneracy = 4N
    where the degeneracy comes from the four orientations (rotating the graph by 90◦).
    Adding all the graphs above together gives us an expansion of the partition function
    in power of e−βJ ⌧1. This is
    Z = 2e2NβJ
    ✓
    1 + Ne−8βJ + 2Ne−12βJ + 1
    2(N 2 + 9N)e−16βJ + . . .
    ◆
    (5.29)
    where the overall factor of 2 originates from the two ground states of the system.
    We’ll make use of the speciﬁc coeﬃcients in this expansion in Section 5.3.4. Before
    we focus on the physics hiding in the low temperature expansion, it’s worth making a
    quick comment that something quite nice happens if we take the log of the partition
    function,
    log Z = log 2 + 2NβJ + Ne−8βJ + 2Ne−12βJ + 9
    2Ne−16βJ + . . .
    The thing to notice is that the N 2 term in the partition function (5.29) has cancelled
    out and log Z is proportional to N, which is to be expected since the free energy of
    the system is extensive. Looking back, we see that the N 2 term was associated to the
    disconnected diagrams in (5.28). There is actually a general lesson hiding here: the
    partition function can be written as the exponential of the sum of connected diagrams.
    We saw exactly the same issue arise in the cluster expansion in (2.37).
    Peierls Droplets
    Continuing the low temperature expansion provides a heuristic, but physically intuitive,
    explanation for why phase transitions happen in d ≥2 dimensions but not in d = 1.
    As we ﬂip more and more spins, the low energy states become droplets, consisting of a
    region of space in which all the spins are ﬂipped, surrounded by a larger sea in which
    the spins have their original alignment. The energy cost of such a droplet is roughly
    E ⇠2JL
    – 160 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    where L is the perimeter of the droplet. Notice that the energy does not scale as the
    area of the droplet since all spins inside are aligned with their neighbours. It is only
    those on the edge which are misaligned and this is the reason for the perimeter scaling.
    To understand how these droplets contribute to the partition function, we also need to
    know their degeneracy. We will now argue that the degeneracy of droplets scales as
    Degeneracy ⇠e↵L
    for some value of ↵. To see this, consider ﬁrstly the problem of a random walk on a 2d
    square lattice. At each step, we can move in one of four directions. So the number of
    paths of length L is
    #paths ⇠4L = eL log 4
    Of course, the perimeter of a droplet is more constrained than a random walk. Firstly,
    the perimeter can’t go back on itself, so it really only has three directions that it can
    move in at each step. Secondly, the perimeter must return to its starting point after L
    steps. And, ﬁnally, the perimeter cannot self-intersect. One can show that the number
    of paths that obey these conditions is
    #paths ⇠e↵L
    where log 2 < ↵< log 3. Since the degeneracy scales as e↵L, the entropy of the droplets
    is proportional to L.
    The fact that both energy and entropy scale with L means that there is an interesting
    competition between them. At temperatures where the droplets are important, the
    partition function is schematically of the form
    Z ⇠
    X
    L
    e↵Le−2βJL
    For large β (i.e. low temperature) the partition function converges. However, as the
    temperature increases, one reaches the critical temperature
    kBTc ⇡2J
    ↵
    (5.30)
    where the partition function no longer converges. At this point, the entropy wins over
    the energy cost and it is favourable to populate the system with droplets of arbitrary
    sizes. This is how one sees the phase transition in the partition function. For temper-
    ature above Tc, the low-temperature expansion breaks down and the ordered magnetic
    phase is destroyed.
    – 161 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    We can also use the droplet argument to see why phase transitions don’t occur in
    d = 1 dimension.
    On a line, the boundary of any droplet always consists of just
    two points. This means that the energy cost to forming a droplet is always E = 2J,
    regardless of the size of the droplet. But, since the droplet can exist anywhere along the
    line, its degeneracy is N. The net result is that the free energy associated to creating
    a droplet scales as
    F ⇠2J −kBT log N
    and, as N ! 1, the free energy is negative for any T > 0. This means that the system
    will prefer to create droplets of arbitrary length, randomizing the spins. This is the
    intuitive reason why there is no magnetic ordered phase in the d = 1 Ising model.
    5.3.3 2d Ising Model: High Temperatures
    We now turn to the 2d Ising model in the opposite limit of high temperature. Here we
    expect the partition function to be dominated by the completely random, disordered
    conﬁgurations of maximum entropy. Our goal is to ﬁnd a way to expand the partition
    function in βJ ⌧1.
    We again work with zero magnetic ﬁeld, B = 0 and write the partition function as
    Z =
    X
    {si}
    exp
    0
    @βJ
    X
    hiji
    sisj
    1
    A =
    X
    {si}
    Y
    hiji
    eβJsisj
    There is a useful way to rewrite eβJsisj which relies on the fact that the product sisj
    only takes ±1. It doesn’t take long to check the following identity:
    eβJsisj = cosh βJ + sisj sinh βJ
    = cosh βJ (1 + sisj tanh βJ)
    Using this, the partition function becomes
    Z =
    X
    {si}
    Y
    hiji
    cosh βJ (1 + sisj tanh βJ)
    = (cosh βJ)qN/2 X
    {si}
    Y
    hiji
    (1 + sisj tanh βJ)
    (5.31)
    where the number of nearest neighbours is q = 4 for the 2d square lattice.
    – 162 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    With the partition function in this form, there is a natural expansion which suggests
    itself.
    At high temperatures βJ ⌧1 which, of course, means that tanh βJ ⌧1.
    But the partition function is now naturally a product of powers of tanh βJ. This is
    somewhat analogous to the cluster expansion for the interacting gas that we met in
    Section 2.5.3. As in the cluster expansion, we will represent the expansion graphically.
    We need no graphics for the leading order term. It has no factors of tanh βJ and is
    simply
    Z ⇡(cosh βJ)2N X
    {si}
    1 = 2N(cosh βJ)2N
    That’s simple.
    Let’s now turn to the leading correction. Expanding the partition function (5.31),
    each power of tanh βJ is associated to a nearest neighbour pair hiji. We’ll represent
    this by drawing a line on the lattice:
    i
    j = sisj tanh βJ
    But there’s a problem: each factor of tanh βJ in (5.31) also comes with a sum over all
    spins si and sj. And these are +1 and −1 which means that they simply sum to zero,
    X
    si,sj
    sisj = +1 −1 −1 + 1 = 0
    How can we avoid this? The only way is to make sure that we’re summing over an even
    number of spins on each site, since then we get factors of s2
    i = 1 and no cancellations.
    Graphically, this means that every site must have an even number of lines attached to
    it. The ﬁrst correction is then of the form
    1
    2
    3
    4
    = (tanh βJ)4 X
    {si}
    s1s2 s2s3 s3s4 s4s1 = 24(tanh βJ)4
    There are N such terms since the upper left corner of the square can be on any one
    of the N lattice sites. (Assuming periodic boundary conditions for the lattice.) So
    including the leading term and ﬁrst correction, we have
    Z = 2N(cosh βJ)2N %
    1 + N(tanh βJ)4 + . . .
    &
    We can go further. The next terms arise from graphs of length 6 and the only possibil-
    ities are rectangles, oriented as either landscape or portrait. Each of them can sit on
    – 163 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    one of N sites, giving a contribution
    +
    = 2N(tanh βJ)6
    Things get more interesting when we look at graphs of length 8. We have four di↵erent
    types of graphs. Firstly, there are the trivial, disconnected pair of squares
    =
    1
    2N(N −5)(tanh βJ)8
    Here the ﬁrst factor of N is the possible positions of the ﬁrst square; the factor of N −5
    arises because the possible location of the upper corner of the second square can’t be
    on any of the vertices of the ﬁrst, but nor can it be on the square one to the left of the
    upper corner of the ﬁrst since that would give a graph that looks like
    which has
    three lines coming o↵the middle site and therefore vanishes when we sum over spins.
    Finally, the factor of 1/2 comes because the two squares are identical.
    The other graphs of length 8 are a large square, a rectangle and a corner. The large
    square gives a contribution
    =
    N(tanh βJ)8
    There are two orientations for the rectangle. Including these gives a factor of 2,
    =
    2N(tanh βJ)8
    Finally, the corner graph has four orientations, giving
    =
    4N(tanh βJ)8
    Adding all contributions together gives us the ﬁrst few terms in high temperature
    expansion of the partition function
    Z = 2N(cosh βJ)2N⇣
    1 + N(tanh βJ)4 + 2N(tanh βJ)6
    + 1
    2(N 2 + 9N)(tanh βJ)8 + . . .
    ⌘
    (5.32)
    – 164 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    There’s some magic hiding in this expansion which we’ll turn to in Section 5.3.4. First,
    let’s just see how the high energy expansion plays out in the d = 1 dimensional Ising
    model.
    The Ising Chain Revisited
    Let’s do the high temperature expansion for the d = 1 Ising
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    mo.md("### Figure 48: Magnetization vs Temperature")
    _T = np.linspace(0, 2, 200)
    _m = np.where(_T < 1, np.sqrt(3*(1-_T)), 0)
    _df = pd.DataFrame({'T': _T, 'm': _m})
    _chart = alt.Chart(_df).mark_line(size=3).encode(
        x=alt.X('T:Q', title='Temperature (T/Tc)'), y=alt.Y('m:Q', title='Spontaneous Magnetization')
    ).properties(width=700, height=400, title="Magnetization as a Function of Temperature").interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    chain with periodic boundary conditions and B = 0. We have the
    same partition function (5.31) and the same issue that only graphs
    with an even number of lines attached to each vertex contribute.
    But, for the Ising chain, there is only one such term: it is the
    closed loop. This means that the partition function is
    Z = 2N(cosh βJ)N %
    1 + (tanh βJ)N&
    In the limit N ! 1, (tanh βJ)N ! 0 at high temperatures and even the contribution
    from the closed loop vanishes. We’re left with
    Z = (2 cosh βJ)N
    This agrees with our exact result for the Ising chain given in (5.27), which can be seen
    by setting B = 0 in (5.26) so that λ+ = 2 cosh βJ.
    5.3.4 Kramers-Wannier Duality
    In the previous sections we computed the partition function perturbatively in two
    extreme regimes of low temperature and high temperature. The physics in the two cases
    is, of course, very di↵erent. At low temperatures, the partition function is dominated by
    the lowest energy states; at high temperatures it is dominated by maximally disordered
    states.
    Yet comparing the partition functions at low temperature (5.29) and high
    temperature (5.32) reveals an extraordinary fact: the expansions are the same! More
    concretely, the two series agree if we exchange
    e−2βJ  ! tanh βJ
    (5.33)
    Of course, we’ve only checked the agreement to the ﬁrst few orders in perturbation
    theory. Below we shall prove that this miracle continues to all orders in perturbation
    theory. The symmetry of the partition function under the interchange (5.33) is known
    as Kramers-Wannier duality. Before we prove this duality, we will ﬁrst just assume
    that it is true and extract some consequences.
    – 165 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    We can express the statement of the duality more clearly. The Ising model at tem-
    perature β is related to the same model at temperature ˜β, deﬁned as
    e−2˜βJ = tanh βJ
    (5.34)
    This way of writing things hides the symmetry of the transformation. A little algebra
    shows that this is equivalent to
    sinh 2˜βJ =
    1
    sinh 2βJ
    Notice that this is a hot/cold duality. When βJ is large, ˜βJ is small. Kramers-Wannier
    duality is the statement that, when B = 0, the partition functions of the Ising model
    at two temperatures are related by
    Z[β] = 2N(cosh βJ)2N
    2e2N ˜βJ
    Z[˜β]
    = 2N−1(cosh βJ sinh βJ)NZ[˜β]
    (5.35)
    This means that if you know the thermodynamics of the Ising model at one temperature,
    then you also know the thermodynamics at the other temperature. Notice however,
    that it does not say that all the physics of the two models is equivalent. In particular,
    when one system is in the ordered phase, the other typically lies in the disordered
    phase.
    One immediate consequence of the duality is that we can use it to compute the
    exact critical temperature Tc. This is the temperature at which the partition function
    is singular in the N ! 1 limit. (We’ll discuss a more reﬁned criterion in Section
    5.4.3.) If we further assume that there is just a single phase transition as we vary the
    temperature, then it must happen at the special self-dual point β = ˜β. This is
    kBT =
    2J
    log(
    p
    2 + 1) ⇡2.269 J
    The exact solution of Onsager conﬁrms that this is indeed the transition temperature.
    It’s also worth noting that it’s fully consistent with the more heuristic Peierls droplet
    argument (5.30) since log 2 < log(
    p
    2 + 1) < log 3.
    Proving the Duality
    So far our evidence for the duality (5.35) lies in the agreement of the ﬁrst few terms
    in the low and high temperature expansions (5.29) and (5.32). Of course, we could
    keep computing further and further terms and checking that they agree, but it would
    be nicer to simply prove the equality between the partition functions. We shall do so
    here.
    – 166 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    The key idea that we need can actually be found by staring hard at the various
    graphs that arise in the two expansions. Eventually, you will realise that they are the
    same, albeit drawn di↵erently. For example, consider the two “corner” diagrams
    vs
    The two graphs are dual. The red lines in the ﬁrst graph intersect the black lines in
    the second as can be seen by placing them on top of each other:
    The same pattern occurs more generally: the graphs appearing in the low temperature
    expansion are in one-to-one correspondence with the dual graphs of the high tempera-
    ture expansion. Here we will show how this occurs and how one can map the partition
    functions onto each other.
    Let’s start by writing the partition function in the form (5.31) that we met in the
    high temperature expansion and presenting it in a slightly di↵erent way,
    Z[β] =
    X
    {si}
    Y
    hiji
    (cosh βJ + sisj sinh βJ)
    =
    X
    {si}
    Y
    hiji
    X
    kij=0,1
    Ckij[βJ] (sisj)kij
    where we have introduced the rather strange variable kij associated to each nearest
    neighbour pair that takes values 0 and 1, together with the functions
    C0[βJ] = cosh βJ
    and
    C1[βJ] = sinh βJ
    The variables in the original Ising model were spins on the lattice sites. The observation
    that the graphs which appear in the two expansions are dual suggests that it might be
    – 167 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    proﬁtable to focus attention on the links between lattice sites. Clearly, we have one link
    for every nearest neighbour pair. If we label these links by l, we can trivially rewrite
    the partition function as
    Z =
    X
    kl=0,1
    Y
    l
    X
    {si}
    Ckl[βJ] (sisj)kl
    Notice that the strange label kij has now become a variable that lives on the links l
    rather than the original lattice sites i.
    At this stage, we do the sum over the spins si. We’ve already seen that if a given
    spin, say si, appears in a term an odd number of times, then that term will vanish when
    we sum over the spin. Alternatively, if the spin si appears an even number of times,
    then the sum will give 2. We’ll say that a given link l is turned on in conﬁgurations
    with kl = 1 and turned o↵when kl = 0. In this language, a term in the sum over spin
    si contributes only if an even number of links attached to site i are turned on. The
    partition function then becomes
    Z = 2N X
    kl
    Y
    l
    Ckl[βJ]
    !!!!!Constrained
    (5.36)
    Now we have something interesting. Rather than summing over spins on lattice sites,
    we’re now summing over the new variables kl living on links.
    This looks like the
    partition function of a totally di↵erent physical system, where the degrees of freedom
    live on the links of the original lattice. But there’s a catch – that big “Constrained”
    label on the sum. This is there to remind us that we don’t sum over all kl conﬁgurations;
    only those for which an even number of links are turned on for every lattice site. And
    that’s annoying. It’s telling us that the kl aren’t really independent variables. There
    are some constraints that must be imposed.
    Fortunately, for the 2d square lattice, there is a simple
    s2
    s5
    4
    s1
    s~
    1
    s~
    4
    s~
    2
    s~
    3
    3s
    k12
    s
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    _mo_caption = mo.md("### Figure 49: Interactive Plot Placeholder")
    _x = np.linspace(0, 10, 100)
    _y = np.sin(_x + int(49))
    _df = pd.DataFrame({'x': _x, 'y': _y})
    _chart = alt.Chart(_df).mark_line(size=3).encode(x='x:Q', y='y:Q', tooltip=['x', 'y']).properties(width=700, height=400, title='Interactive Placeholder 49').interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    way to solve the constraint. We introduce yet more variables, ˜si
    which, like the original spin variables, take values ±1. However,
    the ˜si do not live on the original lattice sites. Instead, they live
    on the vertices of the dual lattice. For the 2d square lattice, the
    dual vertices are drawn in the ﬁgure. The original lattice sites
    are in white; the dual lattice sites in black.
    The link variables kl are related to the two nearest spin vari-
    ables ˜si as follows:
    k12 = 1
    2(1 −˜s1˜s2)
    – 168 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    k13 = 1
    2(1 −˜s2˜s3)
    k14 = 1
    2(1 −˜s3˜s4)
    k15 = 1
    2(1 −˜s1˜s4)
    Notice that we’ve replaced four variables kl taking values 0, 1 with four variables ˜si
    taking values ±1. Each set of variables gives 24 possibilities. However, the map is not
    one-to-one. It is not possible to construct for all values of kl using the parameterization
    in terms of ˜si. To see this, we need only look at
    k12 + k13 + k14 + k15 = 2 −1
    2(˜s1˜s2 + ˜s2˜s3 + ˜s3˜s4 + ˜s1˜s4)
    = 2 −1
    2(˜s1 + ˜s3)(˜s2 + ˜s4)
    = 0, 2, or 4
    In other words, the number of links that are turned on must be even. But that’s exactly
    what we want! Writing the kl in terms of the auxiliary spins ˜si automatically solves the
    constraint that is imposed on the sum in (5.36). Moreover, it is simple to check that for
    every conﬁguration {kl} obeying the constraint, there are two conﬁgurations of {˜si}.
    This means that we can replace the constrained sum over {kl} with an unconstrained
    sum over {˜si}. The only price we pay is an additional factor of 1/2.
    Z[β] = 1
    2 2N X
    {˜si}
    Y
    hiji
    C 1
    2 (1−˜si˜sj)[βj]
    Finally, we’d like to ﬁnd a simple expression for C0 and C1 in terms of ˜si. That’s easy
    enough. We can write
    Ck[βJ] = cosh βJ exp (k log tanh βJ)
    = (sinh βJ cosh βJ)1/2 exp
    ✓
    −1
    2˜si˜sj log tanh βJ
    ◆
    Substituting this into our newly re-written partition function gives
    Z[β] = 2N−1 X
    {˜si}
    Y
    hiji
    (sinh βJ cosh βJ)1/2 exp
    ✓
    −1
    2˜si˜sj log tanh βJ
    ◆
    = 2N−1(sinh βJ cosh βJ)N X
    {˜si}
    exp
    0
    @−1
    2 log tanh βJ
    X
    hiji
    ˜si˜sj
    1
    A
    – 169 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    But this ﬁnal form of the partition function in terms of the dual spins ˜si has exactly the
    same functional form as the original partition function in terms of the spins si. More
    precisely, we can write
    Z[β] = 2N−1(sinh 2βJ)NZ[˜β]
    where
    e−2˜βJ = tanh βJ
    as advertised previously in (5.34). This completes the proof of Kramers-Wannier duality
    in the 2d Ising model on a square lattice.
    The concept of duality of this kind is a major feature in much of modern theoretical
    physics. The key idea is that when the temperature gets large there may be a di↵erent
    set of variables in which a theory can be written where it appears to live at low tem-
    perature. The same idea often holds in quantum theories, where duality maps strong
    coupling problems to weak coupling problems.
    The duality in the Ising model is special for two reasons: ﬁrstly, the new variables
    ˜si are governed by the same Hamiltonian as the original variables si. We say that the
    Ising model is self-dual. In general, this need not be the case — the high temperature
    limit of one system could look like the low-temperature limit of a very di↵erent system.
    Secondly, the duality in the Ising model can be proven explicitly. For most systems,
    we have no such luck. Nonetheless, the idea that there may be dual variables in other,
    more diﬃcult theories, is compelling. Commonly studied examples include the exchange
    particles and vortices in two dimensions, and electrons and magnetic monopoles in three
    dimensions.
    5.4 Landau Theory
    We saw in Sections 5.1 and 5.2 that the van der Waals equation and mean ﬁeld Ising
    model gave the same (sometimes wrong!) answers for the critical exponents. This
    suggests that there should be a uniﬁed way to look at phase transitions. Such a method
    was developed by Landau. It is worth stressing that, as we saw above, the Landau
    approach to phase transitions often only gives qualitatively correct results. However, its
    advantage is that it is extremely straightforward and easy. (Certainly much easier than
    the more elaborate methods needed to compute critical exponents more accurately.)
    – 170 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    The Landau theory of phase transitions is based around the free energy. We will
    illustrate the theory using the Ising model and then explain how to extend it to di↵erent
    systems. The free energy of the Ising model in the mean ﬁeld approximation is readily
    attainable from the partition function (5.17),
    F = −1
    β log Z = 1
    2JNqm2 −N
    β log (2 cosh βBe↵)
    (5.37)
    So far in this course, we’ve considered only systems in equilibrium. The free energy,
    like all other thermodynamic potentials, has only been deﬁned on equilibrium states.
    Yet the equation above can be thought of as an expression for F as a function of m.
    Of course, we could substitute in the equilibrium value of m given by solving (5.18),
    but it seems a shame to throw out F(m) when it is such a nice function. Surely we can
    put it to some use!
    The key step in Landau theory is to treat the function F = F(T, V ; m) seriously.
    This means that we are extending our viewpoint away from equilibrium states to a
    whole class of states which have a constant average value of m. If you want some words
    to drape around this, you could imagine some external magical power that holds m
    ﬁxed. The free energy F(T, V ; m) is then telling us the equilibrium properties in the
    presence of this magical power. Perhaps more convincing is what we do with the free
    energy in the absence of any magical constraint. We saw in Section 4 that equilibrium
    is guaranteed if we sit at the minimum of F. Looking at extrema of F, we have the
    condition
    @F
    @m = 0
    )
    m = tanh βBe↵
    But that’s precisely the condition (5.18) that we saw previously. Isn’t that nice!
    In the context of Landau theory, m is called an order parameter. When it takes non-
    zero values, the system has some degree of order (the spins have a preferred direction
    in which they point) while when m = 0 the spins are randomised and happily point in
    any direction.
    For any system of interest, Landau theory starts by identifying a suitable order
    parameter. This should be taken to be a quantity which vanishes above the critical
    temperature at which the phase transition occurs, but is non-zero below the critical
    temperature. Sometimes it is obvious what to take as the order parameter; other times
    less so. For the liquid-gas transition, the relevant order parameter is the di↵erence in
    densities between the two phases, vgas −vliquid. For magnetic or electric systems, the
    order parameter is typically some form of magnetization (as for the Ising model) or
    – 171 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    the polarization. For the Bose-Einstein condensate, superﬂuids and superconductors,
    the order parameter is more subtle and is related to o↵-diagonal long-range order in
    the one-particle density matrix11, although this is usually rather lazily simpliﬁed to say
    that the order parameter can be thought of as the macroscopic wavefunction | |2.
    Starting from the existence of a suitable order parameter, the next step in the Landau
    programme is to write down the free energy. But that looks tricky. The free energy
    for the Ising model (5.37) is a rather complicated function and clearly contains some
    detailed information about the physics of the spins. How do we just write down the
    free energy in the general case? The trick is to assume that we can expand the free
    energy in an analytic power series in the order parameter. For this to be true, the order
    parameter must be small which is guaranteed if we are close to a critical point (since
    m = 0 for T > Tc). The nature of the phase transition is determined by the kind of
    terms that appear in the expansion of the free energy. Let’s look at a couple of simple
    examples.
    5.4.1 Second Order Phase Transitions
    We’ll consider a general system (Ising model; liquid-gas; BEC; whatever) and denote
    the order parameter as m. Suppose that the expansion of the free energy takes the
    general form
    F(T; m) = F0(T) + a(T)m2 + b(T)m4 + . . .
    (5.38)
    One common reason why the free energy has this form is because the theory has a
    symmetry under m ! −m, forbidding terms with odd powers of m in the expansion.
    For example, this is the situation in the Ising model when B = 0.
    Indeed, if we
    expand out the free energy (5.37) for the Ising model for small m using cosh x ⇡
    1 + 1
    2x2 + 1
    4!x4 + . . . and log(1 + y) ⇡y −1
    2y2 + . . . we get the general form above with
    explicit expressions for F0(T), a(T) and b(T),
    FIsing(T; m) = −NkBT log 2 +
    ✓NJq
    2
    (1 −Jqβ)
    ◆
    m2 +
    ✓Nβ3J4q4
    24
    ◆
    m4 + . . .
    The leading term F0(T) is unimportant for our story. We are interested in how the free
    energy changes with m. The condition for equilibrium is given by
    @F
    @m = 0
    (5.39)
    But the solutions to this equation depend on the sign of the coeﬃcients a(T) and
    11See, for example, the book “Quantum Liquids” by Anthony Leggett
    – 172 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    F
    m
    F
    m
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    _mo_caption = mo.md("### Figure 50: Interactive Plot Placeholder")
    _x = np.linspace(0, 10, 100)
    _y = np.sin(_x + int(50))
    _df = pd.DataFrame({'x': _x, 'y': _y})
    _chart = alt.Chart(_df).mark_line(size=3).encode(x='x:Q', y='y:Q', tooltip=['x', 'y']).properties(width=700, height=400, title='Interactive Placeholder 50').interactive()
    return


@app.cell
def _(alt, mo, np, pd):
    _mo_caption = mo.md("### Figure 51: Interactive Plot Placeholder")
    _x = np.linspace(0, 10, 100)
    _y = np.sin(_x + int(51))
    _df = pd.DataFrame({'x': _x, 'y': _y})
    _chart = alt.Chart(_df).mark_line(size=3).encode(x='x:Q', y='y:Q', tooltip=['x', 'y']).properties(width=700, height=400, title='Interactive Placeholder 51').interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    b(T). Moreover, this sign can change with temperature. This is the essence of the
    phase transitions. In the following discussion, we will assume that b(T) > 0 for all T.
    (If we relax this condition, we have to also consider the m6 term in the free energy
    which leads to interesting results concerning so-called tri-critical points.)
    The two ﬁgures above show sketches of the free energy in the case where a(T) > 0
    and a(T) < 0. Comparing to the explicit free energy of the Ising model, a(T) > 0
    when T > Tc = Jq/kB and a(T) < 0 when T < Tc. When a(T) > 0, we have just
    a single equilibrium solution to (5.39) at m = 0. This is typically the situation at
    high temperatures. In contrast, at a(T) < 0, there are three solutions to (5.39). The
    solution m = 0 clearly has higher free energy: this is now the unstable solution. The
    two stable solutions sit at m = ±m0. For example, if we choose to truncate the free
    energy (5.38) at quartic order, we have
    m0 =
    r
    −a
    2b
    T < Tc
    If a(T) is a smooth function then the equilibrium value of m changes continuously from
    m = 0 when a(T) > 0 to m 6= 0 at a(T) < 0. This describes a second order phase
    transition occurring at Tc, deﬁned by a(Tc) = 0.
    Once we know the equilibrium value of m, we can then substitute this back into
    the free energy F(T; m) in (5.38). This gives the thermodynamic free energy F(T) of
    the system in equilibrium that we have been studying throughout this course. For the
    quartic free energy, we have
    F(T) =
    (
    F0(T)
    T > Tc
    F0(T) −a2/4b
    T < Tc
    (5.40)
    Because a(Tc) = 0, the equilibrium free energy F(T) is continuous at T = Tc. Moreover,
    the entropy S = −@F/@T is also continuous at T = Tc. However, if you di↵erentiate the
    – 173 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    equilibrium free energy twice, you will get a term a0 2/b which is generically not vanishing
    at T = Tc. This means that the heat capacity C = T@S/@T changes discontinuously
    at T = Tc, as beﬁts a second order phase transition. A word of warning: if you want to
    compute equilibrium quantities such as the heat capacity, it’s important that you ﬁrst
    substitute in the equilibrium value of m and work with (5.40) rather than with (5.38).
    If you don’t, you miss the fact that the magnetization also changes with T.
    We can easily compute critical exponents within the context of Landau theory. We
    need only make further assumptions about the behaviour of a(T) and b(T) in the
    vicinity of Tc. If we assume that near T = Tc, we can write
    b(T) ⇡b0
    ,
    a(T) ⇡a0(T −Tc)
    (5.41)
    then we have
    m0 ⇡±
    r a0
    2b0
    (Tc −T)1/2
    T < Tc
    which reproduces the critical exponent (5.9) and (5.20) that we derived for the van der
    Waals equation and Ising model respectively.
    Notice that we didn’t put any discontinuities into the free energy. Everything in
    F(T; m) was nice and smooth. When Taylor expanded, it has only integer powers of
    m and T as shown in (5.38) and (5.41). But the minima of F behave in a non-analytic
    fashion as seen in the expression for m0 above.
    Landau’s theory of phase transitions predicts this same critical exponent for all values
    of the dimension d of the system. But we’ve already mentioned in previous contexts
    that the critical exponent is in fact only correct for d ≥4. We will understand how to
    derive this criterion from Landau theory in the next section.
    Spontaneous Symmetry Breaking
    As we approach the end of the course, we’re touching upon a number of ideas that
    become increasingly important in subsequent developments in physics.
    We already
    brieﬂy met the idea of universality and critical phenomena. Here I would like to point
    out another very important idea: spontaneous symmetry breaking.
    The free energy (5.38) is invariant under the Z2 symmetry m ! −m. Indeed, we
    said that one common reason that we can expand the free energy only in even powers
    of m is that the underlying theory also enjoys this symmetry. But below Tc, the system
    must pick one of the two ground states m = +m0 or m = −m0. Whichever choice it
    makes breaks the Z2 symmetry. We say that the symmetry is spontaneously broken by
    the choice of ground state of the theory.
    – 174 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Spontaneous symmetry breaking has particularly dramatic consequences when the
    symmetry in question is continuous rather than discrete.
    For example, consider a
    situation where the order parameter is a complex number  and the free energy is given
    by (5.38) with m = | |2. (This is e↵ectively what happens for BECs, superﬂuids and
    superconductors.) Then we should only look at the m > 0 solutions so that the ground
    state has | |2 = +m0. But this leaves the phase of  completely undetermined. So
    there is now a continuous choice of ground states: we get to sit anywhere on the circle
    parameterised by the phase of  . Any choice that the system makes spontaneously
    breaks the U(1) rotational symmetry which acts on the phase of  . Some beautiful
    results due to Nambu and Goldstone show that much of the physics of these systems
    can be understood simply as a consequence of this symmetry breaking.
    The ideas
    of spontaneous symmetry breaking are crucial in both condensed matter physics and
    particle physics. In the latter context, it is intimately tied with the Higgs mechanism.
    5.4.2 First Order Phase Transitions
    Let us now consider a situation where the expansion of the free energy also includes
    odd powers of the order parameter
    F(T; m) = F0(T) + ↵(T)m + a(T)m2 + γ(T)m3 + b(T)m4 + . . .
    For example, this is the kind of expansion that we get for the Ising model free energy
    (5.37) when B 6= 0, which reads
    FIsing(T; m) = −NkBT log 2 + JNq
    2
    m2 −
    N
    2kBT (B + Jqm)2 +
    N
    24(kBT)3(B + Jqm)4 + . . .
    Notice that there is no longer a symmetry relating m ! −m: the B ﬁeld has a
    preference for one sign over the other.
    If we again assume that b(T) > 0 for all temperatures, the crude shape of the free
    energy graph again has two choices: there is a single minimum, or two minima and a
    local maximum.
    Let’s start at suitably low temperatures for which the situation is depicted in Figure
    52. The free energy once again has a double well, except now slightly skewed. The local
    maximum is still an unstable point. But this time around, the minima with the lower
    free energy is preferred over the other one. This is the true ground state of the system.
    In contrast, the point which is locally, but not globally, a minimum corresponds to a
    meta-stable state of the system. In order for the system to leave this state, it must
    ﬁrst ﬂuctuate up and over the energy barrier separating the two.
    – 175 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    F
    F
    m
    F
    m
    m
    B<0
    B=0
    B>0
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    _mo_caption = mo.md("### Figure 52: Interactive Plot Placeholder")
    _x = np.linspace(0, 10, 100)
    _y = np.sin(_x + int(52))
    _df = pd.DataFrame({'x': _x, 'y': _y})
    _chart = alt.Chart(_df).mark_line(size=3).encode(x='x:Q', y='y:Q', tooltip=['x', 'y']).properties(width=700, height=400, title='Interactive Placeholder 52').interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    In this set-up, we can initiate a ﬁrst order phase transition. This occurs when the
    coeﬃcient of the odd terms, ↵(T) and γ(T) change sign and the true ground state
    changes discontinuously from m < 0 to m > 0.
    In some systems this behaviour
    occurs when changing temperature; in others it could occur by changing some external
    parameter. For example, in the Ising model the ﬁrst order phase transition is induced
    by changing B.
    At very high temperature, the double well poten-
    F
    m
    """)
    return


@app.cell
def _(alt, mo, np, pd):
    _mo_caption = mo.md("### Figure 53: Interactive Plot Placeholder")
    _x = np.linspace(0, 10, 100)
    _y = np.sin(_x + int(53))
    _df = pd.DataFrame({'x': _x, 'y': _y})
    _chart = alt.Chart(_df).mark_line(size=3).encode(x='x:Q', y='y:Q', tooltip=['x', 'y']).properties(width=700, height=400, title='Interactive Placeholder 53').interactive()
    return


@app.cell
def _(mo):
    mo.md("""
    tial is lost in favour of a single minimum as depicted in
    the ﬁgure to the right. There is a unique ground state, al-
    beit shifted from m = 0 by the presence of the ↵(T) term
    above (which translates into the magnetic ﬁeld B in the
    Ising model). The temperature at which the meta-stable
    ground state of the system is lost corresponds to the spin-
    odal point in our discussion of the liquid-gas transition.
    One can play further games in Landau theory, looking at how the shape of the free
    energy can change as we vary temperature or other parameters. One can also use this
    framework to give a simple explanation of the concept of hysteresis. You can learn
    more about these from the links on the course webpage.
    5.4.3 Lee-Yang Zeros
    You may have noticed that the ﬂavour of our discussion of phase transitions is a little
    di↵erent from the rest of the course. Until now, our philosophy was to derive everything
    from the partition function. But in this section, we dumped the partition function as
    soon as we could, preferring instead to work with the macroscopic variables such as the
    free energy. Why didn’t we just stick with the partition function and examine phase
    transitions directly?
    – 176 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    The reason, of course, is that the approach using the partition function is hard! In
    this short section, which is somewhat tangential to our main discussion, we will describe
    how phase transitions manifest themselves in the partition function.
    For concreteness, let’s go back to the classical interacting gas of Section 2.5, although
    the results we derive will be more general. We’ll work in the grand canonical ensemble,
    with the partition function
    Z(z, V, T) =
    X
    N
    zNZ(N, V, T) =
    X
    N
    zN
    N!λ3N
    Z Y
    i
    d3ri e−β P
    j<k U(rjk)
    (5.42)
    To regulate any potential diﬃculties with short distances, it is useful to assume that
    the particles have hard-cores so that they cannot approach to a distance less than r0.
    We model this by requiring that the potential satisﬁes
    U(rjk) = 1
    for
    rjk < r0
    But this has an obvious consequence: if the particles have ﬁnite size, then there is a
    maximum number of particles, NV , that we can ﬁt into a ﬁnite volume V . (Roughly
    this number is NV ⇠V/r3
    0.) But that, in turn, means that the canonical partition
    function Z(N, V, T) = 0 for N > NV , and the grand partition function Z is therefore
    a ﬁnite polynomial in the fugacity z, of order NV . But if the partition function is a
    ﬁnite polynomial, there can’t be any discontinuous behaviour associated with a phase
    transition. In particular, we can calculate
    pV = kBT log Z
    (5.43)
    which gives us pV as a smooth function of z. We can also calculate
    N = z @
    @z log Z
    (5.44)
    which gives us N as a function of z. Eliminating z between these two functions (as
    we did for both bosons and fermions in Section 3) tells us that pressure p is a smooth
    function of density N/V . We’re never going to get the behaviour that we derived from
    the Maxwell construction in which the plot of pressure vs density shown in Figure 37
    exhibits a discontinous derivative.
    The discussion above is just re-iterating a statement that we’ve alluded to several
    times already: there are no phase transitions in a ﬁnite system. To see the discontinuous
    behaviour, we need to take the limit V ! 1. A theorem due to Lee and Yang12 gives
    us a handle on the analytic properties of the partition function in this limit.
    12This theorem was ﬁrst proven for the Ising model in 1952. Soon afterwards, the same Lee and
    Yang proposed a model of parity violation in the weak interaction for which they won the 1957 Nobel
    prize.
    – 177 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    The surprising insight of Lee and Yang is that if you’re interested in phase transitions,
    you should look at the zeros of Z in the complex z-plane. Let’s ﬁrstly look at these
    when V is ﬁnite. Importantly, at ﬁnite V there can be no zeros on the positive real axis,
    z > 0. This follows from the deﬁnition of Z given in (5.42) where it is a sum of positive
    quantities. Moreover, from (5.44), we can see that Z is a monotonically increasing
    function of z because we necessarily have N > 0. Nonetheless, Z is a polynomial in
    z of order NV so it certainly has NV zeros somewhere in the complex z-plane. Since
    Z?(z) = Z(z?), these zeros must either sit on the real negative axis or come in complex
    pairs.
    However, the statements above rely on the fact that Z is a ﬁnite polynomial. As we
    take the limit V ! 1, the maximum number of particles that we can ﬁt in the system
    diverges, NV ! 1, and Z is now deﬁned as an inﬁnite series. But inﬁnite series can do
    things that ﬁnite ones can’t. The Lee-Yang theorem says that as long as the zeros of Z
    continue to stay away from the positive real axis as V ! 1, then no phase transitions
    can happen. But if one or more zeros happen to touch the positive real axis, life gets
    more interesting.
    More concretely, the Lee-Yang theorem states:
    • Lee-Yang Theorem: The quantity
    ⇥= lim
    V !1
    ✓1
    V log Z(z, V, T)
    ◆
    exists for all z > 0. The result is a continuous, non-decreasing function of z which
    is independent of the shape of the box (up to some sensible assumptions such as
    Surface Area/V ⇠V −1/3 which ensures that the box isn’t some stupid fractal
    shape).
    Moreover, let R be a ﬁxed, volume independent, region in the complex z plane
    which contains part of the real, positive axis. If R contains no zero of Z(z, V, T)
    for all z 2 R then ⇥is an analytic function of z for all z 2 R. In particular, all
    derivatives of ⇥are continuous.
    In other words, there can be no phase transitions in the region R even in the V ! 1
    limit.
    The last result means that, as long as we are safely in a region R, taking
    derivatives with respect to z commutes with the limit V ! 1. In other words, we are
    allowed to use (5.44) to write the particle density n = N/V as
    lim
    V !1 n = lim
    V !1 z @
    @z
    ✓p
    kBT
    ◆
    = z@⇥
    @z
    – 178 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    However, if we look at points z where zeros appear on the positive real axis, then ⇥
    will generally not be analytic. If d⇥/dz is discontinuous, then the system is said to
    undergo a ﬁrst order phase transition. More generally, if dm⇥/dzm is discontinuous for
    m = n, but continuous for all m < n, then the system undergoes an nth order phase
    transition. We won’t o↵er a proof of the Lee-Yang theorem. Instead we just illustrate
    the general idea with an example.
    A Made-Up Example
    Ideally, we would like to start with a Hamiltonian which exhibits a ﬁrst order phase
    transition, compute the associated grand partition function Z and then follow its zeros
    as V ! 1. However, as we mentioned above, that’s hard! Instead we will simply
    make up a partition function Z which has the appropriate properties. Our choice is
    somewhat artiﬁcial,
    Z(z, V ) = (1 + z)[↵V ](1 + z[↵V ])
    Here ↵is a constant which will typically depend on temperature, although we’ll suppress
    this dependence in what follows. Also,
    [x] = Integer part of x
    Although we just made up the form of Z, it does have the behaviour that one would
    expect of a partition function. In particular, for ﬁnite V , the zeros sit at
    z = −1
    and
    z = e⇡i(2n+1)/[↵V ]
    n = 0, 1, . . . , [↵V ] −1
    As promised, none of the zeros sit on the positive real axis. However, as we increase
    V , the zeros become denser and denser on the unit circle. From the Lee-Yang theorem,
    we expect that no phase transition will occur for z 6= 1 but that something interesting
    could happen at z = 1.
    Let’s look at what happens as we send V ! 1. We have
    ⇥= lim
    V !1
    1
    V log Z(z, V )
    = lim
    V !1
    1
    V
    %
    [↵V ] log(1 + z) + log(1 + z[↵V ])
    &
    =
    (
    ↵log(1 + z)
    |z| < 1
    ↵log(1 + z) + ↵log z
    |z| > 1
    We see that ⇥is continuous for all z as promised. But it is only analytic for |z| 6= 1.
    – 179 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    We can extract the physics by using (5.43) and (5.44) to eliminate the dependence
    on z. This gives us the equation of state, with pressure p as a function of n = N/V .
    For |z| < 1, we have
    p = ↵kBT log
    ✓
    ↵
    ↵−n
    ◆
    n 2 [0, ↵/2) ,
    p < kBT log 2
    While for |z| > 1, we have
    p = ↵kBT log
    ✓↵(n −↵)
    (2↵−n)2
    ◆
    n 2 (3↵/2, 2↵) ,
    p > kBT log 2
    The key point is that there is a jump in particle density of ∆n = ↵at p = ↵kBT log 2.
    Plotting this as a function of p vs v = 1/n, we ﬁnd that we have a curve that is qualita-
    tively identical to the pressure-volume plot of the liquid-gas phase diagram under the
    co-existence curve. (See, for example, ﬁgure 37.) This is a ﬁrst order phase transition.
    5.5 Landau-Ginzburg Theory
    Landau’s theory of phase transition focusses only on the average quantity, the order
    parameter. It ignores the ﬂuctuations of the system, assuming that they are negligible.
    Here we sketch a generalisation which attempts to account for these ﬂuctuations. It is
    known as Landau-Ginzburg theory.
    The idea is to stick with the concept of the order parameter, m. But now we allow
    the order parameter to vary in space so it becomes a function m(~r). Let’s restrict
    ourselves to the situation where there is a symmetry of the theory m ! −m so we
    need only consider even powers in the expansion of the free energy. We add to these
    a gradient term whose role is to captures the fact that there is some sti↵ness in the
    system, so it costs energy to vary the order parameter from one point to another. (For
    the example of the Ising model, this is simply the statement that nearby spins want to
    be aligned.) The free energy is then given by
    F[m(~r)] =
    Z
    ddr
    ⇥
    a(T)m2 + b(T)m4 + c(T)(rm)2⇤
    (5.45)
    where we have dropped the constant F0(T) piece which doesn’t depend on the order
    parameter and hence plays no role in the story.
    Notice that we start with terms
    quadratic in the gradient: a term linear in the gradient would violate the rotational
    symmetry of the system.
    – 180 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    We again require that the free energy is minimised. But now F is a functional – it is
    a function of the function m(~r). To ﬁnd the stationary points of such objects we need
    to use the same kind of variational methods that we use in Lagrangian mechanics. We
    write the variation of the free energy as
    δF =
    Z
    ddr
    ⇥
    2am δm + 4bm3 δm + 2crm · rδm
    ⇤
    =
    Z
    ddr
    ⇥
    2am + 4bm3 −2cr2m
    ⇤
    δm
    where to go from the ﬁrst line to the second we have integrated by parts. (We need
    to remember that c(T) is a function of temperature but does not vary in space so
    that r doesn’t act on it.) The minimum of the free energy is then determined by
    setting δF = 0 which means that we have to solve the Euler-Lagrange equations for
    the function m(~r),
    cr2m = am + 2bm3
    (5.46)
    The simplest solutions to this equation have m constant, reducing us back to Landau
    theory. We’ll assume once again that a(T) > 0 for T > Tc and a(T) < 0 for T < Tc.
    Then the constant solutions are m = 0 for T > Tc and m = ±m0 = ±
    p
    −a/2b for
    T < Tc. However, allowing for the possibility of spatial variation in the order parameter
    also opens up the possibility for us to search for more interesting solutions.
    Domain Walls
    Suppose that we have T < Tc so there exist two degenerate ground states, m = ±m0.
    We could cook up a situation in which one half of space, say x < 0, lives in the ground
    state m = −m0 while the other half of space, x > 0 lives in m = +m0. This is exactly
    the situation that we already met in the liquid-gas transition and is depicted in Figure
    38. It is also easy to cook up the analogous conﬁguration in the Ising model. The two
    regions in which the spins point up or down are called domains. The place where these
    regions meet is called the domain wall.
    We would like to understand the structure of the domain wall. How does the system
    interpolate between these two states?
    The transition can’t happen instantaneously
    because that would result in the gradient term (rm)2 giving an inﬁnite contribution
    to the free energy. But neither can the transition linger too much because any point at
    which m(~r) di↵ers signiﬁcantly from the value m0 costs free energy from the m2 and
    m4 terms. There must be a happy medium between these two.
    – 181 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    To describe the system with two domains, m(~r) must vary but it need only change
    in one direction: m = m(x). Equation (5.46) then becomes an ordinary di↵erential
    equation,
    d2m
    dx2 = am
    c + 2bm3
    c
    This equation is easily solved. We should remember that in order to have two vacua,
    T < Tc which means that a < 0. We then have
    m = m0 tanh
     r
    −a
    2c x
    !
    where m0 =
    p
    −a/2b is the constant ground state solution for the spin. As x ! ±1,
    the tanh function tends towards ±1 which means that m ! ±m0. So this solution
    indeed interpolates between the two domains as required. We learn that the width of
    the domain wall is given by
    p
    −2c/a. Outside of this region, the magnetisation relaxes
    exponentially quickly back to the ground state values.
    We can also compute the cost in free energy due to the presence of the domain wall.
    To do this, we substitute the solution back into the expression for the free energy (5.45).
    The cost is not proportional to the volume of the system, but instead proportional to
    the area of the domain wall. This means that if the system has linear size L then the
    free energy of the ground state scales as Ld while the free energy required by the wall
    scales only as Ld−1. It is simple to ﬁnd the parametric dependence of this domain wall
    energy without doing any integrals; the energy per unit area scales as
    p
    −ca3/b. Notice
    that as we approach the critical point, and a ! 0, the two vacua are closer, the width
    of the domain wall increases and its energy decreases.
    5.5.1 Correlations
    One of the most important applications of Landau-Ginzburg theory is to understand
    the correlations between ﬂuctuations of the system at di↵erent points in space. Suppose
    that we know that the system has an unusually high ﬂuctuation away from the average
    at some point in space, let’s say the origin ~r = 0. What is the e↵ect of this on nearby
    points?
    There is a simple way to answer this question that requires us only to solve the
    di↵erential equation (5.46). However, there is also a more complicated way to derive
    the same result which has the advantage of stressing the underlying physics and the
    role played by ﬂuctuations. Below we’ll start by deriving the correlations in the simple
    manner. We’ll then see how it can also be derived using more technical machinery.
    – 182 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    We assume that the system sits in a given ground state, say m = +m0, and imagine
    small perturbations around this. We write the magnetisation as
    m(~r) = m0 + δm(~r)
    (5.47)
    If we substitute this into equation (5.46) and keep only terms linear in δm, we ﬁnd
    r2δm + 2a
    c δm = 0
    where we have substituted m2
    0 = −a/2b to get this result.
    (Recall that a < 0 in
    the ordered phase.) We now perturb the system. This can be modelled by putting a
    delta-function source at the origin, so that the above equation becomes
    r2δm + 2a
    c δm = 1
    2c δd(0)
    where the strength of the delta function has been chosen merely to make the equation
    somewhat nicer. It is straightforward to solve the asymptotic behaviour of this equa-
    tion. Indeed, it is the same kind of equation that we already solved when discussing
    the Debye-H¨uckel model of screening. Neglecting constant factors, it is
    δm(~r) ⇠
    e−r/⇠
    r(d−1)/2
    (5.48)
    This tells us how the perturbation decays as we move away from the origin.
    This
    equation has several names, reﬂecting the fact that it arises in many contexts.
    In
    liquids, it is usually called the Ornstein-Zernike correlation. It also arises in particle
    physics as the Yukawa potential. The length scale ⇠is called the correlation length
    ⇠=
    r
    −c
    2a
    (5.49)
    The correlation length provides a measure of the distance it takes correlations to decay.
    Notice that as we approach a critical point, a ! 0 and the correlation length diverges.
    This provides yet another hint that we need more powerful tools to understand the
    physics at the critical point. We will now take the ﬁrst baby step towards developing
    these tools.
    5.5.2 Fluctuations
    The main motivation to allow the order parameter to depend on space is to take into
    account the e↵ect of ﬂuctuations. To see how we can do this, we ﬁrst need to think a
    little more about the meaning of the quantity F[m(~r)] and what we can use it for.
    – 183 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    To understand this point, it’s best if we go back to basics. We know that the true
    free energy of the system can be equated with the log of the partition function (1.36).
    We’d like to call the true free energy of the system F because that’s the notation that
    we’ve been using throughout the course. But we’ve now called the Landau-Ginzburg
    functional F[m(~r)] and, while it’s closely related to the true free energy, it’s not quite
    the same thing as we shall shortly see. So to save some confusion, we’re going to change
    notation at this late stage and call the true free energy A. Equation (1.36) then reads
    A = −kBT log Z, which we write as
    e−βA = Z =
    X
    n
    e−βEn
    We would like to understand the right way to view the functional F[m(~r)] in this frame-
    work. Here we give a heuristic and fairly handwaving argument. A fuller treatment
    involves the ideas of the renormalisation group.
    The idea is that each microstate |ni of the system can be associated to some speciﬁc
    function of the spatially varying order parameter m(~r). To illustrate this, we’ll talk
    in the language of the Ising model although the discussion generalises to any system.
    For the Ising model, we could associate a magnetisation m(~r) to each lattice site by
    simply averaging over all the spins within some distance of that point. Clearly, this will
    only lead to functions that take values on lattice sites rather than in the continuum.
    But if the functions are suitably well behaved it should be possible to smooth them out
    into continuous functions m(~r) which are essentially constant on distance scales smaller
    than the lattice spacing. In this way, we get a map from the space of microstates to
    the magnetisation, |ni 7! m(~r). But this map is not one-to-one. For example, if the
    averaging procedure is performed over enough sites, ﬂipping the spin on just a single
    site is unlikely to have much e↵ect on the average. In this way, many microstates map
    onto the same average magnetisation. Summing over just these microstates provides a
    ﬁrst principles construction of the F[m(~r)],
    e−βF[m(~r)] =
    X
    n|m(~r)
    e−βEn
    (5.50)
    Of course, we didn’t actually perform this procedure to get to (5.45): we simply wrote
    down the most general form in the vicinity of a critical point with a bunch of unknown
    coeﬃcients a(T), b(T) and c(T). But if we were up for a challenge, the above procedure
    tells us how we could go about ﬁguring out those functions from ﬁrst principles. More
    importantly, it also tells us what we should do with the Landau-Ginzburg free energy.
    Because in (5.50) we have only summed over those states that correspond to a particular
    – 184 –
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    value of m(~r). To compute the full partition function, we need to sum over all states.
    But we can do that by summing over all possible values of m(~r). In other words,
    Z =
    Z
    Dm(~r) e−βF[m(~r)]
    (5.51)
    This is a tricky beast: it is a functional integral. We are integrating over all possible
    functions m(~r), which is the same thing as performing an inﬁnite number of integra-
    tions. (Actually, because the order parameters m(~r) arose from an underlying lattice
    and are suitably smooth on short distance scales, the problem is somewhat mitigated.)
    The result (5.51) is physically very nice, albeit mathematically somewhat daunting.
    It means that we should view the Landau-Ginzburg free energy as a new e↵ective
    Hamiltonian for a continuous variable m(~r). It arises from performing the partition
    function sum over much of the microscopic information, but still leaves us with a ﬁnal
    sum, or integral, over ﬂuctuations in an averaged quantity, namely the order parameter.
    To complete the problem, we need to perform the functional integral (5.51). This
    is hard.
    Here “hard” means that the majority of unsolved problems in theoretical
    physics can be boiled down to performing integrals of this type. Yet the fact it’s hard
    shouldn’t dissuade us, since there is a wealth of rich and beautiful physics hiding in
    the path integral, including the deep reason behind the magic of universality. We will
    start to explore some of these ideas in next year’s course on Statistical Field Theory.
    – 185 –
    """)
    return


if __name__ == "__main__":
    app.run()

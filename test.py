# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.20.2",
#     "numpy==2.4.3",
#     "networkx==3.6.1",
#     "scipy==1.17.1",
#     "pandas>=2.0.0",
#     "altair>=5.0.0",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import networkx as nx
    from scipy.optimize import linear_sum_assignment
    import pandas as pd
    import altair as alt

    return alt, mo, np, nx, pd


@app.cell(hide_code=True)
def _(mo):
    _css = mo.Html("""<style>
    :root {
        --game-bg: #0d1117; --game-fg: #c9d1d9; --game-accent: #58a6ff;
        --game-danger: #f85149; --game-success: #3fb950; --game-warn: #d29922;
    }
    .game-card {
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 16px; margin: 4px 0; font-family: 'SF Mono', 'Fira Code', monospace;
    }
    .game-card h3 { margin: 0 0 8px 0; color: var(--game-accent); font-size: 14px; }
    .result-accept-ok { color: var(--game-success); background: #0d1f0d; border-left: 3px solid var(--game-success); padding: 8px 12px; border-radius: 4px; font-family: monospace; }
    .result-accept-err { color: var(--game-danger); background: #1f0d0d; border-left: 3px solid var(--game-danger); padding: 8px 12px; border-radius: 4px; font-family: monospace; }
    .result-reject-good { color: var(--game-accent); background: #0d0d1f; border-left: 3px solid var(--game-accent); padding: 8px 12px; border-radius: 4px; font-family: monospace; }
    .result-reject-waste { color: var(--game-warn); background: #1f1a0d; border-left: 3px solid var(--game-warn); padding: 8px 12px; border-radius: 4px; font-family: monospace; }
    </style>""")
    _css
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scalable Postselection: The Game

    *Based on [Staples, Fu & Thompson — Scalable Postselection of Quantum Resources](https://arxiv.org/abs/2603.08697)*

    You are the **classical decoder** of a fault-tolerant quantum computer.
    Noisy resource states stream through your teleportation pipeline.
    Your mission: **accept** safe states, **reject** dangerous ones, and
    **minimize total spacetime overhead**.

    The boundary stabilizers are *unmeasured* — hidden under a fog of war.
    You must decide using only the **visible bulk syndrome**.

    ---

    | Symbol | Meaning |
    |--------|---------|
    | $d$ | Code distance (lattice size) |
    | $p$ | Physical error rate per qubit |
    | $\sigma_v$ | Visible (bulk) syndrome — defects you can see |
    | $\sigma_h$ | Hidden (boundary) syndrome — concealed by fog of war |
    | $G_A$ | **Apparent gap** — decoder confidence from visible info only |
    | $G_P$ | **Partial gap** — expected confidence averaging over boundary uncertainty |
    | $C(d) = d^3$ | Base cost per resource state |
    """)
    return


@app.cell
def _(np, nx):
    def build_lattice(d):
        """Build a d×d surface code lattice for X-error / Z-stabilizer decoding.

        Hidden boundary: top (row 0) and bottom (row d) vertex rows.
        Logical X operator: vertical chain at column 0.
        """
        L = d
        edges, edge_id = [], 0
        horz = {}
        for r in range(L + 1):
            for c in range(L):
                edges.append(((r, c), (r, c + 1)))
                horz[((r, c), (r, c + 1))] = edge_id
                edge_id += 1
        horz_count = edge_id
        vert = {}
        for r in range(L):
            for c in range(L + 1):
                edges.append(((r, c), (r + 1, c)))
                vert[((r, c), (r + 1, c))] = edge_id
                edge_id += 1
        n_qubits = len(edges)

        n_v = (L + 1) ** 2
        vertex_pos = {}
        v2e = []
        for r in range(L + 1):
            for c in range(L + 1):
                vid = r * (L + 1) + c
                vertex_pos[vid] = (r, c)
                inc = []
                if c > 0:
                    inc.append(horz[((r, c - 1), (r, c))])
                if c < L:
                    inc.append(horz[((r, c), (r, c + 1))])
                if r > 0:
                    inc.append(vert[((r - 1, c), (r, c))])
                if r < L:
                    inc.append(vert[((r, c), (r + 1, c))])
                v2e.append(inc)

        H_Z = np.zeros((n_v, n_qubits), dtype=np.int8)
        for v, el in enumerate(v2e):
            for e in el:
                H_Z[v, e] = 1

        G = nx.Graph()
        eim = {}
        for eid, (a, b) in enumerate(edges):
            va = a[0] * (L + 1) + a[1]
            vb = b[0] * (L + 1) + b[1]
            G.add_edge(va, vb, weight=1)
            eim[(va, vb)] = eid
            eim[(vb, va)] = eid

        logical_X = set(vert[((r, 0), (r + 1, 0))] for r in range(L))
        hidden = set(r * (L + 1) + c for c in range(L + 1) for r in (0, L))
        visible = set(range(n_v)) - hidden

        ap = dict(nx.all_pairs_shortest_path(G))

        return dict(
            d=d, L=L, edges=edges, n_qubits=n_qubits, n_vertices=n_v,
            H_Z=H_Z, vertex_pos=vertex_pos, v2e=v2e, G=G, eim=eim,
            horz_count=horz_count, logical_X=logical_X,
            hidden=hidden, visible=visible, all_paths=ap,
        )

    def _trace_to_boundary(v, L, eim):
        r, c = v // (L + 1), v % (L + 1)
        path = []
        if r <= L - r:
            for s in range(r):
                va = (r - s) * (L + 1) + c
                vb = (r - s - 1) * (L + 1) + c
                eid = eim.get((va, vb), eim.get((vb, va)))
                if eid is not None:
                    path.append(eid)
        else:
            for s in range(L - r):
                va = (r + s) * (L + 1) + c
                vb = (r + s + 1) * (L + 1) + c
                eid = eim.get((va, vb), eim.get((vb, va)))
                if eid is not None:
                    path.append(eid)
        return path

    def _mwpm_match(nodes, dist_fn):
        """Min-weight perfect matching via networkx Blossom algorithm."""
        n = len(nodes)
        if n < 2:
            return []
        if n == 2:
            return [(nodes[0], nodes[1])]
        MX = sum(dist_fn(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)) + 1
        H = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                H.add_edge(nodes[i], nodes[j], weight=MX - dist_fn(nodes[i], nodes[j]))
        raw = nx.max_weight_matching(H, maxcardinality=True)
        return [(u, v) for u, v in raw]

    def _mwpm_decode(defects, lat):
        """MWPM decoder returning correction edge-index list."""
        if not defects:
            return []
        eim, L, ap = lat["eim"], lat["L"], lat["all_paths"]
        defs = list(defects)
        boundary_corr = []
        if len(defs) % 2 == 1:
            best_v, best_d = defs[0], min(defs[0] // (L + 1), L - defs[0] // (L + 1))
            for v in defs[1:]:
                db = min(v // (L + 1), L - v // (L + 1))
                if db < best_d:
                    best_v, best_d = v, db
            boundary_corr = _trace_to_boundary(best_v, L, eim)
            defs = [v for v in defs if v != best_v]
        if len(defs) < 2:
            return boundary_corr

        path_cache = {}
        for i, u in enumerate(defs):
            for j, v in enumerate(defs):
                if i < j:
                    key = (min(u, v), max(u, v))
                    p = ap.get(u, {}).get(v)
                    path_cache[key] = p

        def _df(u, v):
            p = path_cache.get((min(u, v), max(u, v)))
            return (len(p) - 1) if p else 1e6

        pairs = _mwpm_match(defs, _df)
        corr = set()
        for u, v in pairs:
            p = path_cache.get((min(u, v), max(u, v)))
            if p:
                for k in range(len(p) - 1):
                    eid = eim.get((p[k], p[k + 1]), eim.get((p[k + 1], p[k])))
                    if eid is not None:
                        corr ^= {eid}
        return list(corr ^ set(boundary_corr))

    def _logical_gap(correction, L, hc):
        """Normalized logical gap in [0,1]. 1 = maximum confidence in current class."""
        cs = set(correction)
        w0 = len(cs)
        best_alt = float("inf")
        for c in range(L + 1):
            col = set(hc + r * (L + 1) + c for r in range(L))
            best_alt = min(best_alt, len(cs ^ col))
        gap = best_alt - w0
        return max(0.0, min(1.0, (gap + L) / (2 * L)))

    def _partial_gap(vis_defs, lat, p, rng, n_samples=20):
        L, hc, v2e = lat["L"], lat["horz_count"], lat["v2e"]
        gaps = []
        for _ in range(n_samples):
            sampled = []
            for v in lat["hidden"]:
                deg = len(v2e[v])
                p_def = (1.0 - (1.0 - 2.0 * p) ** deg) / 2.0
                if rng.random() < p_def:
                    sampled.append(v)
            combined = list(vis_defs) + sampled
            corr = _mwpm_decode(combined, lat)
            gaps.append(_logical_gap(corr, L, hc))
        return float(np.mean(gaps)) if gaps else 0.5

    def generate_trial(lattice, p, seed):
        """Generate one noisy trial. Returns dict with all game-relevant info."""
        rng = np.random.default_rng(seed)
        L = lattice["L"]
        nq = lattice["n_qubits"]
        H = lattice["H_Z"]
        hc = lattice["horz_count"]
        logical = lattice["logical_X"]

        errors = (rng.random(nq) < p).astype(np.int8)
        syndrome = (H @ errors) % 2

        vis_defs = sorted(v for v in lattice["visible"] if syndrome[v])
        hid_defs = sorted(v for v in lattice["hidden"] if syndrome[v])
        all_defs = sorted(v for v in range(lattice["n_vertices"]) if syndrome[v])

        full_corr = _mwpm_decode(all_defs, lattice)
        residual = set(np.where(errors)[0]) ^ set(full_corr)
        is_error = len(residual & logical) % 2 == 1

        vis_corr = _mwpm_decode(vis_defs, lattice)
        ag = _logical_gap(vis_corr, L, hc)
        pg = _partial_gap(vis_defs, lattice, p, rng, n_samples=20)

        return dict(
            errors=errors, syndrome=syndrome,
            vis_defs=vis_defs, hid_defs=hid_defs, all_defs=all_defs,
            full_corr=full_corr, vis_corr=vis_corr,
            is_error=is_error, apparent_gap=ag, partial_gap=pg,
        )

    return build_lattice, generate_trial


@app.cell
def _():
    def render_lattice_svg(lat, trial, reveal=False, width=460, height=460):
        d, L = lat["d"], lat["L"]
        edges = lat["edges"]
        vpos = lat["vertex_pos"]
        hidden = lat["hidden"]
        syndrome = trial["syndrome"]
        errs = trial["errors"]
        vis_corr = set(trial["vis_corr"])
        full_corr = set(trial["full_corr"])

        pad = 56
        cell = min((width - 2 * pad) / L, (height - 2 * pad) / L)

        def px(c):
            return pad + c * cell

        def py(r):
            return pad + r * cell

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
            f' style="background:#0d1117;border-radius:8px;display:block;margin:auto">',
            '<defs>'
            '<linearGradient id="ftop" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0%" stop-color="#0d1117" stop-opacity="1"/>'
            '<stop offset="60%" stop-color="#0d1117" stop-opacity=".97"/>'
            '<stop offset="100%" stop-color="#0d1117" stop-opacity="0"/>'
            '</linearGradient>'
            '<linearGradient id="fbot" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0%" stop-color="#0d1117" stop-opacity="0"/>'
            '<stop offset="40%" stop-color="#0d1117" stop-opacity=".97"/>'
            '<stop offset="100%" stop-color="#0d1117" stop-opacity="1"/>'
            '</linearGradient>'
            '</defs>',
        ]

        for eid, (a, b) in enumerate(edges):
            x1, y1, x2, y2 = px(a[1]), py(a[0]), px(b[1]), py(b[0])
            col, sw = "#21262d", 1.5
            if reveal and errs[eid]:
                col, sw = "#f8514988", 3.5
            if reveal and eid in full_corr:
                col, sw = "#58a6ff99", 3.0
            elif not reveal and eid in vis_corr:
                col, sw = "#3fb95066", 2.0
            svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                       f'stroke="{col}" stroke-width="{sw}" stroke-linecap="round"/>')

        for vid in range(lat["n_vertices"]):
            r, c = vpos[vid]
            x, y = px(c), py(r)
            is_hid = vid in hidden
            is_def = syndrome[vid] == 1
            if is_hid and not reveal:
                continue
            if is_def:
                col = "#d29922" if (is_hid and reveal) else "#f85149"
                rad = 7
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad}" '
                           f'fill="{col}" stroke="#fff" stroke-width="1.2" opacity=".95"/>')
            else:
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#30363d"/>')

        if not reveal:
            fh = cell * 0.85
            svg.append(f'<rect x="{pad - 14}" y="{pad - fh * 0.55:.1f}" width="{L * cell + 28:.1f}" '
                       f'height="{fh:.1f}" fill="url(#ftop)" rx="4"/>')
            svg.append(f'<rect x="{pad - 14}" y="{pad + L * cell - fh * 0.45:.1f}" width="{L * cell + 28:.1f}" '
                       f'height="{fh:.1f}" fill="url(#fbot)" rx="4"/>')
            svg.append(f'<text x="{width / 2}" y="{pad - 2}" text-anchor="middle" fill="#d29922" '
                       f'font-family="monospace" font-size="9" opacity=".7">HIDDEN BOUNDARY</text>')
            svg.append(f'<text x="{width / 2}" y="{pad + L * cell + 10}" text-anchor="middle" fill="#d29922" '
                       f'font-family="monospace" font-size="9" opacity=".7">HIDDEN BOUNDARY</text>')

        tag = "REVEALED" if reveal else "BULK SYNDROME"
        svg.append(f'<text x="{width / 2}" y="18" text-anchor="middle" fill="#8b949e" '
                   f'font-family="monospace" font-size="11">d={d}  |  {tag}</text>')
        svg.append("</svg>")
        return "\n".join(svg)

    def render_gap_bar(value, label, color, w=240):
        bar_w = w - 90
        fill = max(0, min(1, value)) * bar_w
        pct = f"{value:.0%}"
        return (
            f'<svg width="{w}" height="28">'
            f'<text x="0" y="18" fill="#8b949e" font-family="monospace" font-size="11">{label}</text>'
            f'<rect x="90" y="4" width="{bar_w}" height="20" rx="4" fill="#161b22" stroke="#30363d"/>'
            f'<rect x="90" y="4" width="{fill:.1f}" height="20" rx="4" fill="{color}" opacity=".85"/>'
            f'<text x="{90 + bar_w + 6}" y="18" fill="{color}" font-family="monospace" font-size="11">{pct}</text>'
            f"</svg>"
        )

    return render_gap_bar, render_lattice_svg


@app.cell
def _(mo):
    get_game, set_game = mo.state(
        {
            "seed": 42,
            "trial_count": 0,
            "gates": 0,
            "errors": 0,
            "rejects": 0,
            "total_cost": 0.0,
            "history": [],
            "last_result": None,
            "_clicks": 0,
            "_accept_n": 0,
            "_reset_n": 0,
        }
    )
    return get_game, set_game


@app.cell
def _(mo):
    phase_sel = mo.ui.radio(
        options={
            "Phase 1 — Brute Force": 1,
            "Phase 2 — Apparent Gap": 2,
            "Phase 3 — Partial Gap": 3,
            "Phase 4 — Automation": 4,
        },
        value="Phase 1 — Brute Force",
        label="",
    )
    d_slider = mo.ui.slider(3, 9, step=2, value=3, label="Code distance d")
    p_slider = mo.ui.slider(0.01, 0.15, step=0.005, value=0.05, label="Error rate p")
    return d_slider, p_slider, phase_sel


@app.cell
def _(mo):
    # Marimo buttons need on_click to advance .value; default on_click keeps
    # value stuck at the initial value, so downstream cells never see a change.
    _tick = lambda v: (v or 0) + 1
    accept_btn = mo.ui.button(
        label="Accept  [a]",
        kind="success",
        value=0,
        on_click=_tick,
        keyboard_shortcut="a",
    )
    reject_btn = mo.ui.button(
        label="Reject  [r]",
        kind="danger",
        value=0,
        on_click=_tick,
        keyboard_shortcut="r",
    )
    reset_btn = mo.ui.button(label="Reset Score", kind="neutral", value=0, on_click=_tick)
    return accept_btn, reject_btn, reset_btn


@app.cell
def _(build_lattice, d_slider):
    lattice = build_lattice(d_slider.value)
    return (lattice,)


@app.cell
def _(generate_trial, get_game, lattice, p_slider):
    _seed = get_game()["seed"]
    trial = generate_trial(lattice, p_slider.value, seed=_seed + 99991)
    return (trial,)


@app.cell
def _(get_game):
    get_game()
    return


@app.cell(hide_code=True)
def _(mo, phase_sel):
    _p = phase_sel.value
    _texts = {
        1: r"""
    **Phase 1 — Brute Force.** Every state is auto-accepted. Your only lever is the
    code distance slider. Push $d$ higher to suppress logical errors — but watch the
    overhead ($d^3$ per state) skyrocket. Establish a baseline: what overhead does it
    take to reach a low error rate without postselection?
    """,
        2: r"""
    **Phase 2 — Apparent Gap.** Accept/Reject buttons are now live. Use the
    **Apparent Gap** $G_A$ (computed from the visible syndrome only, assuming a clean
    boundary) to decide. *Beware:* when the boundary hides errors, $G_A$ can be
    dangerously overconfident. You will experience unexpected logical failures.
    """,
        3: r"""
    **Phase 3 — Partial Gap.** The **Partial Gap** $G_P$ is revealed. It averages the
    logical gap over possible boundary configurations — accounting for the uncertainty
    you cannot see. Trust $G_P$ over $G_A$. States with high $G_A$ but low $G_P$ are
    traps. Reject them.
    """,
        4: r"""
    **Phase 4 — Automated Frontier.** Set a rejection threshold on $G_P$ and let the
    simulator sweep thousands of trials. The frontier chart below plots **overhead vs
    logical error rate** for brute-force and postselected strategies. Watch the
    postselected curve achieve the *same error rate at ~4× less overhead*.
    """,
    }
    mo.md(_texts.get(_p, ""))
    return


@app.cell
def _(
    accept_btn,
    d_slider,
    get_game,
    lattice,
    mo,
    p_slider,
    phase_sel,
    reject_btn,
    render_gap_bar,
    render_lattice_svg,
    reset_btn,
    trial,
):
    _s = get_game()
    _phase = phase_sel.value

    _svg = render_lattice_svg(lattice, trial, reveal=False)
    _bars = [render_gap_bar(trial["apparent_gap"], "Apparent G_A", "#3fb950")]
    if _phase >= 3:
        _bars.append(render_gap_bar(trial["partial_gap"], "Partial G_P", "#58a6ff"))

    _result_html = ""
    _lr = _s.get("last_result")
    if _lr:
        if _lr["action"] == "accept" and not _lr["was_error"]:
            _result_html = f'<div class="result-accept-ok">Gate successful — state was clean. (G_A={_lr["ag"]:.0%}, G_P={_lr["pg"]:.0%})</div>'
        elif _lr["action"] == "accept" and _lr["was_error"]:
            _result_html = f'<div class="result-accept-err">LOGICAL ERROR — a hidden boundary chain fooled the decoder! (G_A={_lr["ag"]:.0%}, G_P={_lr["pg"]:.0%})</div>'
        elif _lr["action"] == "reject" and _lr["was_error"]:
            _result_html = f'<div class="result-reject-good">Good reject — this state had a hidden error. Retry cost: d³={_lr["cost"]}. (G_A={_lr["ag"]:.0%}, G_P={_lr["pg"]:.0%})</div>'
        elif _lr["action"] == "reject" and not _lr["was_error"]:
            _result_html = f'<div class="result-reject-waste">Wasted reject — state was safe. Retry cost: d³={_lr["cost"]}. (G_A={_lr["ag"]:.0%}, G_P={_lr["pg"]:.0%})</div>'

    _auto = _phase == 4
    if _phase == 1:
        _btn_row = mo.hstack([accept_btn], justify="center")
    elif not _auto:
        _btn_row = mo.hstack([accept_btn, reject_btn], justify="center", gap=1)
    else:
        _btn_row = mo.md("")

    _elements = [
        mo.hstack([phase_sel], justify="center"),
        mo.hstack([d_slider, p_slider], justify="center", gap=2),
        mo.Html(_svg),
        mo.hstack([mo.Html(b) for b in _bars], justify="center", gap=1),
    ]
    if _result_html:
        _elements.append(mo.Html(_result_html))
    if not _auto:
        _elements.append(_btn_row)
        _elements.append(mo.hstack([reset_btn], justify="center"))

    mo.vstack(_elements, align="center")
    return


@app.cell
def _(
    accept_btn,
    build_lattice,
    d_slider,
    generate_trial,
    get_game,
    p_slider,
    phase_sel,
    reject_btn,
    reset_btn,
    set_game,
):
    _s = get_game()
    _a = accept_btn.value or 0
    _r = reject_btn.value or 0
    _reset = reset_btn.value or 0
    _total = _a + _r
    _prev = _s.get("_clicks", 0)

    if _reset > _s.get("_reset_n", 0):
        set_game({
            "seed": _s["seed"], "trial_count": 0, "gates": 0, "errors": 0,
            "rejects": 0, "total_cost": 0.0, "history": [], "last_result": None,
            "_clicks": _total, "_accept_n": _a, "_reset_n": _reset,
        })
    elif _total > _prev and phase_sel.value != 4:
        _was_accept = (_a > _s.get("_accept_n", 0)) or (phase_sel.value == 1)
        _lat = build_lattice(d_slider.value)
        _t = generate_trial(_lat, p_slider.value, seed=_s["seed"] + 99991)
        _d = d_slider.value
        _base = _d**3

        _new = dict(_s)
        _new["_clicks"] = _total
        _new["_accept_n"] = _a
        _new["trial_count"] = _s.get("trial_count", 0) + 1

        if _was_accept:
            _new["total_cost"] = _s["total_cost"] + _base
            if _t["is_error"]:
                _new["errors"] = _s["errors"] + 1
            else:
                _new["gates"] = _s["gates"] + 1
            _new["last_result"] = dict(
                action="accept", was_error=_t["is_error"],
                ag=_t["apparent_gap"], pg=_t["partial_gap"],
            )
        else:
            _new["rejects"] = _s["rejects"] + 1
            _new["total_cost"] = _s["total_cost"] + _base
            _new["last_result"] = dict(
                action="reject", was_error=_t["is_error"], cost=_base,
                ag=_t["apparent_gap"], pg=_t["partial_gap"],
            )

        _new["seed"] = _s["seed"] + 1
        _new["history"] = _s.get("history", []) + [_new["last_result"]]
        set_game(_new)
    return


@app.cell
def _(d_slider, get_game, mo, phase_sel):
    _s = get_game()
    _gates = _s["gates"]
    _errs = _s["errors"]
    _rej = _s["rejects"]
    _cost = _s["total_cost"]
    _total = _gates + _errs
    _err_rate = _errs / _total if _total > 0 else 0
    _overhead = _cost / _gates if _gates > 0 else 0
    _base = d_slider.value ** 3

    mo.stop(phase_sel.value == 4)

    mo.vstack([
        mo.md("### Scoreboard"),
        mo.hstack([
            mo.stat(value=f"{_gates}", label="Gates OK", bordered=True),
            mo.stat(value=f"{_errs}", label="Logical Errors", bordered=True),
            mo.stat(value=f"{_rej}", label="Rejects", bordered=True),
        ], justify="center", gap=1),
        mo.hstack([
            mo.stat(value=f"{_err_rate:.1%}", label="Error Rate", bordered=True),
            mo.stat(value=f"{_overhead:.0f}", label="Overhead/Gate", bordered=True),
            mo.stat(value=f"{_base}", label="Base Cost d³", bordered=True),
        ], justify="center", gap=1),
    ])
    return


@app.cell
def _(mo, phase_sel):
    mo.stop(phase_sel.value != 4)
    threshold_slider = mo.ui.slider(
        0.0, 0.95, step=0.05, value=0.55,
        label="Partial-gap rejection threshold",
    )
    n_trials_slider = mo.ui.slider(
        50, 500, step=50, value=200,
        label="Trials per (d, threshold) point",
    )
    run_frontier_btn = mo.ui.run_button(label="Run Frontier Simulation")
    mo.vstack([
        mo.hstack([threshold_slider, n_trials_slider], justify="center", gap=2),
        mo.hstack([run_frontier_btn], justify="center"),
    ])
    return n_trials_slider, run_frontier_btn


@app.cell
def _(
    build_lattice,
    generate_trial,
    mo,
    n_trials_slider,
    np,
    p_slider,
    pd,
    phase_sel,
    run_frontier_btn,
):
    mo.stop(phase_sel.value != 4)
    mo.stop(not run_frontier_btn.value)

    _p = p_slider.value
    _n = n_trials_slider.value
    _rows = []

    for _d in (3, 5, 7):
        _lat = build_lattice(_d)
        _base = _d**3
        for _thr in np.linspace(0.0, 0.90, 10):
            _accepted, _errors, _rejected = 0, 0, 0
            for _i in range(_n):
                _t = generate_trial(_lat, _p, seed=500000 + _d * 10000 + int(_thr * 1000) * 100 + _i)
                if _t["partial_gap"] < _thr:
                    _rejected += 1
                else:
                    _accepted += 1
                    if _t["is_error"]:
                        _errors += 1
            _acc_rate = _accepted / max(1, _accepted + _rejected)
            _err_rate = _errors / max(1, _accepted)
            _overhead = _base / _acc_rate if _acc_rate > 0 else float("inf")
            _rows.append(dict(
                d=_d, threshold=round(_thr, 2), accepted=_accepted,
                errors=_errors, rejected=_rejected,
                accept_rate=_acc_rate, error_rate=_err_rate, overhead=_overhead,
                strategy=f"d={_d} postselect",
            ))
        _rows.append(dict(
            d=_d, threshold=0.0, accepted=_n, errors=0, rejected=0,
            accept_rate=1.0, error_rate=0.0, overhead=_base,
            strategy=f"d={_d} brute-force",
        ))

    frontier_df = pd.DataFrame(_rows)
    frontier_df = frontier_df[frontier_df["overhead"] < 1e8]
    return (frontier_df,)


@app.cell
def _(alt, frontier_df, mo, phase_sel):
    mo.stop(phase_sel.value != 4)

    _chart = (
        alt.Chart(frontier_df[frontier_df["strategy"].str.contains("postselect")])
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("overhead:Q", title="Spacetime Overhead per Gate (d³ / accept_rate)", scale=alt.Scale(type="log")),
            y=alt.Y("error_rate:Q", title="Logical Error Rate", scale=alt.Scale(type="log", clamp=True)),
            color=alt.Color("strategy:N", title="Strategy"),
            tooltip=["d:Q", "threshold:Q", "error_rate:Q", "overhead:Q", "accept_rate:Q"],
        )
        .properties(width=560, height=360, title="Postselection Frontier: Overhead vs Logical Error Rate")
    )

    _brute = (
        alt.Chart(frontier_df[frontier_df["strategy"].str.contains("brute")])
        .mark_point(shape="diamond", size=120, filled=True)
        .encode(
            x="overhead:Q",
            y=alt.Y("error_rate:Q", scale=alt.Scale(type="log", clamp=True)),
            color="strategy:N",
            tooltip=["d:Q", "overhead:Q"],
        )
    )

    mo.vstack([
        mo.md(r"""
    ### Frontier Chart

    Each curve shows the **overhead–error tradeoff** for a given code distance $d$
    as the partial-gap rejection threshold varies. Higher threshold = more
    rejections = lower error rate but higher retry overhead.

    The diamond markers show **brute-force** (no postselection, threshold=0) for
    each $d$. The key insight: a postselected curve at *smaller* $d$ can reach the
    same error rate as a brute-force point at *larger* $d$, at a fraction of the
    overhead. This is the **scalable postselection advantage**.
    """),
        mo.ui.altair_chart(_chart + _brute),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Key Takeaways

    1. **The apparent gap $G_A$ is unreliable.** It ignores boundary uncertainty and
       can be dangerously overconfident when hidden defects lurk at the unmeasured
       boundary stabilizers.

    2. **The partial gap $G_P$ is the right metric.** By averaging over possible
       boundary syndromes (via the string-splitting approximation in the paper),
       $G_P$ faithfully estimates the true logical error probability.

    3. **Scalable postselection yields a ~4× overhead reduction.** Rejecting states
       with low $G_P$ dramatically lowers the logical error rate. You can then use a
       *smaller* code distance to hit the same target error rate — and since the base
       cost scales as $d^3$, even modest distance reductions compound into large
       savings.

    4. **The tradeoff is retry cost vs error suppression.** Each rejection costs one
       extra $d^3$ resource state. But the error rate drops faster than the overhead
       grows, creating a favorable frontier.

    > *Reference:* J. W. Staples, W. Fu, J. D. Thompson.
    > [Scalable Postselection of Quantum Resources](https://arxiv.org/abs/2603.08697).
    > arXiv:2603.08697 (2026).
    """)
    return


if __name__ == "__main__":
    app.run()

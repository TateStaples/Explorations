import marimo

__generated_with = "0.20.2"
app = marimo.App(
    width="medium",
)


@app.cell
def _():
    import marimo as mo
    import altair as alt

    alt.data_transformers.disable_max_rows()

    import numpy as np
    import pandas as pd
    import scipy.optimize as sco

    def chart_no_config(chart):
        """Strip top-level-only keys so chart can be used inside alt.vconcat/hconcat (Altair v6)."""
        spec = chart.to_dict(validate=False)

        _TOPLEVEL_ONLY = {"config", "$schema", "background", "autosize", "padding", "datasets"}

        def _strip(s):
            if isinstance(s, dict):
                s = dict(s)
                for k in _TOPLEVEL_ONLY:
                    s.pop(k, None)
                for k in ("hconcat", "vconcat", "layer"):
                    if k in s and isinstance(s[k], list):
                        s[k] = [_strip(x) for x in s[k]]
                return s
            return s

        spec = _strip(spec)
        if "hconcat" in spec:
            return alt.HConcatChart.from_dict(spec)
        if "vconcat" in spec:
            return alt.VConcatChart.from_dict(spec)
        if "layer" in spec:
            return alt.LayerChart.from_dict(spec)
        return alt.Chart.from_dict(spec)

    return alt, chart_no_config, mo, np, pd, sco


@app.cell
def _(alt, np, pd):
    # Minimal repro: single Altair chart to verify rendering
    _df = pd.DataFrame({"x": np.arange(5), "y": np.arange(5) ** 2})
    minimal_chart = (
        alt.Chart(_df)
        .mark_point(size=80)
        .encode(x="x:Q", y="y:Q")
        .properties(width=200, height=160, title="Minimal chart (repro)")
    )
    return (minimal_chart,)


@app.cell
def _(mo):
    # Global simulation parameters
    n_assets_ui = mo.ui.slider(
        start=3, stop=10, step=1, value=5, label="Number of Assets"
    )
    base_corr_ui = mo.ui.slider(
        start=0.0, stop=0.9, step=0.1, value=0.3, label="Base Correlation"
    )
    market_vol_ui = mo.ui.slider(
        start=0.1, stop=0.4, step=0.05, value=0.15, label="Market Volatility"
    )

    global_controls = mo.vstack(
        [
            mo.md(
                "### Market Environment Settings\nAdjust these parameters to change the simulated universe. All subsequent models will update automatically."
            ),
            mo.hstack([n_assets_ui, base_corr_ui, market_vol_ui]),
        ]
    )
    return base_corr_ui, global_controls, market_vol_ui, n_assets_ui


@app.cell
def _(base_corr_ui, market_vol_ui, n_assets_ui, np, pd):
    # Generate simulated returns based on UI inputs
    np.random.seed(42)  # For reproducibility but with changing parameters
    _n_assets = n_assets_ui.value
    _base_corr = base_corr_ui.value
    _market_vol = market_vol_ui.value

    # Simulate means and vols
    _means = np.random.uniform(0.04, 0.12, _n_assets)
    _vols = np.random.uniform(_market_vol * 0.5, _market_vol * 1.5, _n_assets)

    # Generate correlation matrix
    _corr = np.full((_n_assets, _n_assets), _base_corr)
    # Add some random noise to correlations
    _noise = np.random.normal(0, 0.1, (_n_assets, _n_assets))
    _noise = (_noise + _noise.T) / 2  # Make symmetric
    _corr = np.clip(_corr + _noise, -0.99, 0.99)
    np.fill_diagonal(_corr, 1.0)

    # Ensure positive semi-definite (Higham method approach)
    _eigvals, _eigvecs = np.linalg.eigh(_corr)
    _eigvals[_eigvals < 0] = 1e-8
    _corr = _eigvecs @ np.diag(_eigvals) @ _eigvecs.T

    # Convert correlation to covariance
    _cov = np.outer(_vols, _vols) * _corr

    # Generate daily returns
    _n_days = 252 * 5  # 5 years
    _daily_means = _means / 252
    _daily_cov = _cov / 252

    _returns_data = np.random.multivariate_normal(_daily_means, _daily_cov, _n_days)
    asset_names = [f"Asset_{i + 1}" for i in range(_n_assets)]
    df_returns = pd.DataFrame(_returns_data, columns=asset_names)

    # Annualized parameters (empirical)
    annual_returns = df_returns.mean() * 252
    annual_cov = df_returns.cov() * 252

    # Correlation matrix for visualization (cov_ij / (sigma_i * sigma_j)); use ndarray so fill_diagonal works
    _cov_arr = np.asarray(annual_cov)
    _vols = np.sqrt(np.diag(_cov_arr))
    corr_matrix = _cov_arr / np.outer(_vols, _vols)
    np.fill_diagonal(corr_matrix, 1.0)

    return annual_cov, annual_returns, asset_names, df_returns, corr_matrix


@app.cell
def _(mo):
    mo.md(r"""
    # Portfolio Optimization

    **A Discovery-Driven Journey Through 5 Investment Frameworks**

    How do you allocate capital among a universe of assets to maximize return while minimizing risk? This notebook explores portfolio optimization progressively, ensuring you understand the *why* before drowning in the *how*.

    For each of the 5 approaches, we will:
    1. **Motivate the Intuition:** Why does this concept exist?
    2. **Explore the Tool:** Play with interactive parameters to build an intuitive grasp of the mechanics.
    3. **Define the Rigor:** Formalize the discoveries made in the tool with academic mathematics.

    ---
    ### Step 0: The Synthetic Market
    Before we can optimize, we need a sandbox. The sliders below generate a synthetic market universe using Monte Carlo simulation. By adjusting the number of assets, their average correlation, and market-wide volatility, you are building the historical dataset ($\Sigma$ covariance matrix and $\mu$ expected returns) that the subsequent 5 models will attempt to solve.
    
    *Try cranking up the volatility and watch how the models adapt below.*
    """)
    return


@app.cell
def _(chart_no_config, alt, annual_cov, annual_returns, asset_names, corr_matrix, mo, np, pd):
    # Step 0: Correlation heatmap and universe scatter
    _n = len(asset_names)
    _heatmap_data = []
    for _i in range(_n):
        for _j in range(_n):
            _heatmap_data.append(
                {
                    "Asset 1": asset_names[_i],
                    "Asset 2": asset_names[_j],
                    "Correlation": float(np.clip(corr_matrix[_i, _j], -1, 1)),
                }
            )
    df_corr = pd.DataFrame(_heatmap_data)

    _vols = np.sqrt(np.diag(annual_cov))
    df_universe = pd.DataFrame(
        {
            "Asset": asset_names,
            "Volatility": _vols,
            "Return": annual_returns.values,
        }
    )

    step0_corr_chart = (
        alt.Chart(df_corr)
        .mark_rect()
        .encode(
            x=alt.X("Asset 1:N", title=None, sort=asset_names),
            y=alt.Y("Asset 2:N", title=None, sort=asset_names),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(domain=[-1, 0, 1], range=["#b22222", "#f0f0f0", "#2271b2"]),
                title="Correlation",
            ),
            tooltip=["Asset 1:N", "Asset 2:N", alt.Tooltip("Correlation:Q", format=".2f")],
        )
        .properties(width=320, height=280, title="Correlation matrix Σ structure")
    )

    step0_universe_chart = (
        alt.Chart(df_universe)
        .mark_circle(size=120, stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("Volatility:Q", title="Volatility (risk)", axis=alt.Axis(format="%")),
            y=alt.Y("Return:Q", title="Expected return", axis=alt.Axis(format="%")),
            color=alt.Color("Asset:N", scale=alt.Scale(scheme="tableau10"), title="Asset"),
            tooltip=[
                "Asset:N",
                alt.Tooltip("Volatility:Q", format=".2%"),
                alt.Tooltip("Return:Q", format=".2%"),
            ],
        )
        .properties(width=320, height=280, title="Universe in return–risk space")
    )

    step0_combined_chart = alt.hconcat(
        chart_no_config(step0_corr_chart),
        chart_no_config(step0_universe_chart),
        spacing=40,
    )
    return df_corr, df_universe, step0_combined_chart, step0_corr_chart, step0_universe_chart


@app.cell
def _(global_controls):
    global_controls


@app.cell
def _(mo):
    mo.md("**Generated universe:**")
    return


@app.cell
def _(mo, step0_corr_chart, step0_universe_chart):
    mo.hstack([step0_corr_chart, step0_universe_chart])
    return


@app.cell
def _(alt, annual_returns, mo, np, pd):
    # Many Monte Carlo paths + adjacent log-normal terminal return PDF
    _mu = float(annual_returns.mean())
    _sigma = 0.20
    _n_paths = 200
    _n_steps = 252
    _rng = np.random.default_rng(99)
    _dt = 1.0 / _n_steps
    _lr = (_mu - 0.5 * _sigma ** 2) * _dt + _sigma * np.sqrt(_dt) * _rng.standard_normal((_n_paths, _n_steps))
    _prices = np.exp(np.cumsum(_lr, axis=1))
    _prices = np.column_stack([np.ones(_n_paths), _prices])

    # Subsample every 5 days to keep data size manageable
    _step = 5
    _days = np.arange(0, _n_steps + 1, _step)
    _rows = []
    for _i in range(_n_paths):
        for _k in _days:
            _rows.append({"Day": int(_k), "Price": float(_prices[_i, _k]), "Path": _i})
    df_mc_paths = pd.DataFrame(_rows)

    # Terminal log-returns + normal fit
    _terminal_lr = np.sum(_lr, axis=1)
    _bins = np.linspace(_terminal_lr.min(), _terminal_lr.max(), 30)
    _hist, _edges = np.histogram(_terminal_lr, bins=_bins, density=True)
    _mid = (_edges[:-1] + _edges[1:]) / 2
    df_mc_pdf = pd.DataFrame({"Log Return": _mid, "Density": _hist})
    _x_fit = np.linspace(_terminal_lr.min(), _terminal_lr.max(), 80)
    from scipy.stats import norm as _norm
    _fit_pdf = _norm.pdf(_x_fit, _terminal_lr.mean(), _terminal_lr.std())
    df_mc_fit = pd.DataFrame({"Log Return": _x_fit, "Density": _fit_pdf})

    _paths_chart = (
        alt.Chart(df_mc_paths)
        .mark_line(strokeWidth=0.8, opacity=0.4, color="#4a7cb8")
        .encode(
            x=alt.X("Day:Q", title="Trading Day"),
            y=alt.Y("Price:Q", title="Growth of $1", scale=alt.Scale(type="log")),
            detail="Path:N",
        )
        .properties(width=420, height=300, title=f"200 Monte Carlo price paths (log scale)")
    )
    _bar = (
        alt.Chart(df_mc_pdf)
        .mark_bar(opacity=0.6, color="#6a9bd8")
        .encode(
            x=alt.X("Log Return:Q", title="Terminal log-return"),
            y=alt.Y("Density:Q", title="Density"),
        )
    )
    _fit_line = (
        alt.Chart(df_mc_fit)
        .mark_line(color="#c0392b", strokeWidth=2.5)
        .encode(x="Log Return:Q", y="Density:Q")
    )
    _pdf_chart = (
        (_bar + _fit_line)
        .properties(width=280, height=300, title="Terminal log-return ≈ Normal (red = fitted)")
    )
    step0_mc_display = mo.hstack([_paths_chart, _pdf_chart])
    return df_mc_paths, df_mc_pdf, step0_mc_display


@app.cell
def _(mo, step0_mc_display):
    mo.vstack([
        mo.md("**Log-normal price process:** Each path follows $dS = \\mu S\\,dt + \\sigma S\\,dW$. The terminal log-return is normally distributed (right panel) — that's the foundation for Black-Scholes and most portfolio models."),
        step0_mc_display,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Foundations of Financial Math

    Before diving into optimization, it's crucial to understand the fundamental concepts that these models rely on.

    ### 1. The Multi-Asset Portfolio
    When managing severe wealth, an investor rarely holds a single asset. A **multi-asset portfolio** distributes capital across various financial instruments (e.g., Apple stock, US Treasury Bonds, Gold) via a vector of weights $w$, where $\sum w_i = 1$. The true power of a portfolio lies in **diversification**: because assets do not move in perfect lockstep (they are not perfectly correlated), combining them can significantly reduce the overall volatility of the portfolio without perfectly proportionally reducing the expected return. This non-linear risk reduction is often referred to as the only "free lunch" in finance.

    ### 2. The Nature of Academic Financial Risk
    In modern portfolio theory, the concept of **"risk"** is formally defined as **volatility**—specifically, the standard deviation ($\sigma$) or variance ($\sigma^2$) of the asset's returns. 
    *   **Academic View:** Risk is a two-sided coin. Both unexpected massive gains *and* massive losses contribute to high volatility. The math tries to minimize *uncertainty* of the outcome.
    *   **Practical Context:** While retail investors define risk purely as "losing money" (drawdown), mathematical optimizers use the covariance matrix ($\Sigma$) to represent how all assets bounce around and interact with one another.

    ### 3. Simple Returns vs Log Returns
    In financial modeling, there is a distinct difference in how we calculate growth.
    *   **Simple Returns** ($R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$): Used for calculating the weighted return of a portfolio at a specific point in time because simple returns aggregate linearly across assets.
    *   **Log Returns** ($r_t = \ln(\frac{P_t}{P_{t-1}})$): Used extensively for modeling over time. They are time-additive (you can simply sum daily log returns to get the annual log return), symmetric, and most importantly, they are mathematically capable of being normally distributed. Simple returns cannot fall below -100%, breaking the assumption of a normal probability curve which optimization models deeply rely upon. 
    
    Throughout these approaches, these mathematical priors—Normal distributions, historical covariance representing true future risk, and continuous compounding—form the bedrock that all 5 optimization strategies balance on.
    
    ---
    """)
    return


@app.cell
def _(chart_no_config, alt, np, pd):
    def _simulate_paths(
        n_paths: int, n_steps: int, mu: float, sigma: float, seed: int
    ) -> tuple:
        rng = np.random.default_rng(seed)
        dt = 1.0 / n_steps
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(
            (n_paths, n_steps)
        )
        cumlog = np.cumsum(log_returns, axis=1)
        prices = np.exp(cumlog)
        prices = np.column_stack([np.ones(n_paths), prices])
        terminal_log_ret = cumlog[:, -1]
        return prices, terminal_log_ret

    # Random walks + terminal return PDF (Foundations)
    _n_paths_viz = 40
    _n_steps = 252
    _mu, _sigma = 0.08, 0.18
    _prices, _terminal_lr = _simulate_paths(
        _n_paths_viz, _n_steps, _mu, _sigma, seed=123
    )
    _path_rows = []
    for _i in range(_n_paths_viz):
        for _k in range(_n_steps + 1):
            _path_rows.append(
                {"Day": _k, "Price": _prices[_i, _k], "Path": f"Path_{_i + 1}"}
            )
    df_walks = pd.DataFrame(_path_rows)

    _n_large = 5000
    _, _term_lr_large = _simulate_paths(_n_large, _n_steps, _mu, _sigma, seed=456)
    _bin_edges = np.linspace(_term_lr_large.min(), _term_lr_large.max(), 35)
    _hist, _edges = np.histogram(_term_lr_large, bins=_bin_edges, density=True)
    _mid = (_edges[:-1] + _edges[1:]) / 2
    df_pdf = pd.DataFrame({"Return": _mid, "Density": _hist})
    _x_norm = np.linspace(_term_lr_large.min(), _term_lr_large.max(), 100)
    from scipy import stats as _scipy_stats

    _pdf_norm = _scipy_stats.norm.pdf(_x_norm, _term_lr_large.mean(), _term_lr_large.std())
    df_norm = pd.DataFrame({"Return": _x_norm, "Density": _pdf_norm, "Curve": "Normal"})

    foundations_walks_chart = (
        alt.Chart(df_walks)
        .mark_line(opacity=0.7, strokeWidth=1.5)
        .encode(
            x=alt.X("Day:Q", title="Time (days)"),
            y=alt.Y("Price:Q", title="Price (normalized)"),
            detail="Path:N",
            color=alt.value("#4a7cb8"),
        )
        .properties(width=340, height=240, title="Random price paths (log-normal)")
    )

    _bar = (
        alt.Chart(df_pdf)
        .mark_bar(opacity=0.6, color="#6a9bd8")
        .encode(
            x=alt.X("Return:Q", title="Terminal log-return"),
            y=alt.Y("Density:Q", title="Density"),
        )
    )
    _line_norm = (
        alt.Chart(df_norm)
        .mark_line(color="#c22", strokeWidth=2.5)
        .encode(x="Return:Q", y="Density:Q")
    )
    foundations_pdf_chart = (
        (_bar + _line_norm)
        .properties(width=340, height=240, title="Terminal return → approximately normal")
    )

    foundations_asset_movement = alt.hconcat(
        chart_no_config(foundations_walks_chart),
        chart_no_config(foundations_pdf_chart),
        spacing=30,
    )
    return (
        df_norm,
        df_pdf,
        df_walks,
        foundations_asset_movement,
        foundations_pdf_chart,
        foundations_walks_chart,
    )


@app.cell
def _(alt, df_returns, mo, np, pd):
    # Risk as volatility: two-asset return distributions (high-vol vs low-vol)
    _cols = list(df_returns.columns)
    _vols = df_returns.std()
    _high_vol_asset = _vols.idxmax()
    _low_vol_asset = _vols.idxmin()
    _r_high = df_returns[_high_vol_asset].values * 100
    _r_low = df_returns[_low_vol_asset].values * 100
    _mu_high = float(np.mean(_r_high))
    _mu_low = float(np.mean(_r_low))
    _sigma_high = float(np.std(_r_high))
    _sigma_low = float(np.std(_r_low))
    _bins = np.linspace(min(_r_high.min(), _r_low.min()), max(_r_high.max(), _r_low.max()), 30)
    _h1, _e1 = np.histogram(_r_high, bins=_bins, density=True)
    _h2, _e2 = np.histogram(_r_low, bins=_bins, density=True)
    _m1 = (_e1[:-1] + _e1[1:]) / 2
    _m2 = (_e2[:-1] + _e2[1:]) / 2
    df_risk_high = pd.DataFrame({"Daily Return %": _m1, "Density": _h1, "Asset": _high_vol_asset})
    df_risk_low = pd.DataFrame({"Daily Return %": _m2, "Density": _h2, "Asset": _low_vol_asset})
    df_risk_dist = pd.concat([df_risk_high, df_risk_low], ignore_index=True)

    # ±1σ reference lines for each asset
    _rule_data = pd.DataFrame([
        {"x": _mu_high - _sigma_high, "Asset": _high_vol_asset, "Label": f"−1σ ({_mu_high - _sigma_high:.2f}%)"},
        {"x": _mu_high + _sigma_high, "Asset": _high_vol_asset, "Label": f"+1σ ({_mu_high + _sigma_high:.2f}%)"},
        {"x": _mu_low - _sigma_low, "Asset": _low_vol_asset, "Label": f"−1σ ({_mu_low - _sigma_low:.2f}%)"},
        {"x": _mu_low + _sigma_low, "Asset": _low_vol_asset, "Label": f"+1σ ({_mu_low + _sigma_low:.2f}%)"},
    ])
    _sigma_rules = (
        alt.Chart(_rule_data)
        .mark_rule(strokeDash=[4, 3], strokeWidth=1.5, opacity=0.7)
        .encode(
            x=alt.X("x:Q"),
            color=alt.Color("Asset:N", scale=alt.Scale(range=["#e74c3c", "#3498db"]), legend=None),
            tooltip=["Asset:N", "Label:N"],
        )
    )

    _areas = (
        alt.Chart(df_risk_dist)
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X("Daily Return %:Q", title="Daily return (%)"),
            y=alt.Y("Density:Q", title="Density"),
            color=alt.Color("Asset:N", scale=alt.Scale(range=["#e74c3c", "#3498db"])),
            tooltip=["Asset:N", "Daily Return %:Q", "Density:Q"],
        )
    )

    foundations_risk_chart = (
        (_areas + _sigma_rules)
        .properties(
            width=560,
            height=240,
            title="Risk as σ: dashed lines mark ±1σ for each asset — wider band = more uncertainty",
        )
    )
    return df_risk_dist, foundations_risk_chart


@app.cell
def _(mo):
    mo.md(r"""
    ### Understanding the covariance matrix

    For two assets, the covariance matrix is $\Sigma = \begin{pmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{pmatrix}$, where $\sigma_i$ are volatilities and $\rho$ is correlation.

    **Try it yourself:** click on the canvas below to draw a cloud of points. The widget computes the live covariance ellipse (solid = 1σ, dashed = 2σ) and shows σ₁, σ₂, cov, and ρ in real time. Right-click a point to delete it.
    """)
    return


@app.cell
def _(mo):
    import anywidget
    import traitlets as _tl

    class ScatterWidget(anywidget.AnyWidget):
        _esm = r"""
        function pts_stats(pts) {
            if (pts.length < 2) return null;
            const n = pts.length;
            const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
            const mx = xs.reduce((a,b)=>a+b,0)/n, my = ys.reduce((a,b)=>a+b,0)/n;
            const cxx = xs.reduce((s,x)=>s+(x-mx)**2,0)/(n-1);
            const cyy = ys.reduce((s,y)=>s+(y-my)**2,0)/(n-1);
            const cxy = xs.reduce((s,x,i)=>s+(x-mx)*(ys[i]-my),0)/(n-1);
            const rho = (cxx>0&&cyy>0) ? cxy/Math.sqrt(cxx*cyy) : 0;
            return {n,mx,my,cxx,cyy,cxy,rho};
        }
        function eigen2(cxx,cxy,cyy) {
            const tr=cxx+cyy, det=cxx*cyy-cxy*cxy;
            const disc=Math.sqrt(Math.max(0,tr*tr/4-det));
            const l1=tr/2+disc, l2=Math.max(0,tr/2-disc);
            const angle=Math.abs(cxy)<1e-10 ? (cxx>=cyy?0:Math.PI/2) : Math.atan2(l1-cxx,cxy);
            return {l1,l2,angle};
        }
        function ellipsePath(cx,cy,a,b,theta,steps=64) {
            const cos=Math.cos(theta),sin=Math.sin(theta);
            const pts=Array.from({length:steps+1},(_,i)=>{
                const t=i/steps*2*Math.PI, ex=a*Math.cos(t), ey=b*Math.sin(t);
                return `${cx+ex*cos-ey*sin},${cy+ex*sin+ey*cos}`;
            });
            return `M ${pts.join(' L ')} Z`;
        }
        function render({model,el}) {
            const W=320,H=300,PAD=28;
            const IW=W-2*PAD, IH=H-2*PAD;
            const R=1.3;  // data range [-R,R]
            const toP=(d)=>PAD+(d+R)/(2*R)*IW;
            const toQ=(d)=>PAD+(-d+R)/(2*R)*IH;
            const fromP=(p)=>(p-PAD)/IW*2*R-R;
            const fromQ=(q)=>-((q-PAD)/IH*2*R-R);
            const scale=IW/(2*R);
            const NS='http://www.w3.org/2000/svg';

            el.innerHTML='';
            el.style.cssText='display:inline-flex;flex-direction:column;align-items:center;gap:6px;font-family:monospace;font-size:12px';

            const svg=document.createElementNS(NS,'svg');
            svg.setAttribute('width',W); svg.setAttribute('height',H);
            svg.style.cssText='cursor:crosshair;border-radius:8px;display:block';

            const statsDiv=document.createElement('div');
            statsDiv.style.cssText='padding:4px 10px;border-radius:4px;min-height:18px;text-align:center;font-size:12px';

            const btnRow=document.createElement('div');
            btnRow.style.cssText='display:flex;gap:8px;align-items:center';
            const clearBtn=document.createElement('button');
            clearBtn.textContent='Clear';
            clearBtn.style.cssText='padding:3px 12px;border-radius:4px;border:1px solid #ccc;cursor:pointer;font-size:11px';
            clearBtn.onclick=()=>{ model.set('points',[]); model.save_changes(); };
            btnRow.appendChild(clearBtn);
            btnRow.appendChild(statsDiv);

            el.appendChild(svg);
            el.appendChild(btnRow);

            function draw() {
                const pts=model.get('points')||[];
                while(svg.firstChild) svg.removeChild(svg.firstChild);

                // Background
                const bg=document.createElementNS(NS,'rect');
                bg.setAttribute('width',W); bg.setAttribute('height',H);
                bg.setAttribute('rx',8); bg.setAttribute('class','sw-bg');
                svg.appendChild(bg);

                // Grid lines
                for(let v=-1;v<=1;v+=0.5){
                    const xl=document.createElementNS(NS,'line');
                    const xp=toP(v);
                    xl.setAttribute('x1',xp); xl.setAttribute('x2',xp);
                    xl.setAttribute('y1',PAD); xl.setAttribute('y2',H-PAD);
                    xl.setAttribute('stroke',v===0?'#888':'#ddd');
                    xl.setAttribute('stroke-width',v===0?1.5:0.8);
                    svg.appendChild(xl);
                    const yl=document.createElementNS(NS,'line');
                    const yp=toQ(v);
                    yl.setAttribute('x1',PAD); yl.setAttribute('x2',W-PAD);
                    yl.setAttribute('y1',yp); yl.setAttribute('y2',yp);
                    yl.setAttribute('stroke',v===0?'#888':'#ddd');
                    yl.setAttribute('stroke-width',v===0?1.5:0.8);
                    svg.appendChild(yl);
                    // tick labels
                    if(v!==0){
                        const tx=document.createElementNS(NS,'text');
                        tx.setAttribute('x',xp); tx.setAttribute('y',H-PAD+14);
                        tx.setAttribute('text-anchor','middle');
                        tx.setAttribute('font-size',9); tx.setAttribute('class','sw-label');
                        tx.textContent=v.toFixed(1); svg.appendChild(tx);
                        const ty=document.createElementNS(NS,'text');
                        ty.setAttribute('x',PAD-4); ty.setAttribute('y',yp+4);
                        ty.setAttribute('text-anchor','end');
                        ty.setAttribute('font-size',9); ty.setAttribute('class','sw-label');
                        ty.textContent=v.toFixed(1); svg.appendChild(ty);
                    }
                }

                // Covariance ellipses
                const s=pts_stats(pts);
                if(s && s.n>=3){
                    const {l1,l2,angle}=eigen2(s.cxx,s.cxy,s.cyy);
                    const cx=toP(s.mx), cy=toQ(s.my);
                    for(const [sigma,fo,dash] of [[2,0.10,'5,3'],[1,0.22,'none']]){
                        const a=sigma*Math.sqrt(l1)*scale;
                        const b=sigma*Math.sqrt(l2)*scale;
                        const pe=document.createElementNS(NS,'path');
                        pe.setAttribute('d',ellipsePath(cx,cy,a,b,-angle));
                        pe.setAttribute('fill','#3498db'); pe.setAttribute('fill-opacity',fo);
                        pe.setAttribute('stroke','#2980b9'); pe.setAttribute('stroke-width',1.8);
                        if(dash!=='none') pe.setAttribute('stroke-dasharray',dash);
                        svg.appendChild(pe);
                    }
                    // Mean cross
                    for(const [x1,y1,x2,y2] of [[cx-7,cy,cx+7,cy],[cx,cy-7,cx,cy+7]]){
                        const ln=document.createElementNS(NS,'line');
                        ln.setAttribute('x1',x1); ln.setAttribute('y1',y1);
                        ln.setAttribute('x2',x2); ln.setAttribute('y2',y2);
                        ln.setAttribute('stroke','#c0392b'); ln.setAttribute('stroke-width',2);
                        svg.appendChild(ln);
                    }
                    const rho=s.rho;
                    const col=rho>0.2?'#27ae60':rho<-0.2?'#e74c3c':'#e67e22';
                    statsDiv.innerHTML=`σ<sub>x</sub>=<b>${Math.sqrt(s.cxx).toFixed(3)}</b> &nbsp; σ<sub>y</sub>=<b>${Math.sqrt(s.cyy).toFixed(3)}</b> &nbsp; cov=<b>${s.cxy.toFixed(3)}</b> &nbsp; <span style="color:${col}">ρ = <b>${rho.toFixed(3)}</b></span> &nbsp; n=${s.n}`;
                } else {
                    statsDiv.textContent=pts.length<3?`Click to add points — need ≥ 3 for ellipse (have ${pts.length})`:'';
                }

                // Data points
                for(const [dx,dy] of pts){
                    const c=document.createElementNS(NS,'circle');
                    c.setAttribute('cx',toP(dx)); c.setAttribute('cy',toQ(dy));
                    c.setAttribute('r',5); c.setAttribute('fill','#e74c3c');
                    c.setAttribute('stroke','white'); c.setAttribute('stroke-width',1.5);
                    svg.appendChild(c);
                }
            }

            svg.addEventListener('click',(e)=>{
                const r=svg.getBoundingClientRect();
                const x=fromP(e.clientX-r.left), y=fromQ(e.clientY-r.top);
                if(Math.abs(x)<=R && Math.abs(y)<=R){
                    model.set('points',[...(model.get('points')||[]),[x,y]]);
                    model.save_changes();
                }
            });
            svg.addEventListener('contextmenu',(e)=>{
                e.preventDefault();
                const r=svg.getBoundingClientRect();
                const mx=fromP(e.clientX-r.left), my=fromQ(e.clientY-r.top);
                const pts=model.get('points')||[];
                if(!pts.length) return;
                const i=pts.reduce((best,[px,py],j)=>
                    (px-mx)**2+(py-my)**2<(pts[best][0]-mx)**2+(pts[best][1]-my)**2?j:best,0);
                model.set('points',pts.filter((_,j)=>j!==i));
                model.save_changes();
            });

            model.on('change:points',draw);
            draw();
        }
        export default {render};
        """
        _css = r"""
        .sw-bg { fill: #fafafa; }
        .sw-label { fill: #aaa; }
        @media (prefers-color-scheme: dark) {
            .sw-bg { fill: #1e1e1e; }
            .sw-label { fill: #666; }
        }
        """
        points = _tl.List([]).tag(sync=True)

    scatter_widget = mo.ui.anywidget(ScatterWidget())
    return ScatterWidget, anywidget, scatter_widget


@app.cell
def _(mo, np, scatter_widget):
    _pts = scatter_widget.value.get("points", [])
    if len(_pts) >= 2:
        _arr = np.array(_pts)
        _cov = np.cov(_arr.T)
        _s1, _s2 = float(np.sqrt(_cov[0, 0])), float(np.sqrt(_cov[1, 1]))
        _c12 = float(_cov[0, 1])
        _rho = _c12 / (_s1 * _s2) if _s1 > 0 and _s2 > 0 else 0.0
        cov_panel = mo.md(rf"""
**Live covariance matrix** (n = {len(_pts)} points):

$$\Sigma = \begin{{pmatrix}} {_cov[0,0]:.4f} & {_c12:.4f} \\ {_c12:.4f} & {_cov[1,1]:.4f} \end{{pmatrix}}$$

| Stat | Value |
|---|---|
| σₓ | {_s1:.4f} |
| σᵧ | {_s2:.4f} |
| cov(x,y) | {_c12:.4f} |
| **ρ** | **{_rho:.4f}** |

*Try a diagonal cloud (ρ≈0), a diagonal cluster (ρ≈0.8), or an anti-diagonal (ρ≈−0.8).*
        """)
    else:
        cov_panel = mo.md(
            "**Draw points on the canvas** to see the live covariance matrix appear here.\n\n"
            "*Hint: try drawing a tight diagonal cluster for high ρ, "
            "or a blob for ρ≈0.*"
        )
    return (cov_panel,)


@app.cell
def _(mo):
    cov_sigma1_ui = mo.ui.slider(0.1, 0.5, step=0.05, value=0.2, label="σ₁")
    cov_sigma2_ui = mo.ui.slider(0.1, 0.5, step=0.05, value=0.25, label="σ₂")
    cov_rho_ui = mo.ui.slider(-0.9, 0.9, step=0.1, value=0.3, label="ρ (correlation)")
    cov_controls = mo.hstack([cov_sigma1_ui, cov_sigma2_ui, cov_rho_ui])
    return cov_controls, cov_rho_ui, cov_sigma1_ui, cov_sigma2_ui


@app.cell
def _(cov_rho_ui, cov_sigma1_ui, cov_sigma2_ui, np, pd, alt):
    from scipy.stats import multivariate_normal

    _s1, _s2, _r = cov_sigma1_ui.value, cov_sigma2_ui.value, cov_rho_ui.value
    _cov_2d = np.array(
        [[_s1**2, _r * _s1 * _s2], [_r * _s1 * _s2, _s2**2]]
    )
    _mean = np.zeros(2)
    _x = np.linspace(-0.6, 0.6, 50)
    _y = np.linspace(-0.6, 0.6, 50)
    _xx, _yy = np.meshgrid(_x, _y)
    _pos = np.column_stack([_xx.ravel(), _yy.ravel()])
    _pdf_vals = multivariate_normal.pdf(_pos, _mean, _cov_2d)
    _pdf_vals = _pdf_vals.reshape(_xx.shape)
    _rows = []
    for _i in range(len(_x)):
        for _j in range(len(_y)):
            _rows.append(
                {
                    "Payoff 1": _x[_i],
                    "Payoff 2": _y[_j],
                    "Density": float(_pdf_vals[_j, _i]),
                }
            )
    df_bivar = pd.DataFrame(_rows)

    foundations_cov_chart = (
        alt.Chart(df_bivar)
        .mark_rect()
        .encode(
            x=alt.X("Payoff 1:Q", bin=alt.Bin(maxbins=50), title="Payoff 1"),
            y=alt.Y("Payoff 2:Q", bin=alt.Bin(maxbins=50), title="Payoff 2"),
            color=alt.Color(
                "Density:Q",
                scale=alt.Scale(scheme="viridis"),
                title="Probability density",
            ),
            tooltip=[
                "Payoff 1:Q",
                "Payoff 2:Q",
                alt.Tooltip("Density:Q", format=".4f"),
            ],
        )
        .properties(
            width=380,
            height=340,
            title="Joint likelihood of payoff pairs (bivariate normal)",
        )
    )
    return df_bivar, foundations_cov_chart


@app.cell
def _(mo):
    mo.md(r"""
    ### Why diversification helps: law of large numbers

    As you add more (imperfectly correlated) assets to an equal-weight portfolio, portfolio volatility tends to fall. Below: portfolio volatility vs number of assets. Optional: change the base correlation to see how diversification benefit shrinks when assets move together.
    """)
    return


@app.cell
def _(mo):
    div_corr_ui = mo.ui.slider(0.0, 0.8, step=0.1, value=0.3, label="Pairwise correlation")
    return (div_corr_ui,)


@app.cell
def _(alt, div_corr_ui, np, pd):
    _rho = div_corr_ui.value
    _n_max = 25
    _base_vol = 0.20
    _vols_n = []
    _nn = []
    for _n in range(2, _n_max + 1):
        _corr = np.full((_n, _n), _rho)
        np.fill_diagonal(_corr, 1.0)
        _v = np.full(_n, _base_vol)
        _cov = np.outer(_v, _v) * _corr
        _w_ew = np.ones(_n) / _n
        _port_var = np.dot(_w_ew, np.dot(_cov, _w_ew))
        _vols_n.append(np.sqrt(_port_var))
        _nn.append(_n)
    df_div = pd.DataFrame({"n_assets": _nn, "Portfolio Volatility": _vols_n})

    foundations_div_chart = (
        alt.Chart(df_div)
        .mark_line(strokeWidth=3, point=True)
        .encode(
            x=alt.X("n_assets:Q", title="Number of assets (equal weight)"),
            y=alt.Y(
                "Portfolio Volatility:Q",
                title="Portfolio volatility",
                axis=alt.Axis(format="%"),
            ),
            tooltip=["n_assets:Q", alt.Tooltip("Portfolio Volatility:Q", format=".2%")],
        )
        .properties(
            width=450,
            height=260,
            title="Diversification: volatility decreases as n grows",
        )
    )
    return df_div, foundations_div_chart


@app.cell
def _(mo, scatter_widget, cov_panel, cov_controls, foundations_cov_chart):
    covariance_tabs = mo.ui.tabs({
        "✍️ Draw & Measure": mo.hstack([scatter_widget, cov_panel], gap="2rem"),
        "📐 Theoretical Model": mo.vstack([
            mo.md("Adjust σ₁, σ₂, and ρ to reshape the joint probability density. "
                  "This is the *theoretical* bivariate normal — compare the ellipse you drew with this heatmap."),
            cov_controls,
            foundations_cov_chart,
        ]),
    })
    return (covariance_tabs,)


@app.cell
def _(mo, foundations_risk_chart, covariance_tabs, div_corr_ui, foundations_div_chart, blend_w1_ui, blend_chart, asset_names):
    _a1, _a2 = asset_names[0], asset_names[1]
    mo.vstack([
        mo.md("---\n### 1 · Risk = Volatility"),
        mo.md(
            "**Why does volatility = risk?** Consider two assets both expected to return +8% per year. "
            "If one always returns *exactly* +8%, it is perfectly predictable — zero risk. "
            "If the other swings between −20% and +40% year-to-year, you genuinely don't know what you'll get — "
            "that uncertainty is risk, and its magnitude is captured by the **standard deviation σ**.\n\n"
            "MPT defines risk this way deliberately: σ measures *spread of outcomes*, not just downside. "
            "A surprisingly good year and a surprisingly bad year both widen the distribution, both increase σ. "
            "The optimizer penalizes *all* uncertainty, not just losses.\n\n"
            "The chart shows the daily-return distribution of the **most volatile** asset (red, wide) vs "
            "the **least volatile** (blue, narrow). The ±1σ bands (dashed lines) bracket ~68% of outcomes. "
            "Notice: the red asset's ±1σ band is far wider — it has more 'surprise potential' in both directions."
        ),
        foundations_risk_chart,

        mo.md("---\n### 2 · Covariance — how assets move together"),
        mo.md("The **covariance matrix** $\\Sigma$ encodes both individual volatilities (diagonal) "
              "and pairwise co-movement (off-diagonal). "
              "Use the tabs below to explore it empirically (draw your own cloud) or theoretically (tune the parameters)."),
        covariance_tabs,

        mo.md("---\n### 3 · Diversification — the *free lunch*"),
        mo.md(
            f"**Portfolio blend (chart below):** drag the slider to mix {_a1} and {_a2}. "
            "The blue curve traces every possible 2-asset portfolio in risk/return space. "
            "It *bows inward* — that bow is the diversification benefit. "
            "The red dashed line is the hypothetical with perfect correlation (ρ = 1, no benefit). "
            "Any point inside the bow gives you *lower* risk than a straight blend would imply.\n\n"
            "**Theory (vol-vs-N chart):** as more equal-weight assets are added, portfolio volatility falls. "
            "Drag the correlation slider to see how quickly the benefit erodes when assets move together."
        ),
        mo.vstack([blend_w1_ui, blend_chart]),
        mo.vstack([div_corr_ui, foundations_div_chart]),
    ])
    return


@app.cell
def _(mo):
    blend_w1_ui = mo.ui.slider(0.0, 1.0, step=0.05, value=0.5, label="Weight in Asset 1")
    return (blend_w1_ui,)


@app.cell
def _(alt, annual_cov, annual_returns, asset_names, blend_w1_ui, np, pd):
    _a1, _a2 = asset_names[0], asset_names[1]
    _r1 = float(annual_returns[_a1])
    _r2 = float(annual_returns[_a2])
    _v1 = float(np.sqrt(annual_cov.loc[_a1, _a1]))
    _v2 = float(np.sqrt(annual_cov.loc[_a2, _a2]))
    _cov12 = float(annual_cov.loc[_a1, _a2])

    _w1_grid = np.linspace(0, 1, 100)
    _blend_rows = []
    for _w in _w1_grid:
        _r = _w * _r1 + (1 - _w) * _r2
        _var = _w**2 * _v1**2 + (1 - _w)**2 * _v2**2 + 2 * _w * (1 - _w) * _cov12
        _blend_rows.append({"Weight in Asset 1": float(_w), "Return": float(_r), "Volatility": float(np.sqrt(max(_var, 0)))})
    df_blend = pd.DataFrame(_blend_rows)

    _w1_cur = blend_w1_ui.value
    _r_cur = _w1_cur * _r1 + (1 - _w1_cur) * _r2
    _var_cur = _w1_cur**2 * _v1**2 + (1 - _w1_cur)**2 * _v2**2 + 2 * _w1_cur * (1 - _w1_cur) * _cov12
    _v_cur = float(np.sqrt(max(_var_cur, 0)))
    df_blend_pt = pd.DataFrame([{"Volatility": _v_cur, "Return": float(_r_cur), "Label": f"w₁={_w1_cur:.0%}"}])
    df_no_div = pd.DataFrame([{"Volatility": _v1, "Return": _r1}, {"Volatility": _v2, "Return": _r2}])

    _curve = (
        alt.Chart(df_blend)
        .mark_line(color="#3498db", strokeWidth=2.5)
        .encode(
            x=alt.X("Volatility:Q", title="Volatility", axis=alt.Axis(format="%")),
            y=alt.Y("Return:Q", title="Expected Return", axis=alt.Axis(format="%")),
            order=alt.Order("Weight in Asset 1:Q"),
            tooltip=[
                alt.Tooltip("Weight in Asset 1:Q", format=".0%"),
                alt.Tooltip("Volatility:Q", format=".1%"),
                alt.Tooltip("Return:Q", format=".1%"),
            ],
        )
    )
    _no_div_line = (
        alt.Chart(df_no_div)
        .mark_line(color="#e74c3c", strokeDash=[6, 4], strokeWidth=2)
        .encode(x="Volatility:Q", y="Return:Q")
    )
    _cur_pt = (
        alt.Chart(df_blend_pt)
        .mark_point(filled=True, size=300, color="#e67e22")
        .encode(
            x="Volatility:Q",
            y="Return:Q",
            tooltip=["Label:N", alt.Tooltip("Volatility:Q", format=".1%"), alt.Tooltip("Return:Q", format=".1%")],
        )
    )
    _cur_text = _cur_pt.mark_text(align="left", dx=10, fontSize=12).encode(text="Label:N")

    blend_chart = (
        (_curve + _no_div_line + _cur_pt + _cur_text)
        .properties(width=540, height=320, title=f"2-Asset Blend: {_a1} + {_a2}")
    )
    return blend_chart, df_blend, df_blend_pt


@app.cell
def _(mo):
    mo.md(r"""
    <a id='approach1'></a>
    ## Approach 1: Mean-Variance Optimization (Markowitz)

    ### The Intuition: Diversification
    If you want high returns, why not just put 100% of your money into the single asset with the highest historical return?

    Because you cannot predict the future, and putting all your eggs in one basket exposes you to catastrophic single-point-of-failure risk. Harry Markowitz realized in 1952 that if you combine assets that *don't always move in the exact same direction* (uncorrelated assets), you can reduce the overall "bumpiness" (variance) of your portfolio without necessarily giving up proportional return. 

    The goal of Mean-Variance Optimization (MVO) is to find the perfect blend of weights that maximizes your expected return for whatever level of risk you are willing to tolerate.

    ### The Tool: The Efficient Frontier
    Below, the code simulates thousands of random, terrible portfolios (the faint blue dots). 
    
    The optimizer computes the **Efficient Frontier** (red dashed line) — the boundary where it is mathematically impossible to get a higher return without accepting more risk. Using the sliders, observe what happens when you change the Risk-Free rate.
    """)
    return


@app.cell
def _(mo):
    mvo_rf_ui = mo.ui.slider(
        start=0.0, stop=0.1, step=0.01, value=0.02, label="Risk-Free Rate"
    )
    mvo_samples_ui = mo.ui.slider(
        start=100, stop=5000, step=100, value=1000, label="Random Portfolios"
    )
    mvo_frontier_ui = mo.ui.slider(
        start=0, stop=49, step=1, value=24, label="Frontier Point (0 = Min Variance → 49 = Max Return)"
    )

    mvo_controls = mo.vstack(
        [
            mo.md("### Markowitz Optimization Controls"),
            mo.hstack([mvo_rf_ui, mvo_samples_ui]),
            mo.md("**Explore weights along the frontier:**"),
            mvo_frontier_ui,
        ]
    )
    return mvo_controls, mvo_frontier_ui, mvo_rf_ui, mvo_samples_ui


@app.cell
def _(annual_cov, annual_returns, mvo_rf_ui, mvo_samples_ui, np, pd, sco):
    _rf = mvo_rf_ui.value
    _n_portfolios = mvo_samples_ui.value
    _n_assets = len(annual_returns)

    # 1. Generate Random Portfolios
    _weights = np.random.random((_n_portfolios, _n_assets))
    _weights /= np.sum(_weights, axis=1)[:, np.newaxis]

    _port_returns = np.dot(_weights, annual_returns)
    _port_vols = np.array(
        [np.sqrt(np.dot(w.T, np.dot(annual_cov, w))) for w in _weights]
    )
    _port_sharpes = (_port_returns - _rf) / _port_vols

    df_random_ports = pd.DataFrame(
        {
            "Return": _port_returns,
            "Volatility": _port_vols,
            "Sharpe": _port_sharpes,
            "Portfolio": "Random",
        }
    )

    # helper functions for optimization
    def _get_ret_vol_sr(w):
        w = np.array(w)
        ret = np.sum(annual_returns * w)
        vol = np.sqrt(np.dot(w.T, np.dot(annual_cov, w)))
        if vol == 0:
            vol = 1e-8
        sr = (ret - _rf) / vol
        return np.array([ret, vol, sr])

    def _neg_sharpe(w):
        return -_get_ret_vol_sr(w)[2]

    def _volatility(w):
        return _get_ret_vol_sr(w)[1]

    _bounds = tuple((0.0, 1.0) for _ in range(_n_assets))
    _init_guess = _n_assets * [
        1.0 / _n_assets,
    ]

    # 2. Maximum Sharpe Portfolio (Tangency)
    _constraints_sum_1 = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    _opt_sharpe = sco.minimize(
        _neg_sharpe,
        _init_guess,
        method="SLSQP",
        bounds=_bounds,
        constraints=_constraints_sum_1,
    )
    _w_star = _opt_sharpe.x
    _max_sr_ret, _max_sr_vol, _max_sr = _get_ret_vol_sr(_w_star)

    # 3. Minimum Variance Portfolio (MVP)
    _opt_var = sco.minimize(
        _volatility,
        _init_guess,
        method="SLSQP",
        bounds=_bounds,
        constraints=_constraints_sum_1,
    )
    _w_min_var = _opt_var.x
    _min_var_ret, _min_var_vol, _min_var_sr = _get_ret_vol_sr(_w_min_var)

    # 4. Efficient Frontier Line (store weights at each frontier point)
    _target_returns = np.linspace(_min_var_ret, max(_port_returns) * 1.1, 50)
    _efficient_vols = []
    _efficient_weights = []

    for _tr in _target_returns:
        _tr_val = float(_tr)  # avoid closure capture bug
        _cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, tv=_tr_val: _get_ret_vol_sr(w)[0] - tv},
        )
        _res = sco.minimize(
            _volatility, _init_guess, method="SLSQP", bounds=_bounds, constraints=_cons
        )
        _efficient_vols.append(_res.fun)
        _efficient_weights.append(_res.x.copy())

    df_ef = pd.DataFrame(
        {
            "Return": _target_returns,
            "Volatility": _efficient_vols,
            "Portfolio": "Efficient Frontier",
        }
    )

    df_special = pd.DataFrame(
        {
            "Return": [_max_sr_ret, _min_var_ret],
            "Volatility": [_max_sr_vol, _min_var_vol],
            "Portfolio": ["Max Sharpe", "Min Variance"],
            "Sharpe": [_max_sr, _min_var_sr],
        }
    )

    mvo_results = {
        "random_ports": df_random_ports,
        "efficient_frontier": df_ef,
        "special_ports": df_special,
        "w_star": _w_star,
        "w_min_var": _w_min_var,
        "efficient_weights": _efficient_weights,
        "target_returns": _target_returns,
    }
    return mvo_results


@app.cell
def _(alt, mvo_results):
    _df_rand = mvo_results["random_ports"]
    _df_ef = mvo_results["efficient_frontier"]
    _df_spec = mvo_results["special_ports"]

    _scatter = (
        alt.Chart(_df_rand)
        .mark_circle(size=20, opacity=0.5)
        .encode(
            x=alt.X(
                "Volatility:Q",
                title="Volatility (Risk)",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "Return:Q",
                title="Expected Return",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "Sharpe:Q", scale=alt.Scale(scheme="viridis"), title="Sharpe Ratio"
            ),
            tooltip=[
                alt.Tooltip("Return:Q", format=".2%"),
                alt.Tooltip("Volatility:Q", format=".2%"),
                alt.Tooltip("Sharpe:Q", format=".2f"),
            ],
        )
    )

    _line = (
        alt.Chart(_df_ef)
        .mark_line(color="red", strokeDash=[5, 5])
        .encode(x="Volatility:Q", y="Return:Q")
    )

    _points = (
        alt.Chart(_df_spec)
        .mark_point(filled=True, size=200, color="red")
        .encode(
            x="Volatility:Q",
            y="Return:Q",
            shape=alt.Shape("Portfolio:N", legend=alt.Legend(title="Key Portfolios")),
            tooltip=[
                "Portfolio:N",
                alt.Tooltip("Return:Q", format=".2%"),
                alt.Tooltip("Volatility:Q", format=".2%"),
                alt.Tooltip("Sharpe:Q", format=".2f"),
            ],
        )
    )

    _text = _points.mark_text(
        align="left", baseline="middle", dx=10, fontSize=12, fontWeight="bold"
    ).encode(text="Portfolio:N")

    mvo_chart = (
        (_scatter + _line + _points + _text)
        .properties(width=700, height=400, title="Mean-Variance Efficient Frontier")
    )

    return mvo_chart


@app.cell
def _(alt, asset_names, mvo_frontier_ui, mvo_results, pd):
    _idx = mvo_frontier_ui.value
    _w = mvo_results["efficient_weights"][_idx]
    _ret = float(mvo_results["target_returns"][_idx])
    _vol = float(mvo_results["efficient_frontier"]["Volatility"].iloc[_idx])
    _rows = [{"Asset": _name, "Weight": float(_w[_i])} for _i, _name in enumerate(asset_names)]
    df_frontier_weights = pd.DataFrame(_rows)

    mvo_frontier_weights_chart = (
        alt.Chart(df_frontier_weights)
        .mark_bar(cornerRadiusEnd=4, color="#e74c3c")
        .encode(
            x=alt.X("Weight:Q", axis=alt.Axis(format="%"), title="Weight"),
            y=alt.Y("Asset:N", title=None, sort=alt.EncodingSortField("Weight", order="descending")),
            tooltip=["Asset:N", alt.Tooltip("Weight:Q", format=".2%")],
        )
        .properties(
            width=520,
            height=30 * len(asset_names) + 40,
            title=f"Weights at frontier point {_idx}  ·  target return {_ret:.2%}  ·  vol {_vol:.2%}",
        )
    )
    return df_frontier_weights, mvo_frontier_weights_chart


@app.cell
def _(mo, mvo_controls):
    mo.vstack([mo.md("---"), mvo_controls])
    return


@app.cell
def _(mo, mvo_chart, mvo_frontier_weights_chart):
    mo.vstack([mvo_chart, mvo_frontier_weights_chart])
    return


@app.cell
def _(mo):
    cal_rf_ui = mo.ui.slider(0.0, 0.08, step=0.005, value=0.02, label="Risk-Free Rate")
    return (cal_rf_ui,)


@app.cell
def _(alt, annual_cov, annual_returns, cal_rf_ui, mvo_results, np, pd, sco):
    _rf = cal_rf_ui.value
    _df_rand = mvo_results["random_ports"]
    _df_ef = mvo_results["efficient_frontier"]

    # Recompute tangency for the current cal_rf_ui rate
    _n = len(annual_returns)
    _bounds = tuple((0.0, 1.0) for _ in range(_n))
    _init = np.ones(_n) / _n

    def _neg_sharpe_cal(w: np.ndarray) -> float:
        w = np.array(w)
        r = float(np.sum(annual_returns * w))
        vol = float(np.sqrt(np.dot(w, np.dot(annual_cov, w))))
        return -(r - _rf) / vol if vol > 0 else 1e9

    _opt = sco.minimize(
        _neg_sharpe_cal, _init, method="SLSQP",
        bounds=_bounds, constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
    )
    _w_tan = _opt.x
    _tan_ret = float(np.sum(annual_returns * _w_tan))
    _tan_vol = float(np.sqrt(np.dot(_w_tan, np.dot(annual_cov, _w_tan))))
    _sharpe_slope = (_tan_ret - _rf) / _tan_vol if _tan_vol > 0 else 0

    _cal_vols = np.linspace(0, _tan_vol * 2.3, 100)
    _cal_rets = _rf + _sharpe_slope * _cal_vols
    df_cal = pd.DataFrame({"Volatility": _cal_vols, "Return": _cal_rets})
    df_rf_pt = pd.DataFrame([{"Volatility": 0.0, "Return": _rf, "Label": f"Risk-Free ({_rf:.1%})"}])
    df_tan_pt = pd.DataFrame([{"Volatility": _tan_vol, "Return": _tan_ret, "Label": "Tangency"}])

    _x_max = float(_df_rand["Volatility"].max()) * 1.1
    _bg = (
        alt.Chart(_df_rand)
        .mark_circle(size=18, opacity=0.25, color="#aaaaaa")
        .encode(
            x=alt.X("Volatility:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0.0, _x_max])),
            y=alt.Y("Return:Q", axis=alt.Axis(format="%")),
        )
    )
    _ef = (
        alt.Chart(_df_ef)
        .mark_line(color="#3498db", strokeWidth=2.5)
        .encode(x="Volatility:Q", y="Return:Q")
    )
    _cal = (
        alt.Chart(df_cal)
        .mark_line(color="#e74c3c", strokeWidth=2.5, strokeDash=[6, 4])
        .encode(x="Volatility:Q", y="Return:Q")
    )
    _rf_pt = (
        alt.Chart(df_rf_pt)
        .mark_point(filled=True, size=220, color="#27ae60")
        .encode(x="Volatility:Q", y="Return:Q", tooltip=["Label:N"])
    )
    _rf_txt = _rf_pt.mark_text(align="left", dx=8, dy=-8, fontSize=12).encode(text="Label:N")
    _tan_pt = (
        alt.Chart(df_tan_pt)
        .mark_point(filled=True, size=320, color="#e74c3c")
        .encode(x="Volatility:Q", y="Return:Q", tooltip=["Label:N"])
    )
    _tan_txt = _tan_pt.mark_text(align="left", dx=8, dy=-8, fontSize=12, fontWeight="bold").encode(text="Label:N")

    cal_chart = (
        (_bg + _ef + _cal + _rf_pt + _rf_txt + _tan_pt + _tan_txt)
        .properties(width=700, height=360, title="Capital Allocation Line (red): dominate the efficient frontier by mixing tangency + cash")
    )
    return cal_chart, df_cal


@app.cell
def _(cal_chart, cal_rf_ui, mo):
    mo.vstack([
        mo.md("### Capital Allocation Line (CAL)"),
        mo.md(
            "**What is mixing cash + tangency?** Every point on the red dashed line is a specific blend:\n\n"
            "- **Left endpoint (vol = 0):** 100% in the risk-free asset (cash/T-bills). You earn exactly $r_f$, zero uncertainty.\n"
            "- **Tangency point:** 100% in the tangency portfolio. Maximum Sharpe ratio — best return per unit of risk among all-risky portfolios.\n"
            "- **Between the two:** e.g. 60% tangency + 40% cash → you get 60% of the tangency's excess return but only 60% of its risk.\n"
            "- **Beyond tangency (rightward):** you *borrow* at $r_f$ to invest more than 100% in the tangency (leverage). Higher expected return, but amplified volatility.\n\n"
            "The CAL **dominates** the blue efficient frontier: for any given volatility level, a point on the dashed line "
            "always offers higher return than a point on the blue curve — because the straight line from $r_f$ through the "
            "tangency is steeper than the curved frontier everywhere else. "
            "Drag the slider to watch the CAL pivot as the risk-free rate changes."
        ),
        cal_rf_ui,
        cal_chart,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Weight comparison:** Max Sharpe (tangency) vs Min Variance portfolios.

    **How to read this chart:**
    * Each faint blue dot represents a randomly generated 100% invested portfolio.
    * The **<span style="color:red">red dashed line</span>** is the Efficient Frontier—the mathematically optimal edge where you cannot get more return without taking on more volatility.
    * The **Max Sharpe** portfolio is the "tangency portfolio" offering the best return per unit of risk, while the **Min Variance** portfolio is the absolute safest combination of these assets possible.
    * *Note that the lowest possible volatility is still significantly greater than 0%. True zero risk is only achievable with a cash-like "Risk-Free" asset, which is exactly what standard MVO excludes until calculating the Sharpe ratio.*

    ### The Rigor: The Markowitz Objective

    When we talk about "optimizing", we are functionally handing a computer an objective function to minimize (or maximize).

    Here, the solver seeks to build a weight vector $w$ that minimizes portfolio variance ($\frac{1}{2} w^T \Sigma w$) while pinning the expected return ($\mu^T w$) to a specific target value.

    $$ \min_w \frac{1}{2} w^T \Sigma w $$
    $$ \text{subject to:} $$
    $$ \mu^T w = \mu_{target} $$
    $$ \sum w_i = 1 $$

    **The Fatal Flaw:** The solver treats the historical inputs ($\mu$ and $\Sigma$) as absolute, infallible truths. Because historical data contains noise, the optimizer acts as an "error maximizer", aggressively allocating huge weights to whichever asset happened to have the highest historical return and lowest historical variance, resulting in concentrated, brittle portfolios.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <a id='approach2'></a>
    ## Approach 2: Capital Asset Pricing Model (CAPM)

    ### The Intuition: Relative Risk
    Markowitz told us we can reduce risk by building a diversified portfolio. CAPM takes this a step further and argues that *because* investors can diversify away asset-specific quirks (idiosyncratic risk), the market will only reward them for taking on risk that cannot be diversified away (systematic risk).
    
    In CAPM, we stop looking at an asset in isolation and start asking: "How does this asset behave relative to the broader market?" If an asset swings wildly but perfectly matches the market's swings, it has a high "Beta". If it moves independently, it has a low Beta.

    ### The Tool: The Security Market Line
    Adjust the expected market return and risk-free rate. Notice how the dashed line—the Security Market Line (SML)—tilts.
    
    The SML dictates exactly how much return an asset *should* generate given its Beta. If an asset (the colored dots) sits exactly on the line, it is correctly priced. 
    """)
    return


@app.cell
def _(mo):
    capm_rf_ui = mo.ui.slider(
        start=0.0, stop=0.1, step=0.01, value=0.02, label="Risk-Free Rate"
    )
    capm_rm_ui = mo.ui.slider(
        start=0.05, stop=0.20, step=0.01, value=0.08, label="Expected Market Return"
    )

    capm_controls = mo.vstack(
        [mo.md("### CAPM Parameters"), mo.hstack([capm_rf_ui, capm_rm_ui])]
    )
    return capm_controls, capm_rf_ui, capm_rm_ui


@app.cell
def _(
    annual_cov, annual_returns, asset_names, capm_rf_ui, capm_rm_ui, df_returns, np, pd
):
    _rf = capm_rf_ui.value
    _rm_target = capm_rm_ui.value
    _n_assets = len(annual_returns)

    # Define market as equal-weight portfolio
    _w_mkt = np.ones(_n_assets) / _n_assets
    _market_returns_ts = df_returns.dot(_w_mkt)
    _market_var = _market_returns_ts.var() * 252

    # Calculate Betas
    _betas = []
    for _i, _asset in enumerate(asset_names):
        _cov_im = df_returns[_asset].cov(_market_returns_ts) * 252
        _beta = _cov_im / _market_var
        _betas.append(_beta)

    _betas = np.array(_betas)

    # CAPM Implied Returns
    _capm_returns = _rf + _betas * (_rm_target - _rf)

    # Empirical vs CAPM Returns
    df_capm = pd.DataFrame(
        {
            "Asset": asset_names,
            "Beta": _betas,
            "Empirical Return": annual_returns.values,
            "CAPM Return": _capm_returns,
        }
    )

    # SML Line data
    _b_min = min(0.0, min(_betas)) - 0.2
    _b_max = max(1.5, max(_betas)) + 0.2
    _b_range = np.linspace(_b_min, _b_max, 2)
    _sml_returns = _rf + _b_range * (_rm_target - _rf)

    df_sml = pd.DataFrame({"Beta": _b_range, "CAPM Return": _sml_returns})

    _market_ret = df_returns.dot(_w_mkt)
    _reg_rows = []
    for _col in asset_names:
        for _idx in range(len(df_returns)):
            _reg_rows.append(
                {
                    "Market Return": _market_ret.iloc[_idx] * 100,
                    "Asset Return": df_returns[_col].iloc[_idx] * 100,
                    "Asset": _col,
                }
            )
    df_capm_scatter = pd.DataFrame(_reg_rows)

    capm_results = {
        "assets": df_capm,
        "sml": df_sml,
        "scatter": df_capm_scatter,
    }
    return capm_results


@app.cell
def _(alt, capm_results, pd):
    _df_assets = capm_results["assets"]
    _df_sml = capm_results["sml"]

    _sml_line = (
        alt.Chart(_df_sml)
        .mark_line(color="#e74c3c", strokeWidth=2.5, strokeDash=[5, 5])
        .encode(
            x=alt.X("Beta:Q", title="Beta β (systematic risk)", scale=alt.Scale(zero=False)),
            y=alt.Y(
                "CAPM Return:Q", title="Expected Return E[r]", axis=alt.Axis(format="%")
            ),
        )
    )

    _emp_points = (
        alt.Chart(_df_assets)
        .mark_circle(size=350, opacity=0.9, stroke="white", strokeWidth=2)
        .encode(
            x="Beta:Q",
            y="Empirical Return:Q",
            color=alt.Color("Asset:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=[
                "Asset:N",
                alt.Tooltip("Beta:Q", format=".2f"),
                alt.Tooltip("Empirical Return:Q", format=".2%", title="Actual return"),
                alt.Tooltip("CAPM Return:Q", format=".2%", title="CAPM predicted return"),
            ],
        )
    )

    _emp_text = _emp_points.mark_text(
        align="left", baseline="middle", dx=14, fontSize=13, fontWeight="bold"
    ).encode(text="Asset:N")

    # Annotation labels: "α > 0 above SML" and "α < 0 below SML"
    _b_mid = float((_df_sml["Beta"].min() + _df_sml["Beta"].max()) / 2)
    _mid_capm = float(_df_assets["CAPM Return"].mean())
    _anno_df = pd.DataFrame([
        {"Beta": _b_mid, "y": _mid_capm * 1.4, "text": "α > 0  (above SML = outperforms)"},
        {"Beta": _b_mid, "y": _mid_capm * 0.6, "text": "α < 0  (below SML = underperforms)"},
    ])
    _anno = (
        alt.Chart(_anno_df)
        .mark_text(fontSize=11, fontStyle="italic", color="#888888")
        .encode(x="Beta:Q", y=alt.Y("y:Q"), text="text:N")
    )

    capm_chart = (
        (_sml_line + _emp_points + _emp_text + _anno)
        .properties(
            width=560,
            height=380,
            title=alt.TitleParams(
                text="Security Market Line: each dot's distance from the line = α",
                fontSize=16,
                anchor="start",
            ),
        )
    )

    return capm_chart


@app.cell
def _(alt, capm_results):
    _df = capm_results["scatter"]
    _scatter = (
        alt.Chart(_df)
        .mark_circle(opacity=0.4, size=40)
        .encode(
            x=alt.X("Market Return:Q", title="Market return (%)"),
            y=alt.Y("Asset Return:Q", title="Asset return (%)"),
            color=alt.Color("Asset:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=["Asset:N", "Market Return:Q", "Asset Return:Q"],
        )
    )
    _reg = (
        alt.Chart(_df)
        .transform_regression("Market Return", "Asset Return", groupby=["Asset"])
        .mark_line(strokeWidth=2, strokeDash=[5, 5])
        .encode(
            x="Market Return:Q",
            y="Asset Return:Q",
            color=alt.Color("Asset:N", scale=alt.Scale(scheme="tableau10")),
        )
    )
    capm_beta_chart = (
        (_scatter + _reg)
        .properties(
            width=380,
            height=380,
            title="β from data: slope of asset vs market (each line's slope = β)",
        )
    )
    return capm_beta_chart


@app.cell
def _(mo):
    mo.md(
        "**How to read the charts below:**\n\n"
        "- **Left chart (SML):** The red dashed line is what CAPM *predicts* each asset should return "
        "given its β. Dots *above* the line beat the prediction (positive α); dots *below* miss it (negative α). "
        "β = 1 means the asset moves 1-for-1 with the market; β > 1 means it amplifies market moves.\n\n"
        "- **Right chart (β from data):** Each colored cloud is one asset's daily return plotted against "
        "the market's daily return. The dashed line is the OLS regression — its *slope* is β. "
        "Steeper slope = higher β = more systematic risk.\n\n"
        "*Note: The market portfolio here is approximated as an equal-weighted portfolio.*"
    )
    return


@app.cell
def _(mo, capm_controls):
    mo.vstack([mo.md("---"), capm_controls])
    return


@app.cell
def _(mo, capm_chart, capm_beta_chart):
    mo.hstack([capm_chart, capm_beta_chart], gap="2rem")
    return


@app.cell
def _(alt, capm_results, pd):
    _df = capm_results["assets"]
    df_alpha = pd.DataFrame({
        "Asset": _df["Asset"].values,
        "Alpha": (_df["Empirical Return"].values - _df["CAPM Return"].values),
    })
    df_alpha["Type"] = df_alpha["Alpha"].apply(
        lambda x: "α > 0 (outperforms)" if x >= 0 else "α < 0 (underperforms)"
    )
    capm_alpha_chart = (
        alt.Chart(df_alpha)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("Asset:N", title=None, sort=alt.EncodingSortField("Alpha", order="descending")),
            y=alt.Y("Alpha:Q", title="α  (Actual − CAPM Predicted)", axis=alt.Axis(format="%")),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["α > 0 (outperforms)", "α < 0 (underperforms)"],
                    range=["#2ecc71", "#e74c3c"],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=["Asset:N", alt.Tooltip("Alpha:Q", format=".2%"), "Type:N"],
        )
        .properties(width=560, height=280, title="CAPM Alpha per asset — gap between actual return and model prediction")
    )
    return capm_alpha_chart, df_alpha


@app.cell
def _(capm_alpha_chart, mo):
    mo.vstack([
        mo.md("**Alpha (α):** Distance from the Security Market Line. Green bars beat the CAPM prediction; red bars underperformed. In a perfectly efficient market all α = 0. Any persistent positive α is the *holy grail* of active management — evidence of genuine skill or an unpriced risk factor."),
        capm_alpha_chart,
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### The Rigor: Beta and Alpha

        **Variable key** — every symbol used below is defined here first:

        | Symbol | Meaning | Where you see it |
        |--------|---------|-----------------|
        | $r_i$ | return of asset $i$ (daily or annual) | y-axis of right chart |
        | $r_m$ | return of the market portfolio (equal-weighted here) | x-axis of right chart |
        | $r_f$ | risk-free rate (the "Risk-Free Rate" slider) | y-intercept of SML in left chart |
        | $E[\cdot]$ | expected value — historical mean | — |
        | $\beta_i$ | systematic risk of asset $i$ | x-axis of left chart; slope of regression in right chart |
        | $\text{Cov}(r_i, r_m)$ | covariance of asset $i$ with the market | captures how tightly $r_i$ moves with $r_m$ |
        | $\text{Var}(r_m)$ | variance of market returns | normalises $\beta$ to be scale-free |
        | $\alpha_i$ | actual return minus model-predicted return | vertical gap from SML in left chart; bar chart below |

        **Core CAPM equation** — the SML line in the left chart:
        $$E[r_i] = r_f + \beta_i\,(E[r_m] - r_f)$$

        **Beta** — the slope of each regression line in the right chart:
        $$\beta_i = \frac{\text{Cov}(r_i,\,r_m)}{\text{Var}(r_m)}$$

        **Alpha** — the vertical gap between each dot and the SML:
        $$\alpha_i = r_i^{\text{actual}} - \bigl(r_f + \beta_i(r_m - r_f)\bigr)$$

        Seeking persistent positive $\alpha$ is the holy grail of active management. CAPM purists argue true $\alpha$ doesn't exist and any apparent $\alpha$ is compensation for an unmeasured risk factor.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    <a id='approach3'></a>
    ## Approach 3: Risk Parity

    ### The Intuition: True Balance
    If you split $100 evenly between 5 assets ($20 each), you have an Equal-Weighted portfolio. It feels balanced.
    
    But what if Asset 1 is an ultra-safe Treasury bond that barely moves, and Asset 2 is an ultra-volatile cryptocurrency? Your portfolio's daily swings will be almost entirely driven by the crypto. Even though your *dollars* are balanced 50/50, your *risk* is allocated 5/95.
    
    Risk Parity ignores expected returns entirely. It focuses purely on volatility, attempting to weight assets such that each one contributes the exact same amount of overall variance to the portfolio.

    ### The Tool: Equalizing Contributions
    The bar chart below compares a naïve "Equal Weight" dollar allocation to a sophisticated "Risk Parity" allocation for our simulated universe. 
    """)
    return


@app.cell
def _(annual_cov, asset_names, np, pd, sco):
    _n_assets = len(asset_names)

    # 1. Equal Weight Portfolio
    _w_ew = np.ones(_n_assets) / _n_assets
    _vol_ew = np.sqrt(np.dot(_w_ew.T, np.dot(annual_cov, _w_ew)))
    _rc_ew = _w_ew * np.dot(annual_cov, _w_ew) / _vol_ew
    _ptr_rc_ew = _rc_ew / _vol_ew  # Percentage Risk Contribution

    # 2. Risk Parity Portfolio Optimization
    def _risk_budget_objective(w):
        w = np.array(w)
        port_vol = np.sqrt(np.dot(w.T, np.dot(annual_cov, w)))
        if port_vol == 0:
            return 1e9
        risk_contributions = w * np.dot(annual_cov, w) / port_vol
        # Target is equal risk contribution
        target_rc = port_vol / _n_assets
        # Minimize sum of squared differences from target
        return np.sum(np.square(risk_contributions - target_rc))

    _bounds = tuple((0.0, 1.0) for _ in range(_n_assets))
    _constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    _init_guess = _w_ew

    _opt_rp = sco.minimize(
        _risk_budget_objective,
        _init_guess,
        method="SLSQP",
        bounds=_bounds,
        constraints=_constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    _w_rp = _opt_rp.x
    _vol_rp = np.sqrt(np.dot(_w_rp.T, np.dot(annual_cov, _w_rp)))
    _rc_rp = _w_rp * np.dot(annual_cov, _w_rp) / _vol_rp
    _ptr_rc_rp = _rc_rp / _vol_rp

    # Prepare DataFrame for Altair
    _data = []
    for _i, _asset in enumerate(asset_names):
        _data.append([_asset, "Equal Weight", "Weight", _w_ew[_i]])
        _data.append([_asset, "Equal Weight", "Risk Contribution", _ptr_rc_ew[_i]])
        _data.append([_asset, "Risk Parity", "Weight", _w_rp[_i]])
        _data.append([_asset, "Risk Parity", "Risk Contribution", _ptr_rc_rp[_i]])

    df_rp = pd.DataFrame(_data, columns=["Asset", "Portfolio", "Metric", "Value"])

    rp_results = {
        "data": df_rp,
        "w_rp": _w_rp,
        "vol_rp": _vol_rp,
        "vol_ew": _vol_ew,
        "ptr_rc_ew": _ptr_rc_ew,
        "ptr_rc_rp": _ptr_rc_rp,
    }
    return rp_results


@app.cell
def _(alt, rp_results):
    _df = rp_results["data"]

    rp_chart = (
        alt.Chart(_df)
        .mark_bar(cornerRadiusEnd=4, size=25)
        .encode(
            x=alt.X(
                "Asset:N", title=None, axis=alt.Axis(labelAngle=-45, labelFontSize=12)
            ),
            y=alt.Y("Value:Q", axis=alt.Axis(format="%")),
            color=alt.Color(
                "Asset:N", legend=None, scale=alt.Scale(scheme="tableau10")
            ),
            column=alt.Column(
                "Portfolio:N",
                title=None,
                header=alt.Header(labelFontSize=14, labelFontWeight="bold"),
            ),
            row=alt.Row(
                "Metric:N",
                title=None,
                header=alt.Header(labelFontSize=14, labelFontWeight="bold"),
            ),
            tooltip=[
                "Asset:N",
                "Portfolio:N",
                "Metric:N",
                alt.Tooltip("Value:Q", format=".2%"),
            ],
        )
        .properties(
            width=300,
            height=200,
            title=alt.TitleParams(
                text="Weight vs Risk Contribution", fontSize=20, anchor="start"
            ),
        )
        .resolve_scale(y="independent")
        .configure_axis(gridOpacity=0.2, labelFontSize=12, titleFontSize=14)
        .configure_view(strokeOpacity=0)
    )

    return rp_chart


@app.cell
def _(alt, asset_names, rp_results, pd):
    _vol_ew = rp_results["vol_ew"]
    _vol_rp = rp_results["vol_rp"]
    df_rp_vol = pd.DataFrame(
        [
            {"Portfolio": "Equal Weight", "Volatility": _vol_ew},
            {"Portfolio": "Risk Parity", "Volatility": _vol_rp},
        ]
    )
    rp_vol_chart = (
        alt.Chart(df_rp_vol)
        .mark_bar(cornerRadiusEnd=6, size=50)
        .encode(
            x=alt.X("Portfolio:N", title=None),
            y=alt.Y("Volatility:Q", axis=alt.Axis(format="%"), title="Portfolio volatility"),
            color=alt.Color("Portfolio:N", scale=alt.Scale(range=["#7b68ee", "#00c853"])),
            tooltip=[alt.Tooltip("Volatility:Q", format=".2%")],
        )
        .properties(width=280, height=200, title="Portfolio volatility: EW vs Risk Parity")
    )

    return df_rp_vol, rp_vol_chart


@app.cell
def _(mo):
    mo.vstack([mo.md("---"), mo.md("### Equal Weight vs Risk Parity Allocation")])
    return


@app.cell
def _(rp_chart):
    rp_chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    **How to read the chart above:**
    * **Top two rows (Weight):** where the dollars go. Equal Weight puts identical dollars in every asset. Risk Parity puts *more* dollars in low-volatility assets and *fewer* in high-volatility ones.
    * **Bottom two rows (Risk Contribution):** how much of the portfolio's total variance each asset drives. Equal Weight has wildly uneven risk contributions — your highest-vol asset dominates. Risk Parity makes all risk contributions identical.

    ### The Rigor: Marginal Risk Contribution

    **Variable key:**

    | Symbol | Meaning |
    |--------|---------|
    | $w$ | weight vector $[w_1, w_2, \ldots, w_n]^T$, with $\sum w_i = 1$ |
    | $\Sigma$ | $n \times n$ covariance matrix of asset returns |
    | $\sigma_p$ | portfolio volatility $= \sqrt{w^T \Sigma w}$ |
    | $MRC_i$ | marginal risk contribution of asset $i$ — how much $\sigma_p$ rises per unit increase in $w_i$ |
    | $RC_i$ | total risk contribution of asset $i$ |

    The **Marginal Risk Contribution** of asset $i$:
    $$MRC_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}$$

    The **Total Risk Contribution** of asset $i$ (weight × marginal contribution):
    $$RC_i = w_i \cdot \frac{(\Sigma w)_i}{\sigma_p}$$

    The Risk Parity objective: find $w$ such that $RC_1 = RC_2 = \cdots = RC_n$.
    This approach gained massive popularity after the 2008 financial crisis — it builds resilient "all-weather" portfolios without requiring the notoriously hard-to-predict return estimates $\mu$.
    """)
    return


@app.cell
def _(mo, rp_vol_chart):
    mo.vstack([
        mo.md(
            "**Result: does Risk Parity actually reduce volatility?** "
            "Because RP avoids concentrating risk in the most volatile assets, it typically produces "
            "a *lower total portfolio volatility* than Equal Weight — even though it doesn't explicitly minimise vol. "
            "It minimises *concentrated* risk, and that indirectly reduces overall variance."
        ),
        rp_vol_chart,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Tradeoff Summary: Comparing the Approaches

    | Model | Required Inputs | Core Strength | Core Weakness |
    |-------|----------------|---------------|---------------|
    | **MVO** (Markowitz) | $\mu$ (expected returns) + $\Sigma$ (covariance) | Mathematically optimal risk/return trade-off on the efficient frontier | "Error maximizer" — small errors in $\mu$ estimates produce extreme, concentrated portfolios |
    | **CAPM** | $\beta$ per asset + market return $r_m$, risk-free rate $r_f$ | Elegant theory; prices systematic risk; no optimisation needed | Assumes perfectly efficient markets; empirically a weak predictor of future returns |
    | **Risk Parity** | $\Sigma$ only (no return estimates) | Robust to return-estimation error; resilient in volatile regimes | Ignores expected returns entirely; can underperform during strong trending bull markets |

    **Rule of thumb:**
    - Trust your return forecasts? → **MVO** for the best risk/return trade-off
    - Want theoretical pricing of risk? → **CAPM** for a benchmark
    - Sceptical of all return forecasts? → **Risk Parity** for a robust baseline
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <a id='approach4'></a>
    ## Approach 4: Black-Litterman Model

    ### The Intuition: The Bayesian Blend
    As we saw in Markowitz, unconstrained optimizers act as "error maximizers." If Asset 1 has historically great returns, the math blindly forces all your capital into Asset 1. But real-world institutional investors know that history is noisy. 
    
    The Black-Litterman approach proposes a brilliant compromise: What if we assume that the current market weights represent the perfect, collective wisdom of the crowd (the "Equilibrium")? We start tightly anchored to the market index, and *only* deviate from it where we possess a strong, mathematical, actionable "View" (e.g., our research analyst thinks Asset 1 will beat Asset 2 by 4%).
    
    By utilizing Bayesian statistics, the Black-Litterman model gracefully blends the wisdom of the market with our specific proprietary views. 

    ### The Tool: Scaling Market Weights
    Below, set a relative View using the slider (e.g. Asset 1 outperforming Asset 2), and denote your uncertainty in that view ($\tau$).
    
    Notice how the orange "Posterior" weights gracefully expand and contract around the Market Equilibrium, without creating massive, jagged 100% allocations.
    """)
    return


@app.cell
def _(mo):
    bl_view_ui = mo.ui.slider(
        start=-0.10,
        stop=0.10,
        step=0.01,
        value=0.05,
        label="Asset 1 vs Asset 2 Return View",
    )
    bl_conf_ui = mo.ui.slider(
        start=0.01, stop=0.10, step=0.01, value=0.05, label="View Uncertainty (tau)"
    )

    bl_controls = mo.vstack(
        [
            mo.md("### Black-Litterman View"),
            mo.md("Specify a relative view: 'Asset 1 will outperform Asset 2 by X%'."),
            mo.hstack([bl_view_ui, bl_conf_ui]),
        ]
    )
    return bl_conf_ui, bl_controls, bl_view_ui


@app.cell
def _(annual_cov, asset_names, bl_conf_ui, bl_view_ui, np, pd):
    _n_assets = len(asset_names)
    _tau = bl_conf_ui.value
    _view = bl_view_ui.value

    # 1. Prior: Implied Equilibrium Returns
    _w_mkt = np.ones(_n_assets) / _n_assets
    _lambda = 2.5  # Risk aversion parameter
    _pi = _lambda * np.dot(annual_cov, _w_mkt)

    # 2. Views Setup
    # P: Link matrix. View: Asset 1 outperforms Asset 2
    _P = np.zeros((1, _n_assets))
    _P[0, 0] = 1.0
    _P[0, 1] = -1.0

    # Q: View vector
    _Q = np.array([_view])

    # Omega: Uncertainty matrix of the views
    # Common assumption: proportional to prior variance
    _omega = np.dot(np.dot(_P, _tau * annual_cov), _P.T)
    # Ensure it's invertible
    if np.abs(_omega[0, 0]) < 1e-8:
        _omega[0, 0] = 1e-6

    # 3. Posterior Returns and Covariance
    _inv_tau_cov = np.linalg.inv(_tau * annual_cov)
    _inv_omega = np.linalg.inv(_omega)

    _post_cov = np.linalg.inv(_inv_tau_cov + np.dot(np.dot(_P.T, _inv_omega), _P))
    _post_returns = _post_cov.dot(
        (_inv_tau_cov.dot(_pi) + _P.T.dot(_inv_omega).dot(_Q))
    )

    # 4. Posterior Weights
    _w_post = np.linalg.inv(_lambda * annual_cov).dot(_post_returns)
    # Normalize weights to sum to 1
    _w_post = _w_post / np.sum(_w_post)

    # 5. Prepare data for plotting
    _data_ret = []
    _data_w = []

    for _i, _asset in enumerate(asset_names):
        # Returns
        _data_ret.append([_asset, "Prior (Market Eq)", _pi[_i]])
        _data_ret.append([_asset, "Posterior (With View)", _post_returns[_i]])
        # Weights
        _data_w.append([_asset, "Prior (Market Eq)", _w_mkt[_i]])
        _data_w.append([_asset, "Posterior (With View)", _w_post[_i]])

    df_bl_ret = pd.DataFrame(_data_ret, columns=["Asset", "Model", "Expected Return"])
    df_bl_w = pd.DataFrame(_data_w, columns=["Asset", "Model", "Weight"])

    bl_results = {"returns": df_bl_ret, "weights": df_bl_w}
    return bl_results


@app.cell
def _(alt, bl_results):
    _df_ret = bl_results["returns"]
    _df_w = bl_results["weights"]

    _color_scale = alt.Scale(
        domain=["Prior (Market Eq)", "Posterior (With View)"],
        range=["#673AB7", "#00E676"],
    )

    bl_ret_chart = (
        alt.Chart(_df_ret)
        .mark_bar(cornerRadiusEnd=4, size=25)
        .encode(
            x=alt.X("Model:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("Expected Return:Q", axis=alt.Axis(format="%")),
            color=alt.Color(
                "Model:N",
                scale=_color_scale,
                legend=alt.Legend(orient="bottom", title=None, labelFontSize=13),
            ),
            column=alt.Column(
                "Asset:N",
                title="Expected Returns: Prior vs Posterior",
                header=alt.Header(titleFontSize=14, labelFontSize=12),
            ),
            tooltip=["Asset:N", "Model:N", alt.Tooltip("Expected Return:Q", format=".2%")],
        )
        .properties(width=80, height=220)
    )

    bl_weights_chart = (
        alt.Chart(_df_w)
        .mark_bar(cornerRadiusEnd=4, size=25)
        .encode(
            x=alt.X("Model:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("Weight:Q", axis=alt.Axis(format="%")),
            color=alt.Color("Model:N", scale=_color_scale, legend=None),
            column=alt.Column(
                "Asset:N",
                title="Portfolio Weights: Prior vs Posterior",
                header=alt.Header(titleFontSize=14, labelFontSize=12),
            ),
            tooltip=["Asset:N", "Model:N", alt.Tooltip("Weight:Q", format=".2%")],
        )
        .properties(width=80, height=220)
    )

    return bl_ret_chart, bl_weights_chart


@app.cell
def _(alt, bl_results, pd):
    _df_ret = bl_results["returns"]
    _prior = _df_ret[_df_ret["Model"] == "Prior (Market Eq)"].set_index("Asset")["Expected Return"]
    _post = _df_ret[_df_ret["Model"] == "Posterior (With View)"].set_index("Asset")["Expected Return"]
    df_bl_scatter = pd.DataFrame(
        {"Prior": _prior, "Posterior": _post}
    ).reset_index()
    _min_r = min(_prior.min(), _post.min()) - 0.01
    _max_r = max(_prior.max(), _post.max()) + 0.01
    df_bl_identity = pd.DataFrame({"Prior": [_min_r, _max_r], "Posterior": [_min_r, _max_r]})

    _pts = (
        alt.Chart(df_bl_scatter)
        .mark_circle(size=220, stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("Prior:Q", title="Prior (market equilibrium) return", axis=alt.Axis(format="%")),
            y=alt.Y("Posterior:Q", title="Posterior return", axis=alt.Axis(format="%")),
            color=alt.Color("Asset:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=["Asset:N", alt.Tooltip("Prior:Q", format=".2%"), alt.Tooltip("Posterior:Q", format=".2%")],
        )
    )
    _ref = (
        alt.Chart(df_bl_identity)
        .mark_line(strokeDash=[6, 4], stroke="#888", strokeWidth=2)
        .encode(x="Prior:Q", y="Posterior:Q")
    )
    bl_prior_posterior_chart = (
        (_ref + _pts)
        .properties(
            width=400,
            height=380,
            title="Prior vs posterior expected return (y=x = no change)",
        )
    )
    return df_bl_identity, df_bl_scatter, bl_prior_posterior_chart


@app.cell
def _(mo, bl_controls):
    mo.vstack(
        [mo.md("---"), bl_controls, mo.md("**Prior vs posterior returns:** Points off the dashed line show how your view shifts expected returns.")]
    )
    return


@app.cell
def _(mo, bl_ret_chart, bl_weights_chart, bl_prior_posterior_chart):
    mo.vstack([bl_ret_chart, bl_weights_chart, bl_prior_posterior_chart])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **How to read this chart:**
    * **Left side (Expected Returns):** The blue bars show the baseline implied returns if we assume the market is currently in perfect equilibrium. The orange bars show how those returns update mathematically after blending in your subjective slider view.
    * **Right side (Portfolio Weights):** The blue bars show the starting market capitalization weights. The orange bars reveal the optimal portfolio you should hold given the new posterior returns.

    ### The Rigor: Reverse Engineering the Market

    Instead of giving the computer Expected Returns ($\mu$) to find Weights ($w$), Black-Litterman performs **Reverse Optimization**. It takes the known market weights ($w_{mkt}$), and calculates the Implied Equilibrium Returns ($\Pi$) the market *must* be expecting:

    $$\Pi = \lambda \Sigma w_{mkt}$$

    We then formalize our subjective "Views" as a matrix equation ($P \cdot E[R] = Q + \tilde{\epsilon}$), where $P$ identifies the assets, $Q$ is the magnitude of the view, and $\Omega$ (or $\tau$) acts as the diagonal covariance/uncertainty matrix of the views.

    Through Bayesian blending, we obtain the **Posterior Expected Returns**:
    $$E[R] = \Pi + \tau \Sigma P^T (P \tau \Sigma P^T + \Omega)^{-1} (Q - P\Pi)$$

    We then feed this much-smoother $E[R]$ vector back into the standard Markowitz optimizer to get our final, remarkably stable weights.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <a id='approach5'></a>
    ## Approach 5: Machine Learning-Based Optimization

    ### The Intuition: Penalizing Extremes
    What if we don't have subjective Views like in Black-Litterman, but we *still* know our historical data is extremely noisy?
    
    Machine learning thrives on identifying patterns in massive datasets, but financial time series are notoriously short, non-stationary, and noisy. If we let an algorithm run totally free on financial data, it will dramatically "overfit" to the noise, building a wildly concentrated portfolio that performed perfectly in the past but will crash in the future.
    
    To fix this, modern quantitative finance borrows a trick from machine learning: **Regularization**. It acts as a mathematical speed bump that penalizes the computer for creating extreme solutions, forcing it to find a "simpler", smoother answer that generalizes better to unseen future data.

    ### The Tool: Ridge Regularization
    Use the slider to slowly turn on the L2 Regularization Penalty. 
    
    Watch the optimizer physically struggle to preserve the high-concentration Standard MVO weights. The higher the penalty, the more the weights get "shrunk" downward into a uniform, balanced distribution to avoid triggering the math penalty.
    """)
    return


@app.cell
def _(mo):
    ml_l2_ui = mo.ui.slider(
        start=0.0, stop=2.0, step=0.1, value=0.5, label="L2 Regularization Penalty"
    )

    ml_controls = mo.vstack(
        [
            mo.md("### Machine Learning: Ridge Regularized Optimization"),
            mo.md(
                "Applying L2 regularization (Ridge) to the Mean-Variance objective prevents extreme weight allocations by penalizing large concentrated positions. This shrinks weights towards a more uniform distribution (similar to equal-weight) and reduces sensitivity to noisy inputs."
            ),
            ml_l2_ui,
        ]
    )
    return ml_controls, ml_l2_ui


@app.cell
def _(annual_cov, annual_returns, asset_names, ml_l2_ui, np, pd, sco):
    _n_assets = len(asset_names)
    _l2_penalty = ml_l2_ui.value

    # 1. Standard Unconstrained Tangency (Max Sharpe)
    # Using risk-free rate = 0.02 for demonstration
    _rf = 0.02

    def _neg_sharpe(w):
        w = np.array(w)
        ret = np.sum(annual_returns * w)
        vol = np.sqrt(np.dot(w.T, np.dot(annual_cov, w)))
        if vol == 0:
            return 1e9
        return -(ret - _rf) / vol

    _bounds = tuple((0.0, 1.0) for _ in range(_n_assets))
    _constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    _init_guess = np.ones(_n_assets) / _n_assets

    _opt_std = sco.minimize(
        _neg_sharpe,
        _init_guess,
        method="SLSQP",
        bounds=_bounds,
        constraints=_constraints,
    )
    _w_std = _opt_std.x

    # 2. Regularized (Ridge) Tangency
    def _regularized_neg_sharpe(w):
        _base_obj = _neg_sharpe(w)
        # L2 Penalty on weights
        _penalty = _l2_penalty * np.sum(w**2)
        return _base_obj + _penalty

    _opt_reg = sco.minimize(
        _regularized_neg_sharpe,
        _init_guess,
        method="SLSQP",
        bounds=_bounds,
        constraints=_constraints,
    )
    _w_reg = _opt_reg.x

    # Prepare Data
    _data = []
    for _i, _asset in enumerate(asset_names):
        _data.append([_asset, "Standard MVO", _w_std[_i]])
        _data.append([_asset, f"Ridge Regularized (L2={_l2_penalty})", _w_reg[_i]])

    df_ml = pd.DataFrame(_data, columns=["Asset", "Optimization Type", "Weight"])

    return (df_ml,)


@app.cell
def _(alt, df_ml):
    ml_chart = (
        alt.Chart(df_ml)
        .mark_bar(cornerRadiusEnd=4, size=30)
        .encode(
            x=alt.X(
                "Optimization Type:N",
                title=None,
                axis=alt.Axis(labels=False, ticks=False),
            ),
            y=alt.Y("Weight:Q", axis=alt.Axis(format="%")),
            color=alt.Color(
                "Optimization Type:N",
                scale=alt.Scale(range=["#3b82f6", "#f43f5e"]),
                legend=alt.Legend(orient="bottom", title=None, labelFontSize=13),
            ),
            column=alt.Column(
                "Asset:N",
                title="Weight Allocation Strategy Comparison",
                header=alt.Header(titleFontSize=16, labelFontSize=13),
            ),
            tooltip=[
                "Asset:N",
                "Optimization Type:N",
                alt.Tooltip("Weight:Q", format=".2%"),
            ],
        )
        .properties(width=120, height=350)
        .configure_axis(gridOpacity=0.2, labelFontSize=12, titleFontSize=14)
        .configure_view(strokeOpacity=0)
    )

    return ml_chart


@app.cell
def _(alt, annual_cov, annual_returns, asset_names, np, pd, sco):
    _rf = 0.02
    _n_assets = len(asset_names)
    _bounds = tuple((0.0, 1.0) for _ in range(_n_assets))
    _constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    _init_guess = np.ones(_n_assets) / _n_assets

    def _neg_sharpe(w):
        w = np.array(w)
        ret = np.sum(annual_returns * w)
        vol = np.sqrt(np.dot(w.T, np.dot(annual_cov, w)))
        if vol == 0:
            return 1e9
        return -(ret - _rf) / vol

    _lambdas = np.linspace(0.0, 2.0, 15)
    _path_rows = []
    for _lam in _lambdas:
        def _reg_obj(w):
            return _neg_sharpe(w) + _lam * np.sum(np.array(w) ** 2)
        _opt = sco.minimize(
            _reg_obj,
            _init_guess,
            method="SLSQP",
            bounds=_bounds,
            constraints=_constraints,
        )
        _w = _opt.x
        for _i, _name in enumerate(asset_names):
            _path_rows.append({"λ (L2 penalty)": float(_lam), "Asset": _name, "Weight": _w[_i]})

    df_ml_path = pd.DataFrame(_path_rows)

    ml_path_chart = (
        alt.Chart(df_ml_path)
        .mark_line(strokeWidth=2.5, point=True)
        .encode(
            x=alt.X("λ (L2 penalty):Q", title="L2 regularization (λ)"),
            y=alt.Y("Weight:Q", axis=alt.Axis(format="%"), title="Weight"),
            color=alt.Color("Asset:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=["λ (L2 penalty):Q", "Asset:N", alt.Tooltip("Weight:Q", format=".2%")],
        )
        .properties(
            width=500,
            height=280,
            title="Regularization path: weights shrink toward equal as λ increases",
        )
    )
    return df_ml_path, ml_path_chart


@app.cell
def _(mo, ml_controls):
    mo.vstack(
        [mo.md("---"), ml_controls, mo.md("**Regularization path:** As λ increases, weights move toward equal allocation.")]
    )
    return


@app.cell
def _(mo, ml_chart, ml_path_chart):
    mo.vstack([ml_chart, ml_path_chart])
    return


@app.cell
def _(alt, annual_cov, annual_returns, asset_names, np, pd, sco):
    # Sharpe ratio and concentration vs regularization strength
    _rf = 0.02
    _n = len(asset_names)
    _bounds = tuple((0.0, 1.0) for _ in range(_n))
    _cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    _init = np.ones(_n) / _n

    def _base_neg_sr(w: np.ndarray) -> float:
        w = np.array(w)
        r = float(np.sum(annual_returns * w))
        vol = float(np.sqrt(np.dot(w, np.dot(annual_cov, w))))
        return -(r - _rf) / vol if vol > 0 else 1e9

    _lambdas = np.linspace(0, 2.0, 20)
    _rows = []
    for _lam in _lambdas:
        def _obj(w, lam=_lam):
            return _base_neg_sr(w) + lam * float(np.sum(np.array(w) ** 2))

        _opt = sco.minimize(_obj, _init, method="SLSQP", bounds=_bounds, constraints=_cons)
        _w = _opt.x
        _r = float(np.sum(annual_returns * _w))
        _vol = float(np.sqrt(np.dot(_w, np.dot(annual_cov, _w))))
        _sr = (_r - _rf) / _vol if _vol > 0 else 0.0
        _hhi = float(np.sum(_w ** 2))
        _rows.append({"λ (L2)": float(_lam), "Sharpe Ratio": _sr, "HHI Concentration": _hhi})
    df_sr_path = pd.DataFrame(_rows)

    _sr_line = (
        alt.Chart(df_sr_path)
        .mark_line(color="#3498db", strokeWidth=2.5, point=alt.OverlayMarkDef(color="#3498db", size=60))
        .encode(
            x=alt.X("λ (L2):Q", title="L2 Regularization (λ)"),
            y=alt.Y("Sharpe Ratio:Q", title="In-Sample Sharpe Ratio"),
            tooltip=["λ (L2):Q", alt.Tooltip("Sharpe Ratio:Q", format=".3f")],
        )
        .properties(width=310, height=240, title="Sharpe Ratio vs λ (falls with more regularization)")
    )
    _hhi_line = (
        alt.Chart(df_sr_path)
        .mark_line(color="#e74c3c", strokeWidth=2.5, point=alt.OverlayMarkDef(color="#e74c3c", size=60))
        .encode(
            x=alt.X("λ (L2):Q", title="L2 Regularization (λ)"),
            y=alt.Y("HHI Concentration:Q", title="HHI (lower = more diversified)"),
            tooltip=["λ (L2):Q", alt.Tooltip("HHI Concentration:Q", format=".3f")],
        )
        .properties(width=310, height=240, title="Concentration vs λ (also falls — less concentrated)")
    )
    ml_bias_var_chart = alt.hconcat(_sr_line, _hhi_line, spacing=30)
    return df_sr_path, ml_bias_var_chart


@app.cell
def _(ml_bias_var_chart, mo):
    mo.vstack([
        mo.md("### Bias-Variance Tradeoff"),
        mo.md("**Left:** Sharpe Ratio falls as λ increases — regularization costs us some return. **Right:** HHI (Herfindahl-Hirschman Index, a measure of concentration) also falls — the portfolio becomes more spread out and diversified. The optimal λ balances these two: accept a small return penalty in exchange for a portfolio that won't blow up on unseen future data."),
        ml_bias_var_chart,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **How to read this chart:**
    * The **Standard MVO** (blue) results in highly concentrated, extreme weights—capitalizing fully on whatever historical noise was present in the data.
    * The **Ridge Regularized** (orange) bars show the effect of the L2 penalty. As you increase the slider, the optimizer is numerically punished for huge weights, dragging the portfolio back toward a safer equal-weight baseline.

    ### The Rigor: L2 Norm Penalty

    In standard Markowitz, our objective function was simply minimizing variance subject to a return constraint. Machine learning introduces a regularization term to that loss function.

    **Ridge Regularization (L2 Norm)** adds the sum of the squared weights to the objective, multiplied by a strictness tuning parameter ($\lambda_{L2}$):

    $$\min_w \quad \frac{1}{2} w^T \Sigma w + \lambda_{L2} \sum_{i=1}^N w_i^2$$

    Because the penalty is squared, a single 80% weight ($0.8^2 = 0.64$) incurs a vastly harsher penalty than four 20% weights ($4 \times 0.2^2 = 0.16$). This non-linear mathematical punishment forces the optimizer to shrink weights evenly, dramatically increasing True Diversification and Out-Of-Sample performance. It's essentially mathematical humility built directly into the codebase.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Comparative Analysis and Synthesis

    ### Summary Table

    | Approach | Core Idea | Key Strength | Main Limitation | Best Use Case |
    |----------|-----------|--------------|-----------------|---------------|
    | **Markowitz** | Maximize return/risk | Theoretical foundation | Estimation error | Strategic allocation |
    | **CAPM** | Market equilibrium | Risk decomposition | Single factor | Performance attribution |
    | **Risk Parity** | Equal risk contribution | Stability, diversification | Ignores returns | Multi-asset portfolios |
    | **Black-Litterman** | Equilibrium + views | Intuitive, stable | Complexity | Active management |
    | **Machine Learning** | Data-driven patterns | Flexibility | Overfitting, black-box | Large datasets |

    ### Integration Framework

    **Strategic Layer (3-5 years):**
    - Risk Parity or Black-Litterman for asset allocation
    - Low turnover, emphasize diversification

    **Tactical Layer (6-12 months):**
    - Black-Litterman with macro views
    - Factor models

    **Alpha Layer (short-term):**
    - Machine Learning for security selection
    - Manage transaction costs

    ### Key Lessons

    1. **No Universal Best:** Each approach has context-dependent strengths
    2. **Estimation Error Matters:** Most important practical consideration
    3. **Diversification is Robust:** Works across all frameworks
    4. **Stability vs Optimality:** Key tradeoff in practice
    5. **Theory + Practice:** Classical methods + ML = future
    6. **Risk Management First:** More important than return optimization
    """)
    return


@app.cell
def _(alt, annual_cov, annual_returns, asset_names, bl_results, mvo_results, np, pd, rp_results, sco):
    def _ret_vol(w: np.ndarray) -> tuple[float, float]:
        r = float(np.dot(w, annual_returns))
        v = float(np.sqrt(np.dot(w, np.dot(annual_cov, w))))
        return r, v

    _n = len(asset_names)
    _w_ew = np.ones(_n) / _n
    _bl_w = bl_results["weights"]
    _w_bl = _bl_w[_bl_w["Model"] == "Posterior (With View)"].set_index("Asset").loc[asset_names, "Weight"].values

    _lam = 0.5
    _bounds = tuple((0.0, 1.0) for _ in range(_n))
    _constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    _init = np.ones(_n) / _n
    def _neg_sr(w):
        w = np.array(w)
        r = np.sum(annual_returns * w)
        vol = np.sqrt(np.dot(w, np.dot(annual_cov, w)))
        if vol == 0:
            return 1e9
        return -(r - 0.02) / vol + _lam * np.sum(w**2)
    _opt_ridge = sco.minimize(_neg_sr, _init, method="SLSQP", bounds=_bounds, constraints=_constraints)
    _w_ridge = _opt_ridge.x

    _portfolios = [
        ("Markowitz (tangency)", mvo_results["w_star"]),
        ("Market (equal weight)", _w_ew),
        ("Risk Parity", rp_results["w_rp"]),
        ("Black-Litterman", _w_bl),
        ("Ridge (λ=0.5)", _w_ridge),
    ]
    _rr = []
    _ww = []
    for _name, _w in _portfolios:
        _r, _v = _ret_vol(_w)
        _rr.append({"Approach": _name, "Return": _r, "Volatility": _v})
        for _i, _a in enumerate(asset_names):
            _ww.append({"Approach": _name, "Asset": _a, "Weight": _w[_i]})
    df_comparative_rr = pd.DataFrame(_rr)
    df_comparative_weights = pd.DataFrame(_ww)

    comparative_rr_chart = (
        alt.Chart(df_comparative_rr)
        .mark_point(filled=True, size=280, stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("Volatility:Q", title="Volatility (risk)", axis=alt.Axis(format="%")),
            y=alt.Y("Return:Q", title="Expected return", axis=alt.Axis(format="%")),
            color=alt.Color("Approach:N", scale=alt.Scale(scheme="tableau10"), title="Approach"),
            tooltip=[
                "Approach:N",
                alt.Tooltip("Return:Q", format=".2%"),
                alt.Tooltip("Volatility:Q", format=".2%"),
            ],
        )
        .properties(
            width=500,
            height=320,
            title="Where each approach lands in risk–return space",
        )
    )

    comparative_weights_chart = (
        alt.Chart(df_comparative_weights)
        .mark_bar(cornerRadiusEnd=3, size=22)
        .encode(
            x=alt.X("Asset:N", title=None, sort=asset_names),
            y=alt.Y("Weight:Q", axis=alt.Axis(format="%"), title="Weight"),
            color=alt.Color("Approach:N", scale=alt.Scale(scheme="tableau10")),
            column=alt.Column("Approach:N", title=None, header=alt.Header(labelFontSize=11)),
            tooltip=["Approach:N", "Asset:N", alt.Tooltip("Weight:Q", format=".2%")],
        )
        .properties(width=140, height=220)
    )

    return (
        df_comparative_rr,
        df_comparative_weights,
        comparative_rr_chart,
        comparative_weights_chart,
    )


@app.cell
def _(mo, comparative_rr_chart, comparative_weights_chart):
    mo.vstack([comparative_rr_chart, comparative_weights_chart])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Conclusions

    ### The Evolution of Portfolio Optimization

    From Markowitz (1952) to modern ML (2020s), portfolio optimization has evolved from elegant mathematics to sophisticated data-driven systems.

    **Journey through the five frameworks:**

    ```mermaid
    flowchart LR
        M[Markowitz 1952]
        C[CAPM]
        R[Risk Parity]
        B[Black-Litterman]
        ML[ML Ridge]
        M --> C --> R --> B --> ML
    ```

    Each step addresses limitations of the previous: CAPM adds equilibrium and Beta; Risk Parity ignores noisy μ; Black-Litterman blends views with the market; ML regularization penalizes extreme weights.

    ### Universal Principles

    1. **Risk-Return Tradeoff:** Higher returns require higher risk
    2. **Diversification:** The only "free lunch" in finance
    3. **Estimation Uncertainty:** Robust methods acknowledge limited data
    4. **Market Efficiency:** Partial efficiency makes alpha generation difficult
    5. **Adaptation:** Markets evolve; strategies must adapt

    ### Practical Wisdom

    **For Individual Investors:**
    - Simple strategies often beat complex ones
    - Focus on asset allocation over security selection
    - Minimize costs (fees, taxes, trading)
    - Stay diversified and rebalance periodically

    **For Institutional Investors:**
    - Combine multiple approaches (strategic/tactical/alpha)
    - Invest in robust estimation methods
    - Understand model limitations
    - Balance sophistication with interpretability

    **As Benjamin Graham said:** *"The essence of investment management is the management of risks, not the management of returns."*

    ---

    ### References

    **Foundational Papers:**
    - Markowitz, H. (1952): "Portfolio Selection", Journal of Finance
    - Sharpe, W. (1964): "Capital Asset Prices"
    - Black, F. & Litterman, R. (1992): "Global Portfolio Optimization"
    - Fama, E. & French, K. (1993): "Common Risk Factors"

    **Books:**
    - Markowitz: "Portfolio Selection: Efficient Diversification"
    - Grinold & Kahn: "Active Portfolio Management"
    - López de Prado: "Advances in Financial Machine Learning"
    - Ang: "Asset Management: A Systematic Approach"

    ---

    *Thank you for following this journey through portfolio optimization!*

    *The best portfolio is one you can stick with through all market conditions.*
    """)
    return


if __name__ == "__main__":
    app.run()

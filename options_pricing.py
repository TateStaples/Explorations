import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt

    # Allow Altair to render >5000 rows (e.g. for high-N Binomial Trees)
    alt.data_transformers.disable_max_rows()

    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from scipy.stats import norm, binom

    return alt, binom, go, mo, norm, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Options Pricing: From Basics to the Greeks

        Welcome! This interactive notebook explores the mathematics and intuition behind options pricing.

        We'll start with no assumptions about prior options knowledge, build up to payoff diagrams,
        introduce the Black-Scholes model, and finally visualize the "Greeks" (the sensitivity of 
        option prices to various factors).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Introduction to Options and Payoffs
        
        Before diving into the math, it's worth asking: **Why do derivatives like options even exist?**
        
        While trading is often associated with speculation, the fundamental economic utility of derivatives is **insurance and risk distribution**. They allow businesses to isolate and offload risks they cannot afford to take, transferring them to market participants willing to bear that risk for a premium.
        
        Consider the **commodities markets**:
        - **Farmers**: A wheat farmer faces catastrophic risk if the price of wheat crashes before the harvest. To sleep securely at night, the farmer can buy a "Put Option"—mathematical insurance that guarantees the right to sell their crop at a minimum floor price, no matter how far the market falls.
        - **Airlines**: Jet fuel is a massive, volatile expense. An airline might buy a "Call Option" on oil, securing the right to buy fuel at a fixed maximum cap, insuring themselves against sudden geopolitical shocks that could otherwise bankrupt the fleet.
        
        At their core, options are just standardized insurance contracts.
        
        ### The Basics

        A **Call Option** gives the buyer the right (but not the obligation) to buy an underlying asset at a specified price (the **Strike Price**, $K$) on or before a specified date.
        A **Put Option** gives the buyer the right to sell the underlying asset at the Strike Price.

        If you hold a call option until expiration, its value is simply the difference between the underlying spot price ($S$) and the strike price ($K$), provided that difference is positive.
        If $S < K$, you wouldn't exercise the right to buy at $K$, so the payoff is 0.

        Call Payoff: $\max(S - K, 0)$  
        Put Payoff: $\max(K - S, 0)$

        Let's look at a payoff diagram. Adjust the strike price below:
        """
    )
    return


@app.cell
def _(mo):
    strike_slider = mo.ui.slider(
        start=50, stop=150, value=100, step=1, label="Strike Price (K)"
    )
    return (strike_slider,)


@app.cell
def _(mo, strike_slider):
    mo.vstack([strike_slider, mo.md(f"Current Strike: **{strike_slider.value}**")])
    return


@app.cell
def _(alt, np, pd, strike_slider):
    def plot_payoffs(K: float) -> alt.Chart:
        # Generate range of spot prices
        S = np.linspace(0, 200, 400)
        call_payoff = np.maximum(S - K, 0)
        put_payoff = np.maximum(K - S, 0)

        df = pd.DataFrame(
            {
                "Spot Price (S)": np.tile(S, 2),
                "Payoff": np.concatenate([call_payoff, put_payoff]),
                "Option Type": ["Call"] * len(S) + ["Put"] * len(S),
            }
        )

        # Base chart for lines
        chart = (
            alt.Chart(df)
            .mark_line(size=3)
            .encode(
                x=alt.X("Spot Price (S):Q", title="Spot Price at Expiration (S)"),
                y=alt.Y("Payoff:Q", title="Payoff at Expiration"),
                color=alt.Color("Option Type:N", title="Option Type"),
                strokeDash=alt.StrokeDash("Option Type:N", title="Option Type"),
                tooltip=["Spot Price (S)", "Payoff", "Option Type"],
            )
            .properties(
                title=f"Option Payoffs at Expiration (Strike = {K})",
                width=600,
                height=400,
            )
            .interactive()
        )
        return chart

    payoff_chart = plot_payoffs(strike_slider.value)
    return payoff_chart, plot_payoffs


@app.cell
def _(payoff_chart):
    payoff_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > **Try this:** Set the strike to **100** (ATM) and compare a Long Call vs a Long Put at the same strike — notice the payoff curves are mirror images. Now set the option type to **Long Straddle** or **Strangle** and drag the strike around: the wider the strangle, the cheaper it is, but the more the stock needs to move before it pays off.
        >
        > **Key insight:** A long call has *unlimited* upside but *limited* downside (premium paid). A short put has *limited* upside (premium collected) but can suffer large losses. This asymmetry — paying for optionality — is the core idea of options.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 1b. Put-Call Parity

        There is a fundamental no-arbitrage relationship between a European call and a European put with the same strike $K$ and expiry $T$:

        $$C - P = S - K e^{-rT}$$

        This says: **holding a long call and short put** is economically identical to **holding the stock and borrowing the present value of the strike**. If this relationship broke, you could earn a riskless profit by buying the cheap side and selling the expensive side — an arbitrage that markets quickly eliminate.

        Rearranging: if you know the call price, you can always compute the put price (and vice versa) without running the full pricing model:
        $$P = C - S + K e^{-rT}$$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Options Pricing: From Discrete to Continuous

        An option's total value has two components:

        - **Intrinsic Value**: The payoff if the option expired *right now*. For a call it's $\max(S - K, 0)$; for a put it's $\max(K - S, 0)$. An out-of-the-money option has zero intrinsic value.
        - **Time Value**: Everything on top of intrinsic value. It represents the *optionality* still remaining before expiration—the chance the option moves further in-the-money before it expires. Time value erodes to zero at expiration, which is why options are called *wasting assets*.

        **How do we price time value?** The core idea is **expected value under the right probability measure**. If we know the distribution of the stock price at expiration, the option's fair value is the discounted expected payoff:

        $$ C = e^{-rT} \, \mathbb{E}^{\mathbb{Q}}[\max(S_T - K, \, 0)] $$

        The $\mathbb{Q}$ superscript is key: we take the expectation under the *risk-neutral* measure, where every asset is assumed to grow at the risk-free rate $r$. This isn't a prediction of real-world returns—it's a mathematical trick that prices options consistently with no-arbitrage. The question then becomes: *what distribution does $S_T$ follow under $\mathbb{Q}$?*

        ### The Binomial Tree Approximation
        Let's start with a discrete, simplified world. Suppose the stock price today is $S_0$. Over a small time step $\Delta t$, the price can only do one of two things:
        1. Go **Up** by a factor $u$
        2. Go **Down** by a factor $d$

        By chaining these steps together, we build a "Binomial Tree" of possible future stock prices. We then work backward from expiration: calculate the option payoff at every final node, then walk back step-by-step, discounting the expected value at each node using risk-neutral probabilities.
        """
    )
    return


@app.cell
def _(mo):
    tree_steps = mo.ui.slider(
        start=1, stop=30, value=15, step=1, label="Tree Steps (N)"
    )
    tree_S = mo.ui.slider(start=50, stop=150, value=100, step=1, label="Spot Price (S)")
    tree_K = mo.ui.slider(start=50, stop=150, value=100, step=1, label="Strike (K)")
    tree_T = mo.ui.slider(
        start=0.1, stop=2.0, value=1.0, step=0.1, label="Time (T, years)"
    )
    tree_vol = mo.ui.slider(
        start=0.05, stop=1.0, value=0.2, step=0.05, label="Volatility (σ)"
    )
    # Defaulting r=0 as it is the most common introductory and intuitive assumption
    tree_r = mo.ui.slider(
        start=0.0, stop=0.2, value=0.0, step=0.01, label="Risk-Free Rate (r)"
    )

    tree_controls = mo.vstack(
        [
            mo.md("**Interactive Binomial Tree Valuer**"),
            mo.hstack([tree_steps, tree_S, tree_K, tree_T, tree_vol, tree_r], wrap=True),
        ]
    )
    return tree_controls, tree_K, tree_S, tree_T, tree_r, tree_steps, tree_vol


@app.cell
def _(alt, binom, mo, np, pd, tree_K, tree_S, tree_T, tree_r, tree_steps, tree_vol):
    def binomial_tree_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        N: int,
        option_type: str = "call",
        is_american: bool = False,
    ) -> tuple[float, pd.DataFrame, pd.DataFrame]:
        N = int(N)
        dt = T / N  # Time step
        drift = (r - 0.5 * sigma**2) * dt  # Per-step log-drift
        vol_step = sigma * np.sqrt(dt)
        u = np.exp(drift + vol_step)  # Up factor (carries drift so tree paths shift with r)
        d = np.exp(drift - vol_step)  # Down factor
        p = 0.5  # Equal risk-neutral probability (drift is in the nodes, not the weights)

        # Initialize asset prices at maturity (step N)
        ST = np.array([S * (u**j) * (d ** (N - j)) for j in range(N + 1)])

        # Probabilities of each node using binomial distribution
        probs = binom.pmf(np.arange(N + 1), N, p)
        pdf_df = pd.DataFrame({"Spot Price (S)": ST, "Probability": probs})

        # Build tree visualization coordinates as a single nan-separated line
        paths = []
        idx = 0
        for i in range(N):
            for j in range(i + 1):
                node_S = S * (u**j) * (d ** (i - j))
                paths.extend(
                    [
                        # Up move
                        {"Step": i, "Spot Price (S)": node_S, "DrawOrder": idx},
                        {
                            "Step": i + 1,
                            "Spot Price (S)": node_S * u,
                            "DrawOrder": idx + 1,
                        },
                        {
                            "Step": np.nan,
                            "Spot Price (S)": np.nan,
                            "DrawOrder": idx + 2,
                        },
                        # Down move
                        {"Step": i, "Spot Price (S)": node_S, "DrawOrder": idx + 3},
                        {
                            "Step": i + 1,
                            "Spot Price (S)": node_S * d,
                            "DrawOrder": idx + 4,
                        },
                        {
                            "Step": np.nan,
                            "Spot Price (S)": np.nan,
                            "DrawOrder": idx + 5,
                        },
                    ]
                )
                idx += 6

        tree_df = pd.DataFrame(paths)

        # Initialize option values at maturity
        if option_type == "call":
            C = np.maximum(ST - K, 0)
        else:
            C = np.maximum(K - ST, 0)

        # Step backwards through the tree
        discount = np.exp(-r * dt)
        for i in range(N - 1, -1, -1):
            C = discount * (p * C[1 : i + 2] + (1 - p) * C[0 : i + 1])
            if is_american:
                S_node = np.array([S * (u**j) * (d ** (i - j)) for j in range(i + 1)])
                if option_type == "call":
                    C = np.maximum(C, S_node - K)
                else:
                    C = np.maximum(C, K - S_node)

        return C[0], tree_df, pdf_df

    bt_call_eu, tree_df, pdf_df = binomial_tree_price(
        tree_S.value,
        tree_K.value,
        tree_T.value,
        tree_r.value,
        tree_vol.value,
        tree_steps.value,
        "call",
        False,
    )
    bt_put_eu, _, _ = binomial_tree_price(
        tree_S.value,
        tree_K.value,
        tree_T.value,
        tree_r.value,
        tree_vol.value,
        tree_steps.value,
        "put",
        False,
    )
    bt_call_am, _, _ = binomial_tree_price(
        tree_S.value,
        tree_K.value,
        tree_T.value,
        tree_r.value,
        tree_vol.value,
        tree_steps.value,
        "call",
        True,
    )
    bt_put_am, _, _ = binomial_tree_price(
        tree_S.value,
        tree_K.value,
        tree_T.value,
        tree_r.value,
        tree_vol.value,
        tree_steps.value,
        "put",
        True,
    )

    tree_output = mo.md(rf"""
    **Binomial Tree Estimates ({tree_steps.value} steps):**

    | Style | Call | Put |
    |---|---|---|
    | **European** | ${bt_call_eu:.4f} | ${bt_put_eu:.4f} |
    | **American** | ${bt_call_am:.4f} | ${bt_put_am:.4f} |
    """)

    tree_graph = (
        alt.Chart(tree_df)
        .mark_line(
            color="#9e9e9e",
            opacity=0.3,
            point=alt.OverlayMarkDef(filled=False, fill="white", color="#2196f3"),
        )
        .encode(
            x=alt.X("Step:Q", title="Step (Time)", axis=alt.Axis(format="d")),
            y=alt.Y(
                "Spot Price (S):Q", scale=alt.Scale(zero=False), title="Spot Price (S)"
            ),
            order="DrawOrder:Q",
            tooltip=["Step", "Spot Price (S)"],
        )
        .properties(title=f"Binomial Tree Paths", width=400, height=350)
    )

    pdf_chart = (
        alt.Chart(pdf_df)
        .mark_bar(orient="horizontal", opacity=0.8, color="#2196f3", size=3)
        .encode(
            y=alt.Y(
                "Spot Price (S):Q",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(labels=False, title=None),
            ),
            x=alt.X("Probability:Q", title="Terminal PDF"),
            tooltip=["Spot Price (S)", "Probability"],
        )
        .properties(title=f"Terminal PDF", width=150, height=350)
    )

    combined_chart = alt.hconcat(tree_graph, pdf_chart).resolve_scale(y="shared")

    return (
        binomial_tree_price,
        bt_call_eu,
        bt_put_eu,
        bt_call_am,
        bt_put_am,
        combined_chart,
        tree_output,
    )


@app.cell
def _(mo, combined_chart, tree_controls, tree_output):
    mo.vstack(
        [
            tree_controls,
            tree_output,
            mo.center(combined_chart),
            mo.md(
                r"*Try increasing the number of steps! As $N \to \infty$, the discrete steps blur into a continuous random walk (Geometric Brownian Motion)...*"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The Black-Scholes-Merton Model
        
        As the number of time steps $N \to \infty$, the Binomial Tree converges perfectly to continuous time pricing! The stock price follows a generic continuous random walk (Geometric Brownian Motion). 
        
        This breakthrough is known as the **Black-Scholes-Merton** model. It gives an elegant theoretical closed-form estimate for the price of European-style options.

        The formula for a Call ($C$) and Put ($P$) are:

        $$C = S N(d_1) - K e^{-rt} N(d_2)$$
        $$P = K e^{-rt} N(-d_2) - S N(-d_1)$$

        Where $d_1, d_2$ are:
        $$d_1 = \frac{\ln(S/K) + (r + \frac{\sigma^2}{2})t}{\sigma \sqrt{t}}$$
        $$d_2 = d_1 - \sigma \sqrt{t}$$

        Let's implement this continuous model and see how the price of an option compares to its expiration payoff continuously across all strikes.
        """
    )
    return


@app.cell
def _(np, norm):
    def black_scholes(
        S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
    ) -> float:
        """Calculates Black-Scholes price for European options."""
        if T <= 0:
            if option_type == "call":
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)

        # To avoid division by zero edge cases
        if sigma <= 0.0:
            sigma = 1e-4

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return price

    return (black_scholes,)


@app.cell
def _(mo):
    bs_strike = mo.ui.slider(start=50, stop=150, value=100, step=1, label="Strike (K)")
    bs_time = mo.ui.slider(
        start=0.01, stop=2.0, value=1.0, step=0.01, label="Time to Exp. (years, T)"
    )
    bs_vol = mo.ui.slider(
        start=0.01, stop=1.0, value=0.2, step=0.01, label="Volatility (σ)"
    )
    # Defaulting r=0 as it is the most common introductory and intuitive assumption
    bs_rate = mo.ui.slider(
        start=0.0, stop=0.2, value=0.0, step=0.01, label="Risk-Free Rate (r)"
    )

    bs_controls = mo.hstack([bs_time, bs_vol, bs_rate], wrap=True)
    return bs_controls, bs_rate, bs_strike, bs_time, bs_vol


@app.cell
def _(bs_controls, mo):
    mo.vstack([mo.md("### Interactive Black-Scholes Pricing"), bs_controls])

    return


@app.cell
def _(alt, black_scholes, bs_rate, bs_strike, bs_time, bs_vol, np, pd):
    def plot_bs_prices(K: float, T: float, sigma: float, r: float) -> alt.Chart:
        # Generate a range of Spot Prices
        S_vals = np.linspace(10, 200, 200)

        call_prices = [black_scholes(S, K, T, r, sigma, "call") for S in S_vals]
        put_prices = [black_scholes(S, K, T, r, sigma, "put") for S in S_vals]
        call_payoff = np.maximum(S_vals - K, 0)
        put_payoff = np.maximum(K - S_vals, 0)

        df = pd.DataFrame(
            {
                "Spot Price (S)": np.tile(S_vals, 4),
                "Price": np.concatenate(
                    [call_prices, put_prices, call_payoff, put_payoff]
                ),
                "Type": ["BS Call"] * len(S_vals)
                + ["BS Put"] * len(S_vals)
                + ["Call Payoff (T=0)"] * len(S_vals)
                + ["Put Payoff (T=0)"] * len(S_vals),
                "IsPayoff": [False] * len(S_vals) * 2 + [True] * len(S_vals) * 2,
            }
        )

        chart = (
            alt.Chart(df)
            .mark_line(size=2.5)
            .encode(
                x=alt.X("Spot Price (S):Q", title="Current Spot Price (S)"),
                y=alt.Y("Price:Q", title="Option Price / Payoff"),
                color=alt.Color(
                    "Type:N", title="Curve Type", scale=alt.Scale(scheme="category10")
                ),
                strokeDash=alt.condition(
                    alt.datum.IsPayoff,
                    alt.value([5, 5]),
                    alt.value([0]),  # Solid lines for BS prices
                ),
                tooltip=["Spot Price (S)", "Price", "Type"],
            )
            .properties(
                title=f"Black-Scholes Prices vs Payoffs (T={T:.2f}, σ={sigma:.2f}, r={r:.2f})",
                width=650,
                height=400,
            )
            .interactive()
        )
        return chart

    bs_chart = plot_bs_prices(
        bs_strike.value, bs_time.value, bs_vol.value, bs_rate.value
    )
    return bs_chart, plot_bs_prices


@app.cell
def _(bs_chart):
    bs_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice how the Black-Scholes price curve is smooth and stays strictly above the payoff line for American/European calls. European put options can dip below parity if rates are high, representing optimal early exercise (a property possessed by American, but not European options).
        The extra value is the **Time Value**, representing the probability that the option will move further into the money before expiration.

        ---

        ## 3. The Greeks (First-Order)

        ### Motivation: The "Risk" Perspective
        Once you accept the Black-Scholes framework, an option stops being just a directional bet on a stock going up or down. Instead, professional traders view an option as a **bundle of isolated risk exposures**. 
        
        Why does this matter? Because institutional players (like Market Makers) don't actually care if the stock goes up or down. Their goal is to supply liquidity while remaining perfectly **market neutral**. To do this, they need a mathematical way to decompose the option into exact risk factors, so they can explicitly hedge against each one individually using other options or the underlying stock.

        These isolated sensitivities are called the **"Greeks"**. Because we have a closed-form formula for $C$ and $P$, we calculate them simply by taking the partial derivatives of the Black-Scholes equation with respect to each parameter!

        ### Delta ($\Delta$)
        The rate of change of the option price with respect to the asset's spot price.
        - Call $\Delta = N(d_1)$ (Ranges from $0$ to $1$)
        - Put $\Delta = N(d_1) - 1$ (Ranges from $-1$ to $0$)
        - **Trading Context**: Delta is the most important Greek. It tells you your directional exposure. A portfolio Delta of +100 acts like owning 100 shares of the stock. Market makers dynamically hedge by buying/selling stock to keep their portfolio "Delta Neutral" ($\Delta = 0$).

        ### Vega ($\mathcal{V}$)
        The sensitivity to volatility ($\sigma$). Vega is strictly positive for both calls and puts.
        - $\mathcal{V} = S \sqrt{t} N'(d_1)$
        - **Trading Context**: If you buy an option, you are "Long Vega". This means you benefit if implied volatility rises. Earnings plays are heavily Vega-dependent; traders often short options before an earnings call to harvest the "Vega crush" (volatility dropping) right after the announcement.

        ### Theta ($\Theta$)
        The sensitivity to the passage of time. As expiration approaches, Time Value decays, so $\Theta$ is generally negative.
        - **Trading Context**: Options are depreciating assets! If you buy an option, you pay Theta (time is your enemy). If you sell an option, you collect Theta (time is your friend). The "Theta Gang" refers to traders who systematically sell options to generate income from time decay.
        
        ### Rho ($\rho$)
        The sensitivity to the interest rate ($r$). Calls generally have positive $\rho$; puts have negative $\rho$.
        - **Trading Context**: Rho is generally the least monitored First-Order Greek for retail day/swing traders because interest rates change slowly. It matters significantly more for institutional traders pricing LEAPS (Long-Term Equity Anticipation Securities) or fixed-income options.

        Let's visualize the First-Order Greeks!
        """
    )
    return


@app.cell
def _(np, norm):
    def bs_first_order_greeks(
        S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
    ) -> dict:
        """Returns Delta, Vega, Theta, Rho."""
        if T <= 0:
            return {"Delta": 0.0, "Vega": 0.0, "Theta": 0.0, "Rho": 0.0}

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        N_prime_d1 = norm.pdf(d1)
        vega = S * np.sqrt(T) * N_prime_d1

        if option_type == "call":
            delta = norm.cdf(d1)
            theta = -(S * N_prime_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(
                -r * T
            ) * norm.cdf(d2)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            delta = norm.cdf(d1) - 1
            theta = -(S * N_prime_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(
                -r * T
            ) * norm.cdf(-d2)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        return {
            "Delta": delta,
            "Vega": vega / 100.0,  # commonly reported per 1% change
            "Theta": theta / 365.0,  # commonly reported per day
            "Rho": rho / 100.0,  # commonly reported per 1% change
        }

    return (bs_first_order_greeks,)


@app.cell
def _(alt, bs_first_order_greeks, bs_rate, bs_strike, bs_time, bs_vol, np, pd):
    def plot_greeks(K: float, T: float, sigma: float, r: float) -> alt.Chart:
        # Generate spot prices
        S_vals = np.linspace(20, 180, 200)

        calls = [bs_first_order_greeks(S, K, T, r, sigma, "call") for S in S_vals]
        puts = [bs_first_order_greeks(S, K, T, r, sigma, "put") for S in S_vals]

        records = []
        for i, S in enumerate(S_vals):
            for greek in ["Delta", "Vega", "Theta", "Rho"]:
                records.append(
                    {
                        "Spot Price (S)": S,
                        "Value": calls[i][greek],
                        "Greek": greek,
                        "Type": "Call",
                    }
                )
                records.append(
                    {
                        "Spot Price (S)": S,
                        "Value": puts[i][greek],
                        "Greek": greek,
                        "Type": "Put",
                    }
                )

        df = pd.DataFrame.from_records(records)

        chart = (
            alt.Chart(df)
            .mark_line(size=2)
            .encode(
                x=alt.X("Spot Price (S):Q"),
                y=alt.Y("Value:Q", title="Greek Value"),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=["Call", "Put"], range=["#1f77b4", "#ff7f0e"]
                    ),
                ),
                tooltip=["Spot Price (S):Q", "Value:Q", "Type:N", "Greek:N"],
            )
            .properties(width=300, height=200)
            .facet(facet=alt.Facet("Greek:N", title=None), columns=2)
            .resolve_scale(y="independent")
            # Marimo correctly displays faceted charts without interactive(), but we can add it for individual subplot interactivity
        )
        return chart

    greek_chart = plot_greeks(
        bs_strike.value, bs_time.value, bs_vol.value, bs_rate.value
    )
    return greek_chart, plot_greeks


@app.cell
def _(bs_controls, greek_chart, mo):
    mo.vstack(
        [
            mo.md("### First-Order Greeks Explorer"),
            bs_controls,
            greek_chart,
        ]
    )
    return


@app.cell
def _(bs_controls):
    greeks_controls = bs_controls
    return (greeks_controls,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > **Try this:** Push **time to expiry → 0** and watch how Delta becomes a step function — deep ITM options have Δ≈1, deep OTM have Δ≈0. Near expiry, Gamma spikes dramatically near the strike (the option is most sensitive to moves when it's "on the fence").
        >
        > **Vega insight:** Drag volatility to extremes. Notice that Vega is always highest at-the-money — this is why ATM options are so sensitive to vol changes (vol trades). OTM options have lower Vega because much of the vol premium is already discounted.
        >
        > **Theta trap:** With very low time remaining, Theta accelerates sharply for ATM options. Option sellers collect Theta, but only if the stock stays still — any large move via Gamma will overwhelm the Theta income.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 4. Higher-Order Greeks

        Second-order Greeks measure the sensitivity of first-order Greeks!

        ### Gamma ($\Gamma$)
        Gamma is the rate of change of Delta with respect to the spot price: $\frac{\partial^2 C}{\partial S^2}$.
        It's identical for both calls and puts!

        $$\Gamma = \frac{N'(d_1)}{S \sigma \sqrt{t}}$$

        High Gamma means Delta is changing rapidly—a portfolio that was Delta-neutral a moment ago can suddenly have large directional exposure after a move. Gamma is highest for at-the-money options near expiration, which is why short-dated ATM options are the most dangerous to hold short.

        - **Trading Context**: Long Gamma positions profit from large moves (you get to re-hedge at favorable prices). Short Gamma positions profit from calm, range-bound markets but bleed on big moves. Gamma is what you're really buying or selling when you trade options—everything else is secondary.

        ### Vanna
        Vanna is the sensitivity of Delta to Volatility, or equivalently, the sensitivity of Vega to Spot Price.

        $$\text{Vanna} = \frac{\partial \Delta}{\partial \sigma} = \frac{\partial \mathcal{V}}{\partial S} = -\frac{N'(d_1) d_2}{\sigma}$$

        - **Trading Context**: Vanna is critical for understanding **skew dynamics**. When implied volatility rises, options gain (or lose) Delta based on their Vanna. A portfolio with negative Vanna will see its Delta become more negative in a vol spike—exactly when you don't want to be forced into extra selling. Dealers who are short large amounts of OTM puts can have significant negative Vanna exposure, causing them to sell the underlying as vol rises (a self-reinforcing crash dynamic).

        ### Volga (Vomma)
        Volga is the sensitivity of Vega to Volatility: $\frac{\partial^2 C}{\partial \sigma^2}$. It measures the convexity of an option's value with respect to implied volatility.

        $$\text{Volga} = \frac{\mathcal{V} d_1 d_2}{\sigma}$$

        - **Trading Context**: Positive Volga means your position benefits from *large moves in implied volatility*, in either direction. Far OTM options ("wings") have the highest Volga—they are cheap in absolute terms but highly convex to vol. Exotic vol traders specifically seek Volga exposure when they believe vol-of-vol is underpriced.

        ### Third-Order Greeks: Speed, Zomma, and Color
        Third-order Greeks measure how the second-order Greeks themselves change—useful for managing dynamic hedges that must remain stable across moves, vol shifts, and time.

        - **Speed** ($\frac{\partial \Gamma}{\partial S}$): How fast Gamma changes as the spot moves. High Speed means your hedge ratio (shares per option) changes quickly after a move—relevant when gamma-hedging in fast markets.
        - **Zomma** ($\frac{\partial \Gamma}{\partial \sigma}$): How Gamma changes with volatility. A trader short options with high Zomma faces compounding risk: a vol spike simultaneously increases their Gamma exposure AND forces more frequent (costly) re-hedging.
        - **Color** ($\frac{\partial \Gamma}{\partial t}$): The daily decay of Gamma. A position that is Gamma-neutral today will drift out of neutrality overnight—Color tells you by how much, and therefore how often you need to re-hedge.
        """
    )
    return


@app.cell
def _(np, norm):
    def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Returns Gamma (rate of change of Delta)."""
        if T <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    return (bs_gamma,)


@app.cell
def _(np, norm):
    def bs_higher_order_greeks(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> dict:
        """Returns Gamma, Vanna, Volga, Speed, Zomma, Color (Identical for Call and Put)."""
        if T <= 0:
            return {
                g: 0.0 for g in ["Gamma", "Vanna", "Volga", "Speed", "Zomma", "Color"]
            }

        # Prevent division by zero
        if sigma <= 0.0:
            sigma = 1e-4

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        N_prime_d1 = norm.pdf(d1)
        vega = S * np.sqrt(T) * N_prime_d1

        # Second Order
        gamma = N_prime_d1 / (S * sigma * np.sqrt(T))
        vanna = -N_prime_d1 * d2 / sigma
        volga = vega * d1 * d2 / sigma

        # Third Order
        speed = -gamma / S * (d1 / (sigma * np.sqrt(T)) + 1)
        zomma = gamma * (d1 * d2 - 1) / sigma
        color = (
            -gamma
            / (2 * T)
            * (1 + (2 * r * T - d2 * sigma * np.sqrt(T)) / (sigma * np.sqrt(T)) * d1)
        )

        return {
            "Gamma": gamma,
            "Vanna": vanna / 100.0,  # scaled for 1% change
            "Volga": volga / 10000.0,  # scaled for 1% change
            "Speed": speed,
            "Zomma": zomma / 100.0,
            "Color": color / 365.0,  # scaled per day
        }

    return (bs_higher_order_greeks,)


@app.cell
def _(alt, bs_gamma, bs_rate, bs_strike, bs_time, bs_vol, np, pd):
    def plot_gamma(K: float, T: float, sigma: float, r: float) -> alt.Chart:
        # Generate spot prices
        S_vals = np.linspace(20, 180, 200)
        gammas = [bs_gamma(S, K, T, r, sigma) for S in S_vals]

        df = pd.DataFrame({"Spot Price (S)": S_vals, "Gamma": gammas})

        chart = (
            alt.Chart(df)
            .mark_line(size=2.5, color="purple")
            .encode(
                x=alt.X("Spot Price (S):Q"),
                y=alt.Y("Gamma:Q"),
                tooltip=["Spot Price (S)", "Gamma"],
            )
            .properties(
                title=f"Gamma for Option (Call and Put are identical)",
                width=650,
                height=300,
            )
            .interactive()
        )
        return chart

    gamma_chart = plot_gamma(
        bs_strike.value, bs_time.value, bs_vol.value, bs_rate.value
    )
    return gamma_chart, plot_gamma


@app.cell
def _(alt, bs_higher_order_greeks, bs_rate, bs_strike, bs_time, bs_vol, np, pd):
    def plot_higher_order(K: float, T: float, sigma: float, r: float) -> alt.Chart:
        # Generate spot prices
        S_vals = np.linspace(20, 180, 200)

        records = []
        for S in S_vals:
            greeks = bs_higher_order_greeks(S, K, T, r, sigma)
            for greek, value in greeks.items():
                records.append({"Spot Price (S)": S, "Value": value, "Greek": greek})

        df = pd.DataFrame.from_records(records)

        # Using a custom color scheme to match Greek "vibes"
        color_scale = alt.Scale(
            domain=["Gamma", "Vanna", "Volga", "Speed", "Zomma", "Color"],
            range=[
                "#9c27b0",
                "#e91e63",
                "#00bcd4",
                "#4caf50",
                "#ff9800",
                "#795548",
            ],  # Purple, Pink, Cyan, Green, Orange, Brown
        )

        chart = (
            alt.Chart(df)
            .mark_line(size=2.5)
            .encode(
                x=alt.X("Spot Price (S):Q"),
                y=alt.Y("Value:Q", title="Value"),
                color=alt.Color("Greek:N", scale=color_scale),
                tooltip=["Spot Price (S):Q", "Value:Q", "Greek:N"],
            )
            .properties(width=400, height=250)
            .facet(facet=alt.Facet("Greek:N", title=None), columns=3)
            .resolve_scale(y="independent")
        )
        return chart

    higher_order_chart = plot_higher_order(
        bs_strike.value, bs_time.value, bs_vol.value, bs_rate.value
    )
    return higher_order_chart, plot_higher_order


@app.cell
def _(greeks_controls, higher_order_chart, mo):
    mo.vstack(
        [
            mo.md("### Higher-Order Greeks Explorer"),
            mo.md("*(Using the same parameters configured above)*"),
            greeks_controls,
            higher_order_chart,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        
        ## 5. Decay Schedules: Gamma & Time Decay (Theta)
        
        A critical concept in options trading is how the Greek profiles "decay" or change shape as expiration approaches ($t \to 0$).
        
        - **Gamma ($\Gamma$) Profile**: Gamma represents the acceleration of Delta as the spot price moves. For At-The-Money (ATM) options, Gamma explodes as expiration approaches (gamma pin risk). For Out-Of-The-Money (OTM) or In-The-Money (ITM) options, Gamma drops to essentially zero.
        - **Theta ($\Theta$) Profile**: Theta measures the time decay of the option's value. The time value decays fastest (steepest negative Theta) for ATM options in the final days before expiration. ITM and OTM options have much slower decay near expiration.
        
        ### The Core Trade-off: The Black-Scholes PDE
        Notice how the shapes of Gamma and -Theta are identical? This is not a coincidence! The entire Black-Scholes framework is built on a no-arbitrage Partial Differential Equation (PDE):
        
        $$ \Theta + rS\Delta + \frac{1}{2}\sigma^2 S^2 \Gamma = rV $$
        
        If a market maker "Delta Hedges" their portfolio so they have zero directional risk ($\Delta = 0$), the equation roughly simplifies to:
        
        $$ \Theta \approx -\frac{1}{2}\sigma^2 S^2 \Gamma $$
        
        This is the fundamental law of options trading: **To own Gamma (convexity), you must pay Theta (time decay).** And if you want to collect Theta (income), you must take on short Gamma risk (explosive losses if the stock moves fast).
        """
    )
    return


@app.cell
def _(
    alt,
    bs_first_order_greeks,
    bs_higher_order_greeks,
    bs_rate,
    bs_strike,
    bs_vol,
    np,
    pd,
):
    def plot_decay_schedules(K: float, sigma: float, r: float) -> alt.Chart:
        # User requested TTE as the X-axis for Gamma and -Theta decay
        T_vals = np.linspace(0.01, 1.0, 100)

        # Test across different Strike Moneyness profiles
        S_profiles = [
            (K * 0.8, "80% (Deep OTM)"),
            (K * 0.9, "90% (OTM)"),
            (K, "100% (ATM)"),
            (K * 1.1, "110% (ITM)"),
            (K * 1.2, "120% (Deep ITM)"),
        ]

        records = []
        for T in T_vals:
            for S, label in S_profiles:
                theta = bs_first_order_greeks(S, K, T, r, sigma, "call")["Theta"]
                gamma = bs_higher_order_greeks(S, K, T, r, sigma)["Gamma"]

                records.append(
                    {
                        "Time to Expiration (T)": T,
                        "Value": -theta,
                        "Greek": "-Theta (Time Decay)",
                        "Moneyness (Spot vs Strike)": label,
                    }
                )
                records.append(
                    {
                        "Time to Expiration (T)": T,
                        "Value": gamma,
                        "Greek": "Gamma (Acceleration)",
                        "Moneyness (Spot vs Strike)": label,
                    }
                )

        df = pd.DataFrame.from_records(records)

        chart = (
            alt.Chart(df)
            .mark_line(size=2.5)
            .encode(
                x=alt.X(
                    "Time to Expiration (T):Q",
                    scale=alt.Scale(reverse=True),
                    title="Time to Expiration (Years) ← Approaching 0",
                ),
                y=alt.Y("Value:Q"),
                color=alt.Color(
                    "Moneyness (Spot vs Strike):N", scale=alt.Scale(scheme="category10")
                ),
                tooltip=[
                    "Time to Expiration (T)",
                    "Value",
                    "Greek",
                    "Moneyness (Spot vs Strike)",
                ],
            )
            .properties(width=300, height=250)
            .facet(facet=alt.Facet("Greek:N", title=None), columns=2)
            .resolve_scale(y="independent")
        )
        return chart

    decay_chart = plot_decay_schedules(bs_strike.value, bs_vol.value, bs_rate.value)
    return decay_chart, plot_decay_schedules


@app.cell
def _(bs_controls, decay_chart, mo):
    mo.vstack(
        [
            mo.md("### Decay Schedules Explorer"),
            mo.md(
                "*(Using the Volatility and Risk-Free Rate configured above)*"
            ),
            bs_controls,
            decay_chart,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 5.5 Volatility Is Not Constant

        Black-Scholes bakes in one of its most problematic assumptions: **volatility is a single constant $\sigma$ for the entire life of the option**. Real markets break this in systematic, predictable ways:

        | Pattern | What Happens | Why It Matters |
        |---------|-------------|----------------|
        | **Intraday rhythm** | Vol spikes at the open (9:30–10 AM) and close (3:30–4 PM), quiet midday | Gamma hedging costs vary by time of day |
        | **Pre-event jumps** | IV surges before earnings, FOMC meetings, CPI releases | Option premiums are mis-priced if the event isn't modeled |
        | **Overnight calm** | Realized vol collapses when markets are closed | Theta decay is not uniform through the 24-hour clock |
        | **Vol clustering** | Turbulent days follow turbulent days (GARCH effects) | Future vol depends on the current regime, not just history |

        ### Effective Volatility

        If volatility is $\sigma_t$ at each moment in time, the Black-Scholes-equivalent "effective vol" for pricing purposes is the **root-mean-square** (not a simple average):

        $$\sigma_{\text{eff}} = \sqrt{\frac{1}{T}\int_0^T \sigma_t^2 \, dt}$$

        A quiet morning followed by an explosive afternoon **does not average linearly**—the squared vol path is what determines option prices. This means front-loading vol matters more than back-loading it.

        **Draw your own intraday vol schedule below** and see how the custom path changes the effective vol and the resulting option price compared to a flat assumption.
        """
    )
    return


@app.cell
def _(mo):
    _hour_labels = ["9–10","10–11","11–12","12–13","13–14","14–15","15–16","16–17"]
    _defaults     = [0.20,  0.35,  0.18,  0.14,  0.15,  0.20,  0.38,  0.28]
    vs0, vs1, vs2, vs3, vs4, vs5, vs6, vs7 = [
        mo.ui.slider(start=0.05, stop=0.80, value=_defaults[i], step=0.01, label=_hour_labels[i])
        for i in range(8)
    ]
    vol_sched_sliders = [vs0, vs1, vs2, vs3, vs4, vs5, vs6, vs7]
    vol_sched_controls = mo.vstack([
        mo.md("**Set hourly implied vol (drag sliders):**"),
        mo.hstack(vol_sched_sliders, wrap=True),
    ])
    return vol_sched_controls, vol_sched_sliders, vs0, vs1, vs2, vs3, vs4, vs5, vs6, vs7


@app.cell
def _(alt, bs_rate, bs_time, bs_vol, mo, norm, np, pd, vol_sched_controls, vs0, vs1, vs2, vs3, vs4, vs5, vs6, vs7):
    _sched = [vs0.value, vs1.value, vs2.value, vs3.value, vs4.value, vs5.value, vs6.value, vs7.value]
    _N = len(_sched)
    _rms = float(np.sqrt(np.mean(np.array(_sched) ** 2)))
    _flat = bs_vol.value
    _S0, _K0, _T0, _r0 = 100.0, 100.0, bs_time.value, bs_rate.value

    def _bs_price(sigma: float) -> float:
        if _T0 <= 0 or sigma <= 0:
            return max(_S0 - _K0, 0.0)
        d1 = (np.log(_S0 / _K0) + (_r0 + 0.5 * sigma ** 2) * _T0) / (sigma * np.sqrt(_T0))
        d2 = d1 - sigma * np.sqrt(_T0)
        return _S0 * norm.cdf(d1) - _K0 * np.exp(-_r0 * _T0) * norm.cdf(d2)

    _pf = _bs_price(_flat)
    _pr = _bs_price(_rms)
    _vega = (_bs_price(_flat + 0.01) - _pf) * 100

    # Bar chart: hourly vol schedule with RMS overlay
    _labels = ["9–10","10–11","11–12","12–13","13–14","14–15","15–16","16–17"]
    _df_bars = pd.DataFrame({"Hour": _labels, "Volatility": _sched})
    _df_rms = pd.DataFrame({"y": [_rms], "label": [f"σ_eff = {_rms*100:.1f}%"]})

    _bar_chart = (
        alt.Chart(_df_bars)
        .mark_bar(color="#1976d2", opacity=0.75)
        .encode(
            x=alt.X("Hour:N", sort=_labels, title="Trading Hour"),
            y=alt.Y("Volatility:Q", scale=alt.Scale(domain=[0, 0.85]), title="Implied Vol"),
            tooltip=["Hour", alt.Tooltip("Volatility:Q", format=".1%")],
        )
    )
    _rms_rule = (
        alt.Chart(_df_rms).mark_rule(color="#e53935", strokeDash=[6, 3], size=2).encode(y="y:Q")
    )
    _rms_text = (
        alt.Chart(_df_rms)
        .mark_text(align="right", dx=-4, dy=-6, color="#e53935", fontWeight="bold")
        .encode(y="y:Q", x=alt.value(556), text="label:N")
    )
    _vol_chart = (_bar_chart + _rms_rule + _rms_text).properties(
        width=550, height=180, title="Intraday Vol Schedule"
    )

    # Variance remaining chart: custom schedule vs flat (linear) decay
    # Shows how much variance is still "unexpired" as the trading day progresses
    _total_var = float(np.sum(np.array(_sched) ** 2))
    _cum_custom = [float(np.sum(np.array(_sched[i:]) ** 2) / max(_total_var, 1e-10)) for i in range(_N + 1)]
    _cum_flat = [(_N - i) / _N for i in range(_N + 1)]
    _hour_pts = list(range(_N + 1))

    _df_var = pd.DataFrame({
        "Hour": _hour_pts * 2,
        "Variance Remaining (%)": [v * 100 for v in _cum_custom] + [v * 100 for v in _cum_flat],
        "Schedule": ["Custom"] * (_N + 1) + ["Flat (BS Assumption)"] * (_N + 1),
    })
    _var_chart = (
        alt.Chart(_df_var)
        .mark_line(size=2.5, interpolate="step-after")
        .encode(
            x=alt.X("Hour:Q", title="Hours Elapsed", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Variance Remaining (%):Q", scale=alt.Scale(domain=[0, 105]), title="Variance Remaining (%)"),
            color=alt.Color(
                "Schedule:N",
                scale=alt.Scale(domain=["Custom", "Flat (BS Assumption)"], range=["#1976d2", "#e53935"]),
            ),
            strokeDash=alt.condition(
                alt.datum.Schedule == "Flat (BS Assumption)", alt.value([6, 3]), alt.value([0])
            ),
            tooltip=["Hour:Q", alt.Tooltip("Variance Remaining (%):Q", format=".1f"), "Schedule:N"],
        )
        .properties(width=550, height=180, title="Variance Remaining Over the Trading Day")
        .interactive()
    )

    _metrics = mo.md(
        f"""
| Metric | Flat Vol | Custom Schedule |
|--------|----------|-----------------|
| Volatility | **{_flat*100:.1f}%** | **{_rms*100:.1f}%** (RMS effective) |
| ATM Call Price | **${_pf:.3f}** | **${_pr:.3f}** |
| Price Difference | -- | **${_pr - _pf:+.3f}** |
| Vega (per 1% vol) | **${_vega:.4f}** | -- |
"""
    )
    mo.vstack([vol_sched_controls, _vol_chart, _var_chart, _metrics])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > **Try this:** Front-load the day with high opening vol (9–11am) then drop it. The variance-remaining curve will fall steeply early and flatten — meaning most of the "option value" is consumed by noon. Compare to a back-loaded schedule where the curve stays high late in the day.
        >
        > **Key insight:** The flat BS assumption (red dashed line) decays variance *linearly*. Any schedule that deviates from this creates a mismatch — front-loaded vol means the option is worth *more* early (buyers benefit from Gamma early), back-loaded vol keeps value alive longer.
        >
        > **Effective vol = RMS, not mean:** Notice that doubling the 9–10am bar has a disproportionate effect on the RMS effective vol. This is because variance adds quadratically — a 40% vol hour contributes 4× more variance than a 20% vol hour.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 6. The Implied Volatility Surface

        Here's the dirty secret of options trading: **Black-Scholes is wrong.** Its core assumption—that volatility $\sigma$ is a single constant that applies to all strikes and all expiries—is observably false. If it were true, every option on the same underlying would imply the same $\sigma$ when you back-solve from its market price. They don't. Not even close.

        ### Reversing the Arrow: Price → Implied Volatility

        Rather than abandoning Black-Scholes entirely, the market found a clever workaround: use it *in reverse*. Instead of plugging in $\sigma$ to get a price, take the market's observed price and numerically invert the formula to solve for the $\sigma$ that would have produced it. This is **Implied Volatility (IV)**—not a forecast, but a translation. Every option gets its own IV.

        This inversion is enormously useful even though the underlying model is misspecified:
        - **Common language**: Traders quote options in IV rather than dollars, making contracts across different strikes, expiries, and underlyings directly comparable.
        - **Model-free intuition**: IV encodes supply and demand, fear, and market expectations into a single normalized number.
        - **Risk signal**: The *shape* of IV across strikes and expiries reveals what the market collectively believes about tail risk, upcoming events, and the distribution of future returns.

        When you map IV across all strikes $K$ and times to expiration $T$, you get the **Volatility Surface**—a 3D fingerprint of market sentiment. Use the sliders to explore how each parameter shapes it.
        """
    )
    return


@app.cell
def _(mo):
    vol_skew = mo.ui.slider(
        start=-2.0, stop=2.0, value=-0.5, step=0.1, label="Skew (Slope)"
    )
    vol_convex = mo.ui.slider(
        start=0.0, stop=5.0, value=1.5, step=0.1, label="Convexity (Smile)"
    )
    vol_term = mo.ui.slider(
        start=-0.5, stop=0.5, value=0.1, step=0.05, label="Term Structure"
    )

    vol_surface_controls = mo.vstack(
        [
            mo.md("**Adjust Synthetic Surface Parameters:**"),
            mo.hstack([vol_skew, vol_convex, vol_term], wrap=True),
        ]
    )
    return vol_convex, vol_skew, vol_surface_controls, vol_term


@app.cell
def _(bs_strike, bs_vol, go, np, vol_convex, vol_skew, vol_term):
    def plot_vol_surface(atm_vol: float, skew: float, convex: float, term: float):
        S = bs_strike.value  # Assume ATM is centered around current strike slider
        K_vals = np.linspace(S * 0.5, S * 1.5, 50)
        T_vals = np.linspace(0.05, 2.0, 50)

        K_mesh, T_mesh = np.meshgrid(K_vals, T_vals)
        moneyness = (K_mesh / S) - 1.0

        # Synthetic IV Model: Smile/Skew * Term Structure
        smile = 1.0 + skew * moneyness + convex * (moneyness**2)
        term_struct = 1.0 + term / np.sqrt(T_mesh)

        iv_mesh = atm_vol * smile * term_struct
        iv_mesh = np.maximum(0.01, iv_mesh)  # IV cannot be negative

        fig = go.Figure(
            data=[
                go.Surface(
                    z=iv_mesh,
                    x=K_vals,
                    y=T_vals,
                    colorscale="Viridis",
                    colorbar=dict(title="Implied Vol"),
                )
            ]
        )

        fig.update_layout(
            title="Synthetic Implied Volatility Surface",
            autosize=False,
            width=600,
            height=500,
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis_title="Strike Price (K)",
                yaxis_title="Time to Exp. (Years)",
                zaxis_title="Implied Volatility",
                xaxis=dict(autorange="reversed"),
                dragmode="turntable",
            ),
        )
        return fig

    vol_surface_chart = plot_vol_surface(
        bs_vol.value, vol_skew.value, vol_convex.value, vol_term.value
    )
    return plot_vol_surface, vol_surface_chart


@app.cell
def _(mo, vol_surface_chart, vol_surface_controls):
    mo.vstack(
        [vol_surface_controls, mo.center(vol_surface_chart)]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### What You're Seeing

        - **Skew (Slope)**: A negative skew tilts the surface downward as you move from low strikes (OTM puts) toward high strikes (OTM calls). In equity markets, OTM puts are bid up by institutions buying crash protection—inflating their IV. The steeper the smirk, the more the market fears a left-tail event.
        - **Convexity (Smile)**: A positive convexity curves the surface upward symmetrically toward both wings. Far OTM options of all types command a premium for tail-risk exposure, creating the classic "smile" shape.
        - **Term Structure**: Controls how IV scales with time. A positive value elevates short-dated IV (typical during stress or before earnings). A negative value creates a "backwardated" surface where long-term uncertainty is priced higher.

        ### Why Skew and Smile Exist: Black-Scholes Violations

        The shape of the IV surface is a direct map of which Black-Scholes assumptions the market *knows to be false*:

        **Skew (the asymmetric smirk)** arises from two compounding forces:

        1. **The Leverage Effect**: When stock prices fall, a company's debt-to-equity ratio rises—the firm becomes *more* leveraged. Higher leverage means higher equity volatility. Stocks and volatility are negatively correlated ($\rho_{S,\sigma} < 0$), so OTM puts (which profit from crashes) are priced with extra IV to reflect this correlation.

        2. **Crash Risk / Fat Left Tail**: Black-Scholes assumes log-normal returns with thin tails. Real equity distributions are negatively skewed (crashes are more abrupt than rallies) and leptokurtic (fat tails). Institutions buying OTM puts as portfolio insurance bid up their prices, inflating their IV disproportionately.

        **Smile (symmetric wings)** arises from:

        1. **Jump Diffusion**: Black-Scholes assumes continuous price paths. Reality includes earnings gaps, biotech data readouts, and macro shocks—discrete jumps that standard GBM cannot capture. Far OTM options of all types gain value because a single jump can send the stock into the deep wings. This lifts both OTM put and OTM call IV symmetrically.

        2. **Stochastic Volatility**: Black-Scholes assumes $\sigma$ is constant. In reality, vol itself is random and mean-reverting (modeled by Heston, SABR, etc.). When $\sigma$ can vary, the distribution of future returns has fatter tails than log-normal on *both* sides. This fattening is symmetric, creating the smile shape.

        **Term Structure** reflects:
        - Short-term IV elevates near known events (earnings, Fed decisions).
        - Long-term IV reflects steady-state uncertainty (the long-run mean of stochastic vol).
        - Inverted term structure (near > far) is common during market stress when immediate fear outweighs long-run uncertainty.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 7. Simple Options Spreads

        #### Linearity: Portfolios of Options

        Option payoffs are **linear in position size** and **additive across legs**. This makes portfolio math straightforward:

        | Strategy | Construction | Payoff at Expiry |
        |----------|-------------|-----------------|
        | Long Straddle | Long Call + Long Put (same K) | $\max(S-K, 0) + \max(K-S, 0) = \|S-K\|$ |
        | Bull Call Spread | Long Call($K_1$) + Short Call($K_2$), $K_1 < K_2$ | Capped upside between $K_1$ and $K_2$ |
        | Butterfly | Long Call($K_1$) + Long Call($K_3$) + 2× Short Call($K_2$) | Peak at $K_2$, zero outside $[K_1, K_3]$ |

        This linearity is why traders build complex option strategies by **combining standard legs** — any desired payoff profile can be approximated by a portfolio of calls and puts (a consequence of the Breeden-Litzenberger theorem we'll see in Section 8).

        Because options are linear combinations of standard legs, **the Greeks are also additive**: the delta (or gamma, vega, theta) of a multi-leg portfolio is simply the sum of each leg's Greek weighted by its position size. This makes risk management of complex strategies tractable — you decompose the whole into known parts.

        #### Building Spreads

        Trading a naked option is expensive relative to the risk you're actually taking on. When you buy a call outright, the seller is on the hook for potentially unlimited losses if the stock rockets upward. To compensate for that open-ended exposure, they charge a hefty premium.

        **Spreads solve this by pairing options so neither side is ever exposed to catastrophic risk.** In a Bull Call Spread, you buy a call at a lower strike and simultaneously sell one at a higher strike. Your upside is capped at the spread width, but so is your counterparty's downside. Because the seller's maximum loss is now bounded, they don't need to charge nearly as much—and that savings passes directly to you as a lower net premium.

        This is why institutional traders almost always trade in spread structures rather than naked options: it dramatically reduces margin requirements and capital at risk on both sides of every trade.

        - **Debit Spreads** (e.g. Bull Call Spread): Pay a small net debit for capped directional exposure. Max loss = debit paid; max gain = spread width − debit.
        - **Credit Spreads** (e.g. Bear Call Spread): Collect premium upfront with a defined max loss. Time decay works in your favor. Max gain = credit received; max loss = spread width − credit.
        - **Iron Condor / Butterfly**: Combines two opposing spreads to collect premium from both sides. The wings on each side cap the loss; profits when the underlying stays range-bound.

        Use the builder below to explore preset structures, or switch to **Custom** to assemble your own strategy leg by leg.
        """
    )
    return


@app.cell
def _(mo):
    spread_strategy = mo.ui.dropdown(
        options=[
            "Long Call",
            "Long Put",
            "Long Straddle",
            "Long Strangle",
            "Bull Call Spread",
            "Bear Put Spread",
            "Iron Condor",
            "Iron Butterfly",
            "Custom",
        ],
        value="Bull Call Spread",
        label="Strategy",
    )
    spread_greek_select = mo.ui.dropdown(
        options=["Payoff (Intrinsic)", "Delta", "Gamma", "Vega", "Theta"],
        value="Payoff (Intrinsic)",
        label="Output Metric",
    )
    return spread_greek_select, spread_strategy


@app.cell
def _(mo):
    spread_center = mo.ui.slider(
        start=50, stop=150, value=100, step=1, label="Center Strike (ATM)"
    )
    spread_width = mo.ui.slider(
        start=1, stop=30, value=10, step=1, label="Spread Width (Points)"
    )
    return spread_center, spread_width


@app.cell
def _(mo):
    get_custom_legs, set_custom_legs = mo.state(
        [
            {"strike": 95, "type": "Call", "position": "Long", "qty": 1},
            {"strike": 105, "type": "Call", "position": "Short", "qty": 1},
        ]
    )
    add_leg_btn = mo.ui.button(
        label="+ Add Leg",
        on_click=lambda _: set_custom_legs(
            get_custom_legs()
            + [{"strike": 100, "type": "Call", "position": "Long", "qty": 1}]
        ),
    )
    return add_leg_btn, get_custom_legs, set_custom_legs


@app.cell
def _(add_leg_btn, get_custom_legs, mo, set_custom_legs):
    _legs = get_custom_legs()
    _strike_ctrls = [
        mo.ui.number(value=leg["strike"], start=1, stop=500, step=1)
        for leg in _legs
    ]
    _type_ctrls = [
        mo.ui.dropdown(["Call", "Put"], value=leg["type"])
        for leg in _legs
    ]
    _pos_ctrls = [
        mo.ui.dropdown(["Long", "Short"], value=leg["position"])
        for leg in _legs
    ]
    _qty_ctrls = [
        mo.ui.number(value=leg["qty"], start=1, stop=10, step=1)
        for leg in _legs
    ]
    _remove_btns = [
        mo.ui.button(
            label="✕",
            on_click=lambda _, i=i: set_custom_legs(
                [l for j, l in enumerate(get_custom_legs()) if j != i]
            ),
        )
        for i in range(len(_legs))
    ]

    _header = mo.hstack(
        [
            mo.md("&nbsp;&nbsp;&nbsp;"),
            mo.md("**Strike**"),
            mo.md("**Type**"),
            mo.md("**Position**"),
            mo.md("**Qty**"),
            mo.md(""),
        ],
        align="center",
    )
    _rows = [
        mo.hstack(
            [
                mo.md(f"**{i + 1}.**"),
                _strike_ctrls[i],
                _type_ctrls[i],
                _pos_ctrls[i],
                _qty_ctrls[i],
                _remove_btns[i],
            ],
            align="center",
        )
        for i in range(len(_legs))
    ]
    custom_leg_table = mo.vstack([_header, *_rows, add_leg_btn])
    # Export controls as a list of tuples so the chart cell can read .value directly
    leg_controls = list(zip(_strike_ctrls, _type_ctrls, _pos_ctrls, _qty_ctrls))
    return custom_leg_table, leg_controls


@app.cell
def _(
    alt,
    bs_first_order_greeks,
    bs_higher_order_greeks,
    bs_rate,
    bs_time,
    bs_vol,
    leg_controls,
    np,
    pd,
    spread_center,
    spread_greek_select,
    spread_strategy,
    spread_width,
):
    def get_strategy_legs(strategy: str, S: float, W: float) -> list[dict]:
        legs = []
        if strategy == "Long Call":
            legs.append({"type": "Call", "position": "Long", "strike": S, "qty": 1, "active": True})
        elif strategy == "Long Put":
            legs.append({"type": "Put", "position": "Long", "strike": S, "qty": 1, "active": True})
        elif strategy == "Long Straddle":
            legs.append({"type": "Call", "position": "Long", "strike": S, "qty": 1, "active": True})
            legs.append({"type": "Put", "position": "Long", "strike": S, "qty": 1, "active": True})
        elif strategy == "Long Strangle":
            legs.append({"type": "Call", "position": "Long", "strike": S + W, "qty": 1, "active": True})
            legs.append({"type": "Put", "position": "Long", "strike": S - W, "qty": 1, "active": True})
        elif strategy == "Bull Call Spread":
            legs.append({"type": "Call", "position": "Long", "strike": S, "qty": 1, "active": True})
            legs.append({"type": "Call", "position": "Short", "strike": S + W, "qty": 1, "active": True})
        elif strategy == "Bear Put Spread":
            legs.append({"type": "Put", "position": "Long", "strike": S, "qty": 1, "active": True})
            legs.append({"type": "Put", "position": "Short", "strike": S - W, "qty": 1, "active": True})
        elif strategy == "Iron Condor":
            legs.append({"type": "Put", "position": "Long", "strike": S - 2 * W, "qty": 1, "active": True})
            legs.append({"type": "Put", "position": "Short", "strike": S - W, "qty": 1, "active": True})
            legs.append({"type": "Call", "position": "Short", "strike": S + W, "qty": 1, "active": True})
            legs.append({"type": "Call", "position": "Long", "strike": S + 2 * W, "qty": 1, "active": True})
        elif strategy == "Iron Butterfly":
            legs.append({"type": "Put", "position": "Long", "strike": S - W, "qty": 1, "active": True})
            legs.append({"type": "Put", "position": "Short", "strike": S, "qty": 1, "active": True})
            legs.append({"type": "Call", "position": "Short", "strike": S, "qty": 1, "active": True})
            legs.append({"type": "Call", "position": "Long", "strike": S + W, "qty": 1, "active": True})
        return legs

    def plot_spread_from_legs(
        legs: list[dict],
        S0: float,
        plot_w: float,
        metric: str,
        T: float,
        sigma: float,
        r: float,
        title: str = "Options Strategy",
    ) -> alt.Chart:
        if not legs:
            return (
                alt.Chart(pd.DataFrame({"x": [S0], "y": [0]}))
                .mark_text(text="No legs defined")
                .encode(x="x:Q", y="y:Q")
                .properties(width=600, height=350)
            )

        plot_width = max(plot_w, 10)
        S_lo = max(1, S0 - plot_width * 1.5)
        S_hi = S0 + plot_width * 1.5
        S_vals = np.linspace(S_lo, S_hi, 300)

        def get_payoff(S_arr, K, type_str, position, qty, premium=0.0):
            intrinsic = np.maximum(S_arr - K, 0) * qty if type_str == "Call" else np.maximum(K - S_arr, 0) * qty
            return intrinsic - premium if position == "Long" else premium - intrinsic

        def get_dummy_premium(K):
            return max(0, 10.0 - abs(K - S0) * 0.2)

        records = []
        total_values = np.zeros_like(S_vals)

        for i, leg in enumerate(legs):
            qty = leg["qty"]
            mult = qty if leg["position"] == "Long" else -qty
            name = f"Leg {i + 1}: {leg['position']} {qty}x {leg['strike']} {leg['type']}"
            leg_values = []
            for S in S_vals:
                if metric == "Payoff (Intrinsic)":
                    prem = get_dummy_premium(leg["strike"]) * qty
                    val = get_payoff(np.array([S]), leg["strike"], leg["type"], leg["position"], qty, prem)[0]
                else:
                    opt_type = leg["type"].lower()
                    if metric in ["Delta", "Vega", "Theta"]:
                        greeks = bs_first_order_greeks(S, leg["strike"], T, r, sigma, opt_type)
                        val = greeks[metric] * mult
                    elif metric == "Gamma":
                        greeks = bs_higher_order_greeks(S, leg["strike"], T, r, sigma)
                        val = greeks[metric] * mult
                    else:
                        val = 0.0
                leg_values.append(val)
            leg_values = np.array(leg_values)
            total_values += leg_values
            for S, val in zip(S_vals, leg_values):
                records.append({"Spot Price (S)": S, "Value": val, "Component": name})

        total_label = f"Total {metric}"
        for S, val in zip(S_vals, total_values):
            records.append({"Spot Price (S)": S, "Value": val, "Component": total_label})

        df = pd.DataFrame.from_records(records)
        components = [c for c in df["Component"].unique() if c != total_label]
        domain = components + [total_label]
        range_colors = ["#9e9e9e", "#bdbdbd", "#757575", "#616161"][: len(components)] + ["#4caf50"]

        chart = (
            alt.Chart(df)
            .mark_line(size=3, clip=True)
            .encode(
                x=alt.X("Spot Price (S):Q", scale=alt.Scale(domain=[S_lo, S_hi])),
                y=alt.Y("Value:Q", title=metric),
                color=alt.Color("Component:N", scale=alt.Scale(domain=domain, range=range_colors)),
                strokeDash=alt.condition(
                    alt.datum.Component == total_label,
                    alt.value([0]),
                    alt.value([5, 5]),
                ),
                tooltip=["Spot Price (S)", "Value", "Component"],
            )
            .properties(title=f"{title}: {metric}", width=600, height=350)
        )
        chart += (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color="black", strokeDash=[2, 2])
            .encode(y="y:Q")
        )
        return chart

    if spread_strategy.value == "Custom":
        _active_legs = [
            {"strike": int(c[0].value), "type": c[1].value, "position": c[2].value, "qty": int(c[3].value)}
            for c in leg_controls
        ]
        _strikes = [l["strike"] for l in _active_legs] if _active_legs else [100]
        _S0 = sum(_strikes) / len(_strikes)
        _W = max((max(_strikes) - min(_strikes)) / 2, 10) if len(_strikes) > 1 else 10
        _title = "Custom Strategy"
    else:
        _raw = get_strategy_legs(spread_strategy.value, spread_center.value, spread_width.value)
        _active_legs = [l for l in _raw if l["active"]]
        _S0 = spread_center.value
        _W = spread_width.value
        _title = spread_strategy.value

    spread_chart = plot_spread_from_legs(
        _active_legs, _S0, _W, spread_greek_select.value,
        bs_time.value, bs_vol.value, bs_rate.value, _title,
    )
    return get_strategy_legs, plot_spread_from_legs, spread_chart


@app.cell
def _(
    custom_leg_table,
    get_strategy_legs,
    mo,
    pd,
    spread_center,
    spread_chart,
    spread_greek_select,
    spread_strategy,
    spread_width,
):
    if spread_strategy.value == "Custom":
        _param_ui = custom_leg_table
        _summary = mo.md("")
    else:
        _param_ui = mo.hstack([spread_center, spread_width])
        _active = [
            l for l in get_strategy_legs(spread_strategy.value, spread_center.value, spread_width.value)
            if l["active"]
        ]
        _legs_df = pd.DataFrame(
            [{"Position": l["position"], "Qty": l["qty"], "Type": l["type"], "Strike": l["strike"]} for l in _active]
        )
        _summary = mo.vstack([mo.md("#### Strategy Legs"), mo.ui.table(_legs_df, selection=None)])

    mo.vstack(
        [
            mo.hstack([spread_strategy, spread_greek_select]),
            _param_ui,
            _summary,
            spread_chart,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > **Try this:** Switch to **Delta** view — notice how a Bull Call Spread's delta is bounded between 0 and 1, never exceeding either leg's individual delta. Compare to a naked Long Call: delta ramps smoothly from 0 to 1.
        >
        > **Iron Condor hint:** Switch to Iron Condor and check both Payoff and Theta views. The Theta profile is the mirror image of Gamma — you collect time decay in the middle zone but pay for it through Gamma exposure on big moves.
        >
        > **Custom strategy:** Build a **Ratio Spread** (1 long call at K, 2 short calls at K+10). In Payoff view, notice how the strategy initially profits from a moderate move but suffers on a large rally — you've sold more options than you bought.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Part 2: Intermediate Options Concepts

        ## 8. Implied Distributions (Breeden-Litzenberger)
        
        The Black-Scholes model assumes the underlying asset follows a log-normal distribution. But what if the market thinks otherwise?
        
        The **Breeden-Litzenberger** theorem provides a powerful way to extract the market's *actual* implied probability distribution directly from option prices! 
        
        The theorem states that the second derivative of the Call option price with respect to the strike price ($K$) is directly proportional to the risk-neutral probability density function (PDF) of the asset at expiration:
        
        $$ f(K) = e^{rT} \frac{\partial^2 C}{\partial K^2} $$
        
        Since $\frac{\partial C}{\partial K}$ is the negative of the probability of the option expiring ITM, the second derivative gives the PDF.
        In practice, we approximate this using a **Butterfly Spread**: buying a call at $K-\Delta K$, selling two calls at $K$, and buying a call at $K+\Delta K$. The cost of this spread is essentially the market's price for the asset landing exactly at $K$.
        """
    )
    return


@app.cell
def _(mo):
    bl_vol = mo.ui.slider(
        start=0.05, stop=1.0, value=0.2, step=0.01, label="ATM Volatility"
    )
    bl_skew = mo.ui.slider(start=-5.0, stop=5.0, value=-0.5, step=0.1, label="Skew")
    bl_kurt = mo.ui.slider(
        start=0.0, stop=15.0, value=1.0, step=0.25, label="Kurtosis (Smile)"
    )

    bl_controls = mo.vstack(
        [
            mo.md("**Adjust Synthetic Market Profile:**"),
            mo.hstack([bl_vol, bl_skew, bl_kurt], wrap=True),
        ]
    )
    # bl_strike kept as a fixed reference; S=100 is the invariant center
    bl_strike = None
    return bl_controls, bl_kurt, bl_skew, bl_strike, bl_vol


@app.cell
def _(alt, bs_rate, bs_time, norm, np, pd, bl_kurt, bl_skew, bl_vol):
    from scipy.stats import norminvgauss as _nig
    from scipy.optimize import fsolve as _fsolve
    import warnings as _warnings

    def _fit_nig(target_skew: float, target_kurt: float) -> tuple[float, float]:
        """Fit NIG(a, b) parameters to match target skewness and excess kurtosis.
        NIG always has non-negative PDF and b=0 gives exactly symmetric distribution."""
        _kurt = max(target_kurt, 0.1)

        def _resid(params: np.ndarray) -> list:
            log_a, b_ratio = params
            a = float(np.exp(log_a))
            b = float(np.tanh(b_ratio) * a * 0.98)
            try:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    s, k = _nig.stats(a, b, moments='sk')
                return [float(s) - target_skew, float(k) - _kurt]
            except Exception:
                return [1e6, 1e6]

        # For b=0 symmetric NIG: excess_kurt ≈ 3/a → a = 3/excess_kurt
        a_init = max(3.0 / _kurt, 0.05)
        x0 = [np.log(a_init), np.clip(target_skew * 0.15, -0.95, 0.95)]
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            sol = _fsolve(_resid, x0)
        a_sol = float(np.exp(sol[0]))
        b_sol = float(np.tanh(sol[1]) * a_sol * 0.98)
        return a_sol, b_sol

    def plot_implied_distribution(
        atm_vol: float, skew: float, excess_kurt: float, T: float, r: float
    ):
        """Compute risk-neutral PDF using Normal Inverse Gaussian (NIG) distribution.
        NIG parameters are fitted numerically to match (skew, excess_kurt) exactly.
        Guarantees: non-negative PDF everywhere; b=0 → exactly symmetric with skew=0."""
        S = 100.0
        K_plot = np.linspace(S * 0.35, S * 2.0, 600)
        dK = K_plot[1] - K_plot[0]

        mu_log = (r - 0.5 * atm_vol ** 2) * T
        sig_log = atm_vol * np.sqrt(T)

        # Log-normal baseline
        x = np.log(K_plot / S)
        z = (x - mu_log) / sig_log
        pdf_lognormal = norm.pdf(z) / (K_plot * sig_log)

        if abs(excess_kurt) < 0.05 and abs(skew) < 0.05:
            pdf_implied = pdf_lognormal.copy()
        else:
            # Fit NIG(a, b) to match target skewness and excess kurtosis
            a, b = _fit_nig(skew, excess_kurt)
            nig_dist = _nig(a, b)
            nig_mean = float(nig_dist.mean())
            nig_std = float(nig_dist.std())
            # Standardize: u ~ N(0,1) mapped to Z = u * nig_std + nig_mean ~ NIG(a,b)
            # so pdf_u(u) = pdf_Z(u * nig_std + nig_mean) * nig_std
            z_nig = z * nig_std + nig_mean
            pdf_u = _nig.pdf(z_nig, a, b) * nig_std
            pdf_implied = pdf_u / (K_plot * sig_log)

        # Normalize to integrate to 1
        pdf_lognormal = pdf_lognormal / max(np.sum(pdf_lognormal) * dK, 1e-10)
        pdf_implied = pdf_implied / max(np.sum(pdf_implied) * dK, 1e-10)

        df = pd.DataFrame({
            "Strike (K)": np.tile(K_plot, 2),
            "Probability Density": np.concatenate([pdf_implied, pdf_lognormal]),
            "Distribution": ["Market Implied"] * len(K_plot) + ["Log-Normal (BSM ATM)"] * len(K_plot),
        })

        pdf_chart = (
            alt.Chart(df)
            .mark_line(size=2.5)
            .encode(
                x=alt.X("Strike (K):Q", title="Strike (K)"),
                y=alt.Y("Probability Density:Q", title="Probability Density"),
                color=alt.Color(
                    "Distribution:N",
                    scale=alt.Scale(
                        domain=["Market Implied", "Log-Normal (BSM ATM)"],
                        range=["#e91e63", "#9e9e9e"],
                    ),
                ),
                strokeDash=alt.condition(
                    alt.datum.Distribution == "Log-Normal (BSM ATM)",
                    alt.value([5, 5]),
                    alt.value([0]),
                ),
                tooltip=["Strike (K)", alt.Tooltip("Probability Density:Q", format=".5f"), "Distribution"],
            )
            .properties(title="Implied Probability Distribution vs Log-Normal", width=600, height=260)
            .interactive()
        )

        # Ratio chart: where does the implied distribution have MORE vs LESS probability than lognormal?
        # Values > 1 = more probability (tails + peak), values < 1 = less (shoulders depleted)
        _ratio = np.where(pdf_lognormal > 1e-10, pdf_implied / pdf_lognormal, 1.0)
        df_ratio = pd.DataFrame({
            "Strike (K)": K_plot,
            "Implied / Log-Normal": _ratio,
        })
        _one_line = pd.DataFrame({"y": [1.0]})
        ratio_chart = (
            alt.Chart(df_ratio)
            .mark_line(size=2.5, color="#e91e63")
            .encode(
                x=alt.X("Strike (K):Q", title="Strike (K)"),
                y=alt.Y("Implied / Log-Normal:Q", title="Implied / Log-Normal ratio"),
                tooltip=["Strike (K)", alt.Tooltip("Implied / Log-Normal:Q", format=".3f")],
            )
        ) + (
            alt.Chart(_one_line)
            .mark_rule(color="#9e9e9e", strokeDash=[4, 3])
            .encode(y="y:Q")
        )
        ratio_chart = ratio_chart.properties(
            title="Probability Ratio: Implied ÷ Log-Normal  (>1 = more, <1 = less than BSM)",
            width=600, height=180,
        ).interactive()

        return alt.vconcat(pdf_chart, ratio_chart).resolve_scale(x="shared")

    bl_chart = plot_implied_distribution(
        bl_vol.value, bl_skew.value, bl_kurt.value, bs_time.value, bs_rate.value,
    )
    return bl_chart, plot_implied_distribution


@app.cell
def _(mo, bl_chart, bl_controls):
    mo.vstack([bl_controls, bl_chart])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 🔍 Deep Dive: Skew, Kurtosis, and Wings
        
        *If you scroll back up to the Breeden-Litzenberger section, you can see how changing the Volatility Smile alters the implied distribution!*
        
        - **Skew**: Downside puts are often bid up by participants buying insurance. This creates negative Skew. Notice how decreasing the Skew slider fattens the left tail of the distribution, implying a higher market-perceived probability of a crash than a standard log-normal distribution.
        - **Kurtosis**: Far Out-Of-The-Money (OTM) options (the "Wings") often trade at a premium due to tail risks. Increasing Kurtosis pulls probability away from the center (ATM) and pumps it into *both* the left and right tails, creating a "fat-tailed" distribution.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        
        ## 9. Gamma Hedging Simulator
        
        Options market makers don't generally bet on whether a stock goes up or down. Instead, they try to remain **Delta Neutral** ($\Delta = 0$). By constantly buying or selling the underlying stock to offset the Delta of the options they've sold, they aim to capture the edge in the option's premium without taking directional risk.
        
        This process is called **Gamma Hedging** (or Delta Hedging). However, because Gamma continuously changes Delta as the stock moves, a delta-neutral portfolio doesn't stay neutral for long! The hedger must frequently rebalance.
        
        But rebalancing has costs:
        1. **Transaction Fees**
        2. **Bid-Ask Spread (Slippage)**
        
        Let's simulate the P&L of hedging a short ATM Call option position over its lifetime under different conditions.
        """
    )
    return


@app.cell
def _(mo):
    gh_vol = mo.ui.slider(
        start=0.1, stop=0.8, value=0.2, step=0.05, label="Underlying Volatility"
    )
    gh_freq = mo.ui.dropdown(
        options={"Immediate (~10 min)": 2500, "Hourly": 8760, "Daily": 252, "Weekly": 52, "Monthly": 12},
        value="Daily",
        label="Hedging Frequency",
    )
    gh_fee = mo.ui.slider(
        start=0.0, stop=1.0, value=0.05, step=0.05, label="Cost per Share ($)"
    )
    gh_seed = mo.ui.run_button(label="Simulate New Path!")

    gh_controls = mo.vstack(
        [
            mo.md("**Simulation Parameters:**"),
            mo.hstack([gh_vol, gh_freq, gh_fee, gh_seed], wrap=True),
        ]
    )
    return gh_controls, gh_fee, gh_freq, gh_seed, gh_vol


@app.cell
def _(alt, bs_first_order_greeks, np, pd, gh_fee, gh_freq, gh_seed, gh_vol):
    def simulate_gamma_hedge(vol: float, steps: int, fee: float, seed: int):
        np.random.seed(seed)

        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, vol
        dt = T / steps
        multiplier = 100

        from scipy.stats import norm as _norm

        def bs_price(S, K, T, r, sig):
            if T <= 0:
                return max(S - K, 0.0)
            d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
            d2 = d1 - sig * np.sqrt(T)
            return S * _norm.cdf(d1) - K * np.exp(-r * T) * _norm.cdf(d2)

        initial_premium = bs_price(S0, K, T, r, sigma) * multiplier

        S_path = np.zeros(steps + 1)
        S_path[0] = S0
        for i in range(1, steps + 1):
            Z = np.random.normal(0, 1)
            S_path[i] = S_path[i - 1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )

        shares_held = 0.0
        cash_account = initial_premium
        total_fees = 0.0

        records = []
        delta_records = []

        for i in range(steps):
            t = i * dt
            time_to_exp = max(0.001, T - t)
            S = S_path[i]

            opt_delta = (
                -bs_first_order_greeks(S, K, time_to_exp, r, sigma, "call")["Delta"]
                * multiplier
            )
            # Net delta BEFORE rebalancing — this is our live exposure
            net_delta_pre = shares_held + opt_delta
            delta_records.append({"t": float(t), "delta": float(net_delta_pre)})

            target_shares = -opt_delta
            trade_qty = target_shares - shares_held
            transaction_cost = abs(trade_qty) * fee

            cash_account -= trade_qty * S
            cash_account -= transaction_cost
            total_fees += transaction_cost
            shares_held = target_shares

            opt_value = -bs_price(S, K, time_to_exp, r, sigma) * multiplier
            portfolio_value = cash_account + shares_held * S + opt_value
            records.append({
                "Time (Years)": t,
                "Spot (S)": S,
                "Shares Held": shares_held,
                "Cash": cash_account,
                "Portfolio Value": portfolio_value,
            })

        S_final = S_path[-1]
        opt_payoff = -max(S_final - K, 0) * multiplier
        final_value = cash_account + shares_held * S_final + opt_payoff
        delta_records.append({"t": float(T), "delta": 0.0})
        records.append({
            "Time (Years)": T, "Spot (S)": S_final, "Shares Held": shares_held,
            "Cash": cash_account, "Portfolio Value": final_value,
        })

        df_sim = pd.DataFrame.from_records(records)

        chart_spot = (
            alt.Chart(df_sim)
            .mark_line(color="#ff9800", size=2)
            .encode(
                x=alt.X("Time (Years):Q"),
                y=alt.Y("Spot (S):Q", scale=alt.Scale(zero=False)),
                tooltip=["Time (Years)", "Spot (S)"],
            )
            .properties(title="Asset Price Path (GBM)", width=580, height=180)
        )
        chart_pnl = (
            alt.Chart(df_sim)
            .mark_line(color="#4caf50", size=2)
            .encode(
                x=alt.X("Time (Years):Q"),
                y=alt.Y("Portfolio Value:Q", scale=alt.Scale(zero=False)),
                tooltip=["Time (Years)", "Portfolio Value", "Cash", "Shares Held"],
            )
            .properties(
                title=f"Delta-Hedged P&L  (fees paid: ${total_fees:.2f}  |  premium received: ${initial_premium:.2f})",
                width=580, height=180,
            )
        )
        charts = alt.vconcat(chart_spot, chart_pnl).resolve_scale(y="independent")
        return charts, delta_records, total_fees, initial_premium


    gh_charts, gh_delta_records, gh_total_fees, gh_premium = simulate_gamma_hedge(
        gh_vol.value, int(gh_freq.value), gh_fee.value, gh_seed.value
    )
    return gh_charts, gh_delta_records, gh_premium, gh_total_fees, simulate_gamma_hedge


@app.cell
def _(alt, gh_delta_records, pd):
    _df_d = pd.DataFrame(gh_delta_records)
    _line = (
        alt.Chart(_df_d)
        .mark_line(color="#2196f3", size=2)
        .encode(
            x=alt.X("t:Q", title="Time (Years)"),
            y=alt.Y("delta:Q", title="Net \u0394 (shares)"),
            tooltip=[alt.Tooltip("t:Q", format=".3f"), alt.Tooltip("delta:Q", format=".2f")],
        )
        .properties(width=600, height=180, title="Delta Exposure Between Rebalances")
    )
    _zero = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="#bbb", strokeDash=[4, 3])
        .encode(y="y:Q")
    )
    gh_delta_chart = (_line + _zero).interactive()
    return (gh_delta_chart,)


@app.cell
def _(alt, gh_fee, gh_freq, gh_vol, np, pd):
    def _sim_pnl(vol: float, steps: int, fee_per_share: float, seed: int) -> float:
        from scipy.stats import norm as _n
        np.random.seed(seed)
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, vol
        dt = T / steps

        def _delta(S: float, t_rem: float) -> float:
            if t_rem <= 1e-6:
                return 1.0 if S > K else 0.0
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t_rem) / (sigma * np.sqrt(t_rem))
            return float(_n.cdf(d1))

        def _price(S: float, t_rem: float) -> float:
            if t_rem <= 1e-6:
                return max(S - K, 0.0)
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t_rem) / (sigma * np.sqrt(t_rem))
            d2 = d1 - sigma * np.sqrt(t_rem)
            return S * _n.cdf(d1) - K * np.exp(-r * t_rem) * _n.cdf(d2)

        cash = _price(S0, T)   # premium received from selling call
        shares = 0.0
        S = S0
        for i in range(steps):
            t_rem = max(1e-6, T - i * dt)
            target = _delta(S, t_rem)
            trade = target - shares
            cash -= trade * S + abs(trade) * fee_per_share
            shares = target
            Z = np.random.normal()
            S = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

        # Close out: sell stock, pay option payoff
        return cash + shares * S - max(S - K, 0.0)

    _n_sims = 300
    _pnls = np.array([
        _sim_pnl(gh_vol.value, int(gh_freq.value), gh_fee.value, s)
        for s in range(_n_sims)
    ])
    _mu, _sd = _pnls.mean(), _pnls.std()
    _sharpe = _mu / (_sd + 1e-10)

    _df_sl = pd.DataFrame({"PnL": _pnls})
    _hist = (
        alt.Chart(_df_sl)
        .mark_bar(color="#4caf50", opacity=0.75)
        .encode(
            x=alt.X("PnL:Q", bin=alt.Bin(maxbins=35), title="Final Hedge P&L ($, per option)"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=["count()"],
        )
        .properties(
            width=580, height=200,
            title=f"Slippage P&L Distribution ({_n_sims} paths)  |  Mean={_mu:.2f}  Std={_sd:.2f}  Sharpe={_sharpe:.2f}",
        )
    )
    _zero_line = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color="#e53935", strokeDash=[4, 3], size=2)
        .encode(x="x:Q")
    )
    gh_slippage_chart = (_hist + _zero_line).interactive()
    return gh_slippage_chart, _sim_pnl


@app.cell
def _(mo, gh_charts, gh_controls, gh_delta_chart, gh_slippage_chart):
    mo.vstack([
        gh_controls,
        gh_delta_chart,
        gh_charts,
        mo.md("""
**Slippage & Sharpe Reality Check**

Under perfect continuous hedging with zero fees, delta hedging an ATM call has **zero expected P&L** — you just break even collecting Gamma income against Theta decay.
Real markets impose two costs:
- **Transaction fees** — each rebalance costs money, slippage turns a zero-EV strategy negative
- **Discrete hedging error** — you can only rebalance finitely often; the residual Gamma exposure creates a P&L distribution with non-zero variance even with zero fees

The histogram below shows the *realized* P&L distribution across {_n_sims} simulated paths. Notice:
- With low fees + high frequency: mean ≈ 0 but variance is low (frequent hedging reduces Gamma risk)
- With higher fees: distribution shifts negative (each rebalance bleeds money)
- **Sharpe ratio** is a better metric than P&L alone — a strategy can have negative mean P&L but still be "good" if variance is very low
""".replace("{_n_sims}", "300")),
        gh_slippage_chart,
    ])
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        
        ## 10. Beyond Black-Scholes: Alternative Models
        
        While Black-Scholes is the bedrock of options theory, its assumption of continuous logging pricing (Geometric Brownian Motion) breaks down in the real world. Real asset prices gap up and down overnight, and volatility is not constant.
        
        ### 10a. Merton Jump-Diffusion Model
        
        The Merton Model adds a **Poisson "jump" process** on top of the continuous random walk to account for sudden, discontinuous price gaps (like earnings shocks or macroeconomic news). 
        
        A standard GBM stock path looks like:
        $$\frac{dS}{S} = \mu \, dt + \sigma \, dW$$

        Merton adds a random jump term $J \, dN$:
        $$\frac{dS}{S} = (\mu - \lambda k) \, dt + \sigma \, dW + (J - 1) \, dQ$$
        
        Where:
        - $dQ$ is a Poisson process with jump intensity $\lambda$ (average jumps per year)
        - $J$ is the random jump size (usually log-normally distributed)
        
        Let's generate some Monte Carlo paths to see the difference!
        """
    )
    return


@app.cell
def _(mo):
    mjd_lambda = mo.ui.slider(
        start=0.0, stop=20.0, value=5.0, step=1.0, label="Jump Intensity (λ)"
    )
    mjd_mu_j = mo.ui.slider(
        start=-0.5, stop=-0.01, value=-0.1, step=0.01, label="Avg Jump Shock (μ_j, log-return)"
    )
    mjd_sig_j = mo.ui.slider(
        start=0.05, stop=0.5, value=0.15, step=0.05, label="Jump Volatility (σ_j)"
    )

    mjd_controls = mo.vstack(
        [mo.md("**Adjust Jump Parameters:**"), mo.hstack([mjd_lambda, mjd_mu_j, mjd_sig_j], wrap=True)]
    )
    return mjd_controls, mjd_lambda, mjd_mu_j, mjd_sig_j


@app.cell
def _(alt, bs_rate, bs_time, bs_vol, mjd_lambda, mjd_mu_j, mjd_sig_j, np, pd):
    def plot_mjd_mc(
        lam: float,
        mu_j: float,
        sig_j: float,
        T: float,
        sigma: float,
        r: float,
        steps: int = 500,
    ):
        np.random.seed(42)  # Fixed seed so they can compare sliders reliably

        S0 = 100.0
        dt = T / steps

        # Compensator for drift to keep it risk-neutral
        k = np.exp(mu_j + 0.5 * sig_j**2) - 1.0

        S_gbm = np.zeros(steps + 1)
        S_mjd = np.zeros(steps + 1)
        S_gbm[0] = S0
        S_mjd[0] = S0

        for i in range(1, steps + 1):
            Z = np.random.normal(0, 1)

            # Standard GBM
            S_gbm[i] = S_gbm[i - 1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )

            # Merton Jump-Diffusion
            # Poisson jump occurrence
            jump_occurs = np.random.poisson(lam * dt)

            # Jump size
            if jump_occurs > 0:
                jump_multiplier = np.exp(
                    np.sum(np.random.normal(mu_j, sig_j, jump_occurs))
                )
            else:
                jump_multiplier = 1.0

            S_mjd[i] = (
                S_mjd[i - 1]
                * np.exp((r - lam * k - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
                * jump_multiplier
            )

        time_grid = np.linspace(0, T, steps + 1)

        df = pd.DataFrame(
            {
                "Time (Years)": np.tile(time_grid, 2),
                "Spot Price (S)": np.concatenate([S_gbm, S_mjd]),
                "Model": ["Standard GBM"] * (steps + 1)
                + ["Merton Jump-Diffusion"] * (steps + 1),
            }
        )

        chart = (
            alt.Chart(df)
            .mark_line(size=2)
            .encode(
                x=alt.X("Time (Years):Q"),
                y=alt.Y("Spot Price (S):Q", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "Model:N", scale=alt.Scale(range=["#2196f3", "#e91e63"])
                ),
                strokeDash=alt.condition(
                    alt.datum.Model == "Standard GBM", alt.value([5, 5]), alt.value([0])
                ),
                tooltip=["Time (Years)", "Spot Price (S)", "Model"],
            )
            .properties(
                title="Monte Carlo Asset Path Simulation", width=600, height=300
            )
        )
        return chart

    mjd_chart = plot_mjd_mc(
        mjd_lambda.value,
        mjd_mu_j.value,
        mjd_sig_j.value,
        bs_time.value,
        bs_vol.value,
        bs_rate.value,
    )
    return mjd_chart, plot_mjd_mc


@app.cell
def _(mo, mjd_chart, mjd_controls):
    mo.vstack([mjd_controls, mo.center(mjd_chart)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > **Try this:** Set jump intensity (λ) to 0 — you get standard GBM paths. Now increase λ to ~5: paths suddenly look like they have "fault lines" where the stock teleports down overnight. This is what earnings surprises, geopolitical shocks, and flash crashes look like.
        >
        > **Key insight:** OTM put options become much more expensive under MJD because jump risk cannot be hedged by continuous delta-hedging — the stock moves too fast. This directly explains why the market IV smile steepens on the downside during stressed periods.
        >
        > **Jump size vs frequency trade-off:** A few large jumps (low λ, large |μ_j|) creates "fat kurtosis" — the distribution has extreme outliers. Many small jumps (high λ, small |μ_j|) starts to approximate diffusive vol (central limit theorem kicks in).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ### 10b. Heston Stochastic Volatility Model
        
        The Heston Model treats volatility itself as a random variable. Instead of being constant, variance ($v_t = \sigma^2$) follows a **mean-reverting stochastic process** (a Cox-Ingersoll-Ross or CIR process).
        
        Price Process:
        $$\frac{dS}{S} = \mu \, dt + \sqrt{v_t} \, dW_1$$

        Variance Process:
        $$dv_t = \kappa (\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW_2$$
        
        Where:
        - $\theta$ is the long-term mean variance.
        - $\kappa$ is the rate of mean reversion (how fast variance snaps back to $\theta$).
        - $\xi$ is the "vol of vol" (how volatile the variance process itself is).
        - $dW_1$ and $dW_2$ are Wiener processes with correlation $\rho$. (Usually negative for equities: when price drops, volatility spikes).
        
        Because of this negative correlation ($\rho < 0$), Heston naturally generates the left-skewed **Volatility Smile** we observe in equity markets!
        """
    )
    return


@app.cell
def _(mo):
    h_v0 = mo.ui.slider(
        start=0.01, stop=0.5, value=0.04, step=0.01, label="Initial Variance (v0)"
    )
    h_theta = mo.ui.slider(
        start=0.01, stop=0.5, value=0.04, step=0.01, label="Long-term Variance (θ)"
    )
    h_kappa = mo.ui.slider(
        start=0.0, stop=10.0, value=2.0, step=0.5, label="Mean Reversion Rate (κ)"
    )
    h_xi = mo.ui.slider(
        start=0.0, stop=1.0, value=0.3, step=0.05, label="Vol of Vol (ξ)"
    )
    h_rho = mo.ui.slider(
        start=-0.9, stop=0.9, value=-0.7, step=0.1, label="Correlation (ρ)"
    )

    h_controls = mo.vstack(
        [mo.md("**Adjust Heston Parameters:**"), mo.hstack([h_v0, h_theta, h_kappa, h_xi, h_rho], wrap=True)]
    )
    return h_controls, h_kappa, h_rho, h_theta, h_v0, h_xi


@app.cell
def _(alt, bs_rate, bs_time, h_kappa, h_rho, h_theta, h_v0, h_xi, np, pd):
    def plot_heston_mc(
        v0: float,
        theta: float,
        kappa: float,
        xi: float,
        rho: float,
        T: float,
        r: float,
        steps: int = 500,
    ):
        np.random.seed(42)

        S0 = 100.0
        dt = T / steps

        S_path = np.zeros(steps + 1)
        v_path = np.zeros(steps + 1)
        S_path[0] = S0
        v_path[0] = v0

        for i in range(1, steps + 1):
            # Generate correlated Brownian motions
            Z1 = np.random.normal(0, 1)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1)

            # Ensure variance doesn't go negative during discrete steps (Full Truncation scheme)
            v_curr = max(v_path[i - 1], 0)

            # Variance step
            v_path[i] = (
                v_curr
                + kappa * (theta - v_curr) * dt
                + xi * np.sqrt(v_curr) * np.sqrt(dt) * Z2
            )
            v_path[i] = max(v_path[i], 0)  # Floor again

            # Price step
            S_path[i] = S_path[i - 1] * np.exp(
                (r - 0.5 * v_curr) * dt + np.sqrt(v_curr * dt) * Z1
            )

        time_grid = np.linspace(0, T, steps + 1)

        df_S = pd.DataFrame(
            {"Time (Years)": time_grid, "Value": S_path, "Metric": "Asset Price (S)"}
        )

        # Plot Volatility (sqrt of variance) instead of variance for better intuition
        df_v = pd.DataFrame(
            {
                "Time (Years)": time_grid,
                "Value": np.sqrt(v_path),
                "Metric": "Volatility (σ)",
            }
        )

        chart_S = (
            alt.Chart(df_S)
            .mark_line(color="#e91e63", size=2)
            .encode(
                x=alt.X("Time (Years):Q"),
                y=alt.Y("Value:Q", scale=alt.Scale(zero=False), title="Asset Price"),
                tooltip=["Time (Years)", "Value"],
            )
            .properties(width=600, height=200, title="Heston Asset Price Path")
        )

        chart_v = (
            alt.Chart(df_v)
            .mark_line(color="#ff9800", size=2)
            .encode(
                x=alt.X("Time (Years):Q"),
                y=alt.Y("Value:Q", scale=alt.Scale(zero=False), title="Volatility"),
                tooltip=["Time (Years)", "Value"],
            )
            .properties(width=600, height=200, title="Stochastic Volatility Path")
        )

        return alt.vconcat(chart_S, chart_v).resolve_scale(y="independent")

    h_chart = plot_heston_mc(
        h_v0.value,
        h_theta.value,
        h_kappa.value,
        h_xi.value,
        h_rho.value,
        bs_time.value,
        bs_rate.value,
    )
    return h_chart, plot_heston_mc


@app.cell
def _(mo, h_chart, h_controls):
    mo.vstack([h_controls, mo.center(h_chart)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > **Try this:** Set mean-reversion speed (κ) to near 0 — vol becomes a random walk that drifts far from its long-run mean. Now increase κ to ~5: vol snaps back quickly, paths look almost like constant-vol GBM. Real equity vol has κ ≈ 2–5.
        >
        > **Vol-of-vol (ξ) insight:** Increase ξ and watch the vol panel (bottom chart) become much more erratic. High vol-of-vol generates fatter tails in returns — the IV smile steepens because big moves become more probable when vol itself can spike.
        >
        > **Correlation (ρ) and skew:** Set ρ = −0.7 (typical for equities). Notice paths tend to drop when vol rises — the leverage effect. This negative correlation between returns and vol is why equity IV smiles are *skewed left* rather than symmetric.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ### 10c. Local Volatility — Dupire's Model

        Stochastic volatility models like Heston are powerful but require calibration to many parameters, and they may not *exactly* reproduce every observed market price. **Dupire's Local Volatility model** takes a different approach: instead of specifying a stochastic process for $\sigma$, it asks:

        > *What deterministic function $\sigma(S, t)$ would make Black-Scholes price every option correctly?*

        Given a complete, smooth implied volatility surface $\sigma_{IV}(K, T)$, Dupire showed (1994) that the unique local vol function consistent with all observed prices is:

        $$\sigma_{loc}^2(K, T) = \frac{\dfrac{\partial C}{\partial T} + rK\dfrac{\partial C}{\partial K}}{\dfrac{1}{2}K^2\dfrac{\partial^2 C}{\partial K^2}}$$

        The denominator is exactly the Breeden-Litzenberger term (the risk-neutral PDF). The numerator captures how option prices change with maturity.

        **Key insight**: Local vol is a *function of spot and time*, not a random process. This means:
        - It perfectly fits the current market surface by construction
        - Future smiles predicted by Dupire are "sticky-strike" (see below)
        - The model is a single-factor model — vol moves only because spot moves
        - In practice, Dupire's predicted future skew is **too flat** — real markets have richer vol dynamics than a deterministic function can capture

        **Visualizing local vol**: The local vol surface computed from a typical equity market shows:
        - **Strike dimension**: Higher local vol at low strikes (leverage effect dominates)
        - **Time dimension**: Local vol tends to revert toward a flat long-run level
        - Local vol at a given $(K, T)$ is roughly related to IV by: $\sigma_{loc}(K,T) \approx \sigma_{IV}(K,T) \cdot \text{correction for skew slope and curvature}$
        """
    )
    return


@app.cell
def _(alt, bs_vol, np, pd, vol_convex, vol_skew, vol_term):
    def compute_local_vol_surface(
        atm_vol: float, skew: float, convex: float, term: float
    ):
        """
        Numerically compute Dupire's local vol surface from a synthetic IV surface.
        Uses finite differences of call prices: σ_loc² = (∂C/∂T + rK∂C/∂K) / (0.5 K² ∂²C/∂K²)
        """
        from scipy.stats import norm as _n

        S = 100.0
        r = 0.05
        K_vals = np.linspace(S * 0.55, S * 1.55, 60)
        T_vals = np.linspace(0.1, 2.0, 40)
        dK = K_vals[1] - K_vals[0]
        dT = T_vals[1] - T_vals[0]

        def get_iv(K_val: float, T_val: float) -> float:
            mono = (K_val / S) - 1.0
            smile = 1.0 + skew * mono + convex * mono ** 2
            ts = 1.0 + term / np.sqrt(T_val)
            return max(0.02, atm_vol * smile * ts)

        def bs_call(K_val: float, T_val: float) -> float:
            sig = get_iv(K_val, T_val)
            d1 = (np.log(S / K_val) + (r + 0.5 * sig ** 2) * T_val) / (sig * np.sqrt(T_val))
            d2 = d1 - sig * np.sqrt(T_val)
            return S * _n.cdf(d1) - K_val * np.exp(-r * T_val) * _n.cdf(d2)

        records = []
        for j, T in enumerate(T_vals[1:-1], start=1):
            for i, K in enumerate(K_vals[1:-1], start=1):
                C = bs_call(K, T)
                C_Kp = bs_call(K + dK, T)
                C_Km = bs_call(K - dK, T)
                C_Tp = bs_call(K, T + dT)
                C_Tm = bs_call(K, T - dT)

                dC_dT = (C_Tp - C_Tm) / (2 * dT)
                dC_dK = (C_Kp - C_Km) / (2 * dK)
                d2C_dK2 = (C_Kp - 2 * C + C_Km) / (dK ** 2)

                num = dC_dT + r * K * dC_dK
                den = 0.5 * K ** 2 * d2C_dK2
                lv_sq = num / den if den > 1e-8 else np.nan
                lv = np.sqrt(max(lv_sq, 0.0)) if (lv_sq is not np.nan and not np.isnan(lv_sq)) else np.nan
                # Cap to reasonable range
                if lv is not np.nan and not np.isnan(lv) and lv < 2.0:
                    records.append({"Strike": K, "Maturity (T)": round(T, 2), "Local Vol": lv})

        df = pd.DataFrame(records)
        if df.empty:
            return alt.Chart(pd.DataFrame({"x": [0]})).mark_text(text="No data").encode()

        chart = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=alt.X("Strike:Q", bin=alt.Bin(maxbins=25), title="Strike (K)"),
                y=alt.Y("Maturity (T):Q", bin=alt.Bin(maxbins=20), title="Maturity T (years)"),
                color=alt.Color("mean(Local Vol):Q", scale=alt.Scale(scheme="viridis"), title="σ_loc"),
                tooltip=[
                    alt.Tooltip("Strike:Q", format=".0f"),
                    alt.Tooltip("Maturity (T):Q", format=".2f"),
                    alt.Tooltip("mean(Local Vol):Q", format=".3f", title="Local Vol"),
                ],
            )
            .properties(
                width=500, height=300,
                title="Dupire Local Volatility Surface  (Heatmap)",
            )
        )
        return chart

    lv_chart = compute_local_vol_surface(
        bs_vol.value, vol_skew.value, vol_convex.value, vol_term.value
    )
    return lv_chart, compute_local_vol_surface


@app.cell
def _(mo, lv_chart):
    mo.vstack([
        mo.md("*(Uses the Skew / Convexity / Term Structure sliders from Section 6 above)*"),
        mo.center(lv_chart),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ### 10d. Skew Dynamics: Sticky-Strike vs. Sticky-Moneyness

        When the spot price $S$ moves, how does the implied volatility surface *move with it*? This seemingly simple question has major consequences for delta hedging, model risk, and P&L attribution.

        **Two canonical regimes**:

        | Regime | Definition | When It Applies |
        |--------|-----------|-----------------|
        | **Sticky-Strike** | IV at a fixed strike $K$ stays constant as spot moves | Dupire local vol, index options in normal markets |
        | **Sticky-Moneyness** (sticky-delta) | IV at a fixed moneyness $K/S$ stays constant — the smile *moves with* the spot | Individual equity options, earnings moves, stressed markets |

        #### Implications for Delta Hedging

        The choice of model determines how you compute **delta** ($\partial C / \partial S$):

        - Under **sticky-strike**: as spot rises, the moneyness $K/S$ falls (the option moves deeper ITM). The IV at the fixed strike stays constant, so you're reading from a fixed point on the surface. Black-Scholes delta is approximately correct.

        - Under **sticky-moneyness**: as spot rises, the smile shifts right. The option stays ATM, and its IV stays constant. The effective delta is *higher* than BS because moving spot changes which part of the smile you're pricing with.

        **The skew correction to delta** (approximate):
        $$\Delta_{sticky-\Delta} \approx \Delta_{BS} - \text{Vega} \cdot \frac{\partial \sigma_{IV}}{\partial S}\bigg|_{K/S = const}$$

        In markets with strong negative skew, the skew-adjusted delta for puts is *larger in magnitude* than BS delta — the option's IV goes up when the stock falls, amplifying the position's sensitivity.

        #### Which Model Is Right?

        Neither perfectly describes reality. Empirically:
        - For short-dated options, **sticky-moneyness** is a better description of intraday behavior
        - For longer-dated options, **sticky-strike** is closer to observed behavior
        - The reality is somewhere in between, and depends on the asset class (indices vs. singles vs. rates)

        Sophisticated dealers use skew-adjusted deltas and run P&L explain reports to monitor which regime their book is in.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 10c. Exotic Options & Path Dependency

        Standard options are **path-independent**: only the final price $S_T$ matters. **Exotic options** depend on the *entire history* of the price path.

        | Option | Payoff | Why cheaper? |
        |---|---|---|
        | **European Call** | $\max(S_T - K,\, 0)$ | Baseline |
        | **Asian Call** | $\max(A_T - K,\, 0)$, $A_T$ = avg price | Averaging dampens effective volatility |
        | **Down-and-Out Call** | $\max(S_T - K,\, 0)$ unless path hits barrier → $0$ | You give up protection if stock crashes |

        The chart below simulates one GBM path. The **orange** line is the knock-out barrier; if the path touches it, the barrier option expires worthless. The **pink** dashed line is the strike. The **green** dashed line is the Asian average.
        """
    )
    return


@app.cell
def _(mo):
    exo_K = mo.ui.slider(start=50, stop=150, value=100, label="Strike (K)")
    exo_barrier = mo.ui.slider(
        start=50, stop=100, value=80, label="Down-and-Out Barrier"
    )
    exo_sim = mo.ui.run_button(label="Simulate Path-Dependent Payoffs")

    exo_controls = mo.vstack(
        [mo.md("**Adjust Exotic Parameters:**"), mo.hstack([exo_K, exo_barrier, exo_sim], wrap=True)]
    )
    return exo_barrier, exo_controls, exo_K, exo_sim


@app.cell
def _(alt, bs_rate, bs_time, bs_vol, exo_barrier, exo_K, exo_sim, np, pd):
    def simulate_exotics(
        K: float,
        barrier: float,
        T: float,
        sigma: float,
        r: float,
        steps: int = 252,
        num_paths: int = 1000,
    ):
        # We simulate multiple paths to find the average expected payoff (Monte Carlo Pricing)
        np.random.seed(exo_sim.value if exo_sim.value else 42)

        S0 = 100.0
        dt = T / steps
        discount = np.exp(-r * T)

        # We will visualize just ONE path, but price using ALL paths
        S_paths = np.zeros((num_paths, steps + 1))
        S_paths[:, 0] = S0

        for i in range(1, steps + 1):
            Z = np.random.normal(0, 1, num_paths)
            S_paths[:, i] = S_paths[:, i - 1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )

        # 1. European Call
        ST = S_paths[:, -1]
        eu_payoffs = np.maximum(ST - K, 0)
        eu_price = discount * np.mean(eu_payoffs)

        # 2. Asian Call (Arithmetic Average)
        A_T = np.mean(S_paths, axis=1)
        asian_payoffs = np.maximum(A_T - K, 0)
        asian_price = discount * np.mean(asian_payoffs)

        # 3. Down-and-Out Barrier Call
        # If the min price of the path is <= barrier, payoff is 0
        min_prices = np.min(S_paths, axis=1)
        barrier_hit = min_prices <= barrier
        barrier_payoffs = np.maximum(ST - K, 0)
        barrier_payoffs[barrier_hit] = 0.0
        barrier_price = discount * np.mean(barrier_payoffs)

        # Visualize the first path and its Exotic features
        demo_path = S_paths[0, :]
        time_grid = np.linspace(0, T, steps + 1)
        demo_avg = np.mean(demo_path)
        demo_min = np.min(demo_path)

        df_path = pd.DataFrame({"Time (Years)": time_grid, "Spot Price (S)": demo_path})

        chart_path = (
            alt.Chart(df_path)
            .mark_line(color="#2196f3", size=3)
            .encode(
                x=alt.X("Time (Years):Q"),
                y=alt.Y("Spot Price (S):Q", scale=alt.Scale(zero=False)),
                tooltip=["Time (Years)", "Spot Price (S)"],
            )
        )

        # Add strike and barrier lines
        chart_strike = (
            alt.Chart(pd.DataFrame({"y": [K]}))
            .mark_rule(color="#e91e63", strokeDash=[5, 5])
            .encode(y="y:Q")
        )
        chart_barrier = (
            alt.Chart(pd.DataFrame({"y": [barrier]}))
            .mark_rule(color="#ff9800", strokeWidth=2)
            .encode(y="y:Q")
        )
        chart_avg = (
            alt.Chart(pd.DataFrame({"y": [demo_avg]}))
            .mark_rule(color="#4caf50", strokeDash=[2, 2])
            .encode(y="y:Q")
        )

        final_chart = (
            chart_path + chart_strike + chart_barrier + chart_avg
        ).properties(
            title="Sample GBM Path",
            width=620,
            height=300,
        )

        return (
            final_chart,
            eu_price,
            asian_price,
            barrier_price,
            demo_path,
            demo_avg,
            demo_min,
        )

    ex_chart, ex_eu, ex_as, ex_ba, ex_path, ex_avg, ex_min = simulate_exotics(
        exo_K.value, exo_barrier.value, bs_time.value, bs_vol.value, bs_rate.value
    )

    ex_hit_text = (
        "🟢 Barrier NOT Hit"
        if ex_min > exo_barrier.value
        else "🔴 Barrier HIT! (Knocked Out)"
    )
    ex_eu_payoff = max(ex_path[-1] - exo_K.value, 0)
    ex_as_payoff = max(ex_avg - exo_K.value, 0)
    ex_ba_payoff = 0.0 if ex_min <= exo_barrier.value else ex_eu_payoff

    ex_output = mo.md(rf"""
| | MC Price (1000 paths) | This path payoff |
|---|---|---|
| **European Call** | ${ex_eu:.2f} | ${ex_eu_payoff:.2f} ($S_T={ex_path[-1]:.1f}$) |
| **Asian Call** | ${ex_as:.2f} | ${ex_as_payoff:.2f} ($A_T={ex_avg:.1f}$) |
| **Barrier Call** | ${ex_ba:.2f} | ${ex_ba_payoff:.2f} — {ex_hit_text} |
    """)

    return (
        ex_avg,
        ex_ba,
        ex_ba_payoff,
        ex_chart,
        ex_eu,
        ex_eu_payoff,
        ex_hit_text,
        ex_min,
        ex_output,
        ex_path,
        ex_as,
        ex_as_payoff,
        simulate_exotics,
    )


@app.cell
def _(mo, ex_chart, exo_controls, ex_output):
    mo.vstack([exo_controls, mo.center(ex_chart), ex_output])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 11. Probability Measures & Change of Measure

        Every probability statement in finance depends on *which measure you're using*. The same event can have dramatically different probabilities under different measures — and understanding this is the key to understanding why Black-Scholes works and where it can fail.

        ### The Two Worlds: $\mathbb{P}$ and $\mathbb{Q}$

        | | Real-World Measure $\mathbb{P}$ | Risk-Neutral Measure $\mathbb{Q}$ |
        |---|---|---|
        | **Also called** | Physical measure, objective measure | Pricing measure, equivalent martingale measure |
        | **Stock drift** | $\mu$ (equity risk premium above $r$) | $r$ (risk-free rate) |
        | **Use case** | Portfolio management, risk forecasting, P&L attribution | Derivatives pricing |
        | **Expectation** | $\mathbb{E}^{\mathbb{P}}[S_T] = S_0 e^{\mu T}$ | $\mathbb{E}^{\mathbb{Q}}[S_T] = S_0 e^{rT}$ |

        Under $\mathbb{P}$: stocks grow at the expected return $\mu > r$ to compensate investors for bearing risk.

        Under $\mathbb{Q}$: stocks grow at the risk-free rate $r$. Investors are "pretend risk-neutral" — they don't demand a premium. This is purely a mathematical device.

        ### Girsanov's Theorem: The Bridge Between Measures

        Under $\mathbb{P}$, the stock follows:
        $$dS = \mu S \, dt + \sigma S \, dW^{\mathbb{P}}$$

        Girsanov's theorem tells us that we can **switch from $\mathbb{P}$ to $\mathbb{Q}$** by adjusting the Brownian motion. Define the **market price of risk**:
        $$\lambda = \frac{\mu - r}{\sigma}$$

        Then:
        $$W^{\mathbb{Q}}_t = W^{\mathbb{P}}_t + \lambda t$$

        is a standard Brownian motion under $\mathbb{Q}$. Substituting:
        $$dS = \mu S \, dt + \sigma S \, (dW^{\mathbb{Q}} - \lambda \, dt) = r S \, dt + \sigma S \, dW^{\mathbb{Q}}$$

        The drift changes from $\mu$ to $r$ — exactly what we needed. The volatility $\sigma$ is **unchanged** by the measure change. This is why implied volatility is the same under both measures.

        ### The Radon-Nikodym Derivative

        Girsanov's theorem works via a density process — the **Radon-Nikodym derivative** $\frac{d\mathbb{Q}}{d\mathbb{P}}$. It re-weights each scenario by:

        $$\left.\frac{d\mathbb{Q}}{d\mathbb{P}}\right|_T = \exp\!\left(-\lambda W^{\mathbb{P}}_T - \tfrac{1}{2}\lambda^2 T\right) =: \mathcal{E}(-\lambda W^{\mathbb{P}})_T$$

        This is the **Doléans-Dade exponential**. Scenarios where the stock did well (large $W^{\mathbb{P}}_T > 0$) get down-weighted under $\mathbb{Q}$; bad scenarios get up-weighted. This effectively removes the risk premium: investors under $\mathbb{Q}$ "don't care" about the extra drift.

        **Risk-Neutral Valuation** follows directly: since $S_t / B_t$ is a $\mathbb{Q}$-martingale (where $B_t = e^{rt}$ is the money-market account):
        $$V_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}}[\text{Payoff}(S_T)]$$

        No-arbitrage forces this to be the unique fair price. If you priced using $\mathbb{E}^{\mathbb{P}}$ instead, you'd need to know $\mu$ — which is unobservable and leads to no-arbitrage violations.

        ### Other Numeraires and Measures

        The risk-neutral measure uses the money-market account $B_t = e^{rt}$ as the **numeraire** (the "unit of account"). Other choices are useful in specific contexts:

        | Measure | Numeraire | Key property | Where used |
        |---------|-----------|-------------|------------|
        | Risk-neutral $\mathbb{Q}$ | Money market $e^{rt}$ | Stock is a $\mathbb{Q}$-martingale after discounting | All vanilla options |
        | $T$-Forward measure $\mathbb{Q}^T$ | Zero-coupon bond $P(t,T)$ | Forward price $F(t,T) = S_t/P(t,T)$ is a $\mathbb{Q}^T$-martingale | Interest rate derivatives, Black-76 |
        | Stock measure $\mathbb{Q}^S$ | Stock $S_t$ | $S_t$ is the numeraire, so exchange options are natural | Margrabe's formula for exchange options |
        | Swap measure | Annuity $A(t)$ | Swap rate is a $\mathbb{Q}^A$-martingale | Swaptions, LIBOR market model (LMM) |

        **Why does the forward measure matter?** Under $\mathbb{Q}^T$, the forward price $F_t = S_t e^{r(T-t)}$ is a martingale — drift-free. This simplifies pricing when the payoff depends on a forward price rather than a spot price (e.g., options on futures).

        ### $\mathbb{P}$ vs $\mathbb{Q}$: Why the Distinction Matters

        | Task | Use $\mathbb{P}$ | Use $\mathbb{Q}$ |
        |------|-----------------|-----------------|
        | Price a call option | ✗ (depends on unknown $\mu$) | ✓ (model-free no-arbitrage price) |
        | Estimate 1-day VaR | ✓ (real-world tail probability) | ✗ (wrong drift, wrong tails) |
        | Calibrate a model | $\mathbb{P}$ for real-world params | $\mathbb{Q}$ for implied vol surface |
        | P&L explanation | ✓ ($\mu$ drives expected profits) | ✗ |
        | Risk-neutral delta hedging | ✓ (hedge neutralizes $\mathbb{P}$ risk) | ✓ (hedge ratio same in both) |

        A common error: using implied volatility (a $\mathbb{Q}$ object) to forecast real-world variance. The VIX, for example, measures $\mathbb{Q}$-volatility — it systematically overstates realized volatility because it includes the variance risk premium (the compensation for bearing vol uncertainty under $\mathbb{P}$).
        """
    )
    return


@app.cell
def _(mo):
    pm_mu = mo.ui.slider(start=0.0, stop=0.30, value=0.10, step=0.01, label="Real-world drift μ (annual)")
    pm_sigma = mo.ui.slider(start=0.05, stop=0.60, value=0.20, step=0.01, label="Volatility σ (annual)")
    pm_r = mo.ui.slider(start=0.0, stop=0.10, value=0.05, step=0.005, label="Risk-free rate r (annual)")
    pm_T = mo.ui.slider(start=0.1, stop=2.0, value=1.0, step=0.1, label="Time to expiry T (years)")
    pm_controls = mo.vstack([
        mo.md("**Adjust parameters to see how $\\mathbb{P}$ and $\\mathbb{Q}$ distributions differ:**"),
        mo.hstack([pm_mu, pm_sigma, pm_r, pm_T], wrap=True),
    ])
    return pm_T, pm_controls, pm_mu, pm_r, pm_sigma


@app.cell
def _(alt, mo, norm, np, pd, pm_T, pm_controls, pm_mu, pm_r, pm_sigma):
    _S0 = 100.0
    _mu, _sigma, _r, _T = pm_mu.value, pm_sigma.value, pm_r.value, pm_T.value
    _lam = (_mu - _r) / _sigma  # market price of risk λ = (μ−r)/σ

    # Terminal stock price grid
    _K = np.linspace(_S0 * 0.1, _S0 * 3.5, 800)
    _x = np.log(_K / _S0)
    _sig = _sigma * np.sqrt(_T)

    # Under P: log(S_T/S_0) ~ N((mu - sigma^2/2)*T, sigma^2*T)
    _mu_p = (_mu - 0.5 * _sigma ** 2) * _T
    _pdf_p = norm.pdf((_x - _mu_p) / _sig) / (_K * _sig)

    # Under Q: drift = r instead of mu (everything else identical)
    _mu_q = (_r - 0.5 * _sigma ** 2) * _T
    _pdf_q = norm.pdf((_x - _mu_q) / _sig) / (_K * _sig)

    # Radon-Nikodym: dQ/dP = exp(-λW_T - ½λ²T), W_T = (log(K/S0) - (μ-½σ²)T)/σ
    _W_T = (_x - _mu_p) / _sigma
    _rn_weight = np.exp(-_lam * _W_T - 0.5 * _lam ** 2 * _T)

    _dK = _K[1] - _K[0]
    _pdf_p /= max(np.sum(_pdf_p) * _dK, 1e-10)
    _pdf_q /= max(np.sum(_pdf_q) * _dK, 1e-10)

    _labels = [f"Real-world ℙ  (drift={_mu*100:.0f}%)", f"Risk-neutral ℚ  (drift={_r*100:.0f}%)"]
    _df = pd.DataFrame({
        "Stock Price at T": np.tile(_K, 2),
        "Probability Density": np.concatenate([_pdf_p, _pdf_q]),
        "Measure": _labels[0:1] * len(_K) + _labels[1:2] * len(_K),
    })

    _dist_chart = (
        alt.Chart(_df)
        .mark_line(size=2.5)
        .encode(
            x=alt.X("Stock Price at T:Q", title="Stock Price at Expiry"),
            y=alt.Y("Probability Density:Q"),
            color=alt.Color("Measure:N", scale=alt.Scale(domain=_labels, range=["#1976d2", "#e53935"])),
            tooltip=["Stock Price at T", alt.Tooltip("Probability Density:Q", format=".5f"), "Measure"],
        )
        .properties(
            title=f"Terminal Stock Distribution: ℙ vs ℚ  (λ = {_lam:.2f})",
            width=620, height=280,
        )
        .interactive()
    )

    _df_rn = pd.DataFrame({"Stock Price at T": _K, "dQ/dP weight": _rn_weight})
    _rn_chart = (
        alt.Chart(_df_rn)
        .mark_line(size=2, color="#9c27b0")
        .encode(
            x=alt.X("Stock Price at T:Q"),
            y=alt.Y("dQ/dP weight:Q", title="dQ/dP  (scenario weight)"),
            tooltip=["Stock Price at T", alt.Tooltip("dQ/dP weight:Q", format=".3f")],
        )
    ) + alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color="#9e9e9e", strokeDash=[4, 3]).encode(y="y:Q")
    _rn_chart = _rn_chart.properties(
        title="Radon-Nikodym dQ/dP: down-weights good outcomes, up-weights bad ones",
        width=620, height=180,
    ).interactive()

    _combined = alt.vconcat(_dist_chart, _rn_chart).resolve_scale(x="shared")
    mo.vstack([pm_controls, _combined])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > **Try this:** Set μ = 10% and r = 5%. The ℙ distribution (blue) is shifted right — the market expects the stock to grow faster than risk-free. The ℚ distribution (red) uses r as drift, so it's centered to the left. The gap between the two peaks is driven by λ = (μ−r)/σ, the **market price of risk**.
        >
        > **Radon-Nikodym insight:** The dQ/dP weight (purple chart) is below 1 for high stock prices (good outcomes get down-weighted under Q) and above 1 for low prices (bad outcomes get up-weighted). This re-weighting exactly removes the risk premium — making investors pretend they're indifferent to risk.
        >
        > **When λ = 0** (μ = r): both distributions are identical. No equity premium means no difference between the real world and the risk-neutral world.

        ---

        ## 12. American Options & Early Exercise

        So far, we've only discussed **European Options** (which can only be exercised exactly at expiration). **American Options** can be exercised *any time* before expiration.

        Because of this added flexibility, an American option is always worth $\geq$ a European option. In fact, Black-Scholes **cannot** price American options! We must use numerical methods like the Binomial Tree or finite difference methods.

        *When is early exercise optimal?*
        - **Calls**: It is rarely optimal to exercise an American Call early *unless* the underlying stock pays a dividend.
        - **Puts**: Deep In-The-Money American Puts are often optimal to exercise early because the time value of the cash received by selling the stock outweighs the remaining time value of the put.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        
        ### 💡 Interesting Greek Properties (Cheat Sheet)

        | Greek | Symbol | Property | Interesting Fact |
        |---|---|---|---|
        | **Delta** | $\Delta$ | Sensitivity to $S$ | Can be interpreted as the rough *probability* the option expires In-The-Money (ITM). Call Delta + Put Delta $\approx$ 1. |
        | **Gamma** | $\Gamma$ | Sensitivity of $\Delta$ to $S$ | Highest for At-The-Money (ATM) options with short expiration. Long Gamma means you profit from large price swings. |
        | **Vega** | $\mathcal{V}$ | Sensitivity to $\sigma$ | Highest for ATM options. You want to be long Vega when you expect a volatility spike. |
        | **Theta** | $\Theta$ | Time decay | Accelerates as expiration approaches for ATM options. Options are deprecating assets! |
        | **Rho** | $\rho$ | Sensitivity to $r$ | Usually the least impactful Greek for short-term retail trading, but critical for long-dated options (LEAPS). |
        | **Vanna** | $\frac{\partial \Delta}{\partial \sigma}$ | Sensitivity of $\Delta$ to $\sigma$ | Changes sign depending on moneyness. If Volatility spikes, an Out-Of-The-Money (OTM) call's Delta increases (more likely to be ITM). |
        | **Volga** | $\frac{\partial \mathcal{V}}{\partial \sigma}$ | Sensitivity of $\mathcal{V}$ to $\sigma$ | Positive for OTM options. Used to price smile/skew convexity. |
        | **Speed** | $\frac{\partial \Gamma}{\partial S}$ | Sensitivity of $\Gamma$ to $S$ | High Speed means your Gamma hedging ratio changes quickly as spot moves. |
        | **Zomma** | $\frac{\partial \Gamma}{\partial \sigma}$ | Sensitivity of $\Gamma$ to $\sigma$ | Indicates how much your portfolio Gamma will shift if VIX spikes. |
        | **Color** | $\frac{\partial \Gamma}{\partial t}$ | Sensitivity of $\Gamma$ to $t$ | Like Theta, but for Gamma. Gamma decay is a crucial factor for traders rolling positions as expiration nears. |
        """
    )
    return


if __name__ == "__main__":
    app.run()

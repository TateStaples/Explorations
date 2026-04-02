import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2D Ising MCMC (Glauber Dynamics)

    This notebook provides a ipython notebook template for the computer experiment described in `lec5.pdf`:
    - lattice size `L=32`, coupling `J=1`
    - run `5000` sweeps at temperatures `T in {1.5, 2.3, 3.5}`
    - compare magnetization traces and autocorrelation

    Notation reminder: `beta = 1 / T`.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    import numpy.typing as npt
    from matplotlib.animation import FuncAnimation

    return FuncAnimation, np, npt, plt, random


@app.cell
def _(np, npt):
    # Use Python type support
    SpinArray = npt.NDArray[np.int_]


    def init_spins(width: int, height: int, rng: np.random.Generator) -> SpinArray:
        """Initialize an (m, n) Ising lattice with spins in {-1, +1}."""
        return np.where(rng.random((width, height)) < 0.5, -1, 1).astype(np.int_)

    def sweep_glauber(
        spins: SpinArray, 
        coupling_strength: float, 
        external_field: float, 
        beta: float, 
        rng: np.random.Generator
        ) -> SpinArray:
        """Run one Glauber sweep (m*n random-site updates) with periodic boundaries."""
        width, height = spins.shape
        for _ in range(width * height):
            random_col = int(rng.integers(0, width))
            random_row = int(rng.integers(0, height))
            # Add code here to compute effective field for spin[i,j] and resample
            # Sum of neighbor spins with periodic boundaries
            neighbors_sum = (
                spins[(random_col - 1) % width, random_row]     # Left neighbor
                + spins[(random_col + 1) % width, random_row]   # Right neighbor
                + spins[random_col, (random_row - 1) % height]  # Up neighbor
                + spins[random_col, (random_row + 1) % height]  # Down neighbor
            )

            # p(sigma = +1) = sigmoid(2 * beta * h_eff)
            energy = external_field + coupling_strength * neighbors_sum
            p_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * energy))

            spins[random_col, random_row] = 1 if rng.random() < p_plus else -1

        return spins

    def sweep_metropolis(
        spins: SpinArray, 
        coupling_strength: float, 
        external_field: float, 
        beta: float, 
        rng: np.random.Generator
        ) -> SpinArray:
        """Run one Metropolis sweep (m*n random-site updates) with periodic boundaries."""
        width, height = spins.shape
        for _ in range(width * height):
            random_col = int(rng.integers(0, width))
            random_row = int(rng.integers(0, height))
            # Add code here to compute effective field for spin[i,j] and resample
            neighbors_sum = (
                spins[(random_col - 1) % width, random_row]     # Left neighbor
                + spins[(random_col + 1) % width, random_row]   # Right neighbor
                + spins[random_col, (random_row - 1) % height]  # Up neighbor
                + spins[random_col, (random_row + 1) % height]  # Down neighbor
            )

            energy_diff = 2 * spins[random_col, random_row] * (external_field + coupling_strength * neighbors_sum)

            if energy_diff <= 0 or rng.random() < np.exp(-beta * energy_diff):
                spins[random_col, random_row] *= -1

        return spins


    def magnetization(spins: SpinArray) -> float:
        """Return magnetization per spin M = mean(sigma)."""
        return float(np.mean(spins))


    def energy_per_spin(spins: SpinArray, coupling_strength: float, external_field: float) -> float:
        """Return energy per spin U = E / N with periodic boundaries."""
        interaction = 0.0
        # Horizontal interactions
        interaction += np.sum(spins * np.roll(spins, shift=-1, axis=0))
        # Vertical interactions
        interaction += np.sum(spins * np.roll(spins, shift=-1, axis=1))

        field = np.sum(spins)
        E = -coupling_strength * interaction - external_field * field
        return float(E / spins.size)


    def run_chain(
        array_size: int,
        sweeps: int,
        temperature: float,
        coupling_strength: float = 1.0,
        external_field: float = 0.0,  # h
        random_seed: int = 0,
        keep_frames: bool = False,
    ) -> dict[str, object]:
        """Simulate 2D Ising Glauber dynamics and record observables by sweep."""
        beta = 1.0 / temperature
        rng = np.random.default_rng(random_seed)

        spins = init_spins(array_size, array_size, rng)
        frames: list[SpinArray] = [spins.copy()] if keep_frames else []
        magnetization_trace: list[float] = []
        energy_trace: list[float] = []

        for _ in range(sweeps):
            sweep_glauber(spins, coupling_strength=coupling_strength, external_field=external_field, beta=beta, rng=rng)
            magnetization_trace.append(magnetization(spins))
            energy_trace.append(energy_per_spin(spins, coupling_strength=coupling_strength, external_field=external_field))
            if keep_frames:
                frames.append(spins.copy())

        return {
            "spins": spins,
            "M": np.asarray(magnetization_trace, dtype=float),
            "U": np.asarray(energy_trace, dtype=float),
            "frames": frames,
            "beta": beta,
            "temperature": temperature,
            "J": coupling_strength,
            "h": external_field,
            "seed": random_seed,
        }

    return SpinArray, run_chain


@app.cell
def _(np, random):
    def wolff_step(lattice, T, J=1.0):
        rows, cols = lattice.shape
        # 1. Pick a random seed
        r, c = random.randint(0, rows-1), random.randint(0, cols-1)
        old_spin = lattice[r, c]
        new_spin = -old_spin
    
        # 2. Probability of adding a neighbor
        p_add = 1.0 - np.exp(-2.0 * J / T)
    
        # 3. Cluster growth (BFS)
        cluster_stack = [(r, c)]
        lattice[r, c] = new_spin # Flip the seed immediately
    
        while cluster_stack:
            curr_r, curr_c = cluster_stack.pop()
        
            # Check all 4 neighbors (with periodic boundary conditions)
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = (curr_r + dr) % rows, (curr_c + dc) % cols
            
                # If neighbor is the same as the original spin, try to add it
                if lattice[nr, nc] == old_spin:
                    if random.random() < p_add:
                        lattice[nr, nc] = new_spin
                        cluster_stack.append((nr, nc))
    
        return lattice

    return


@app.cell
def _(FuncAnimation, SpinArray, plt):
    def animate_spins(
        frames: list[SpinArray],
        title: str = "Spin configuration",
        interval_ms: int = 50,
    ) -> FuncAnimation:
        """Create a matplotlib animation for a sequence of Ising lattices."""
        _fig, _ax = plt.subplots(figsize=(5, 5))
        im = _ax.imshow(
            frames[0], cmap="GnBu_r", vmin=-1, vmax=1, interpolation="nearest"
        )
        _ax.set_xticks([])
        _ax.set_yticks([])
        title_text = _ax.set_title(f"{title} (sweep 0)")

        def update(k: int):
            im.set_data(frames[k])
            title_text.set_text(f"{title} (sweep {k})")
            return (im, title_text)

        anim = FuncAnimation(
            _fig,
            update,
            frames=len(frames),
            interval=interval_ms,
            blit=False,
            repeat=False,
        )
        plt.close(_fig)
        return anim

    return (animate_spins,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo run parameters

    - random initialization
    - `L=32`, `J=1`, `h=0`
    - `S=800` sweeps at `T=2.3` (`beta=1/2.3`)
    - animate spin configuration after each sweep
    """)
    return


@app.cell
def _(animate_spins, mo, run_chain):
    demo = run_chain(
        array_size=32, 
        sweeps=800, 
        temperature=2.3, 
        coupling_strength=1.0, 
        external_field=0.0, 
        random_seed=1, 
        keep_frames=True
    )
    anim = animate_spins(
        demo["frames"],
        title=f"2D Ising (L=32, beta={demo['beta']:.3f}, J={demo['J']}, h={demo['h']})",
        interval_ms=40,
    )
    mo.Html(anim.to_html5_video())
    return (demo,)


@app.cell
def _(demo, mo):
    chosen_idx = mo.ui.slider(0, len(demo["frames"]) - 1, step=1, label="Sweep")
    chosen_idx
    return (chosen_idx,)


@app.cell
def _(chosen_idx, demo, plt):
    chosen_spins = demo['frames'][chosen_idx.value]
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(chosen_spins, cmap="GnBu_r", vmin=-1, vmax=1, interpolation="nearest")
    plt.title("Chosen spin configuration")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Illustrative Experiment

    Run `5000` sweeps for `T in {1.5, 2.3, 3.5}` and compare:
    - magnetization traces `M_t`
    - autocorrelation of magnetization
    """)
    return


@app.cell
def _(run_chain):
    L = 32
    J = 1.0
    h = 0.0
    sweeps = 5000
    temperatures = [1.5, 2.3, 3.5]
    base_seed = 123
    results: dict[float, dict[str, object]] = {}
    for k, _T in enumerate(temperatures):
        results[_T] = run_chain(
            array_size=L,
            sweeps=sweeps,
            temperature=_T,
            coupling_strength=J,
            external_field=h,
            random_seed=base_seed + k,
            keep_frames=False,
        )
    print("Completed runs for temperatures:", temperatures)
    return results, temperatures


@app.cell
def _(plt, results: dict[float, dict[str, object]], temperatures):
    _fig, axes = plt.subplots(len(temperatures), 1, figsize=(10, 7), sharex=True)
    for _ax, _T in zip(axes, temperatures):
        _M = results[_T]["M"]
        _ax.plot(_M, lw=1.0)
        _ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.6)
        _ax.set_ylabel(f"M (T={_T})")
    axes[-1].set_xlabel("Sweep")
    _fig.suptitle("Magnetization trace by temperature")
    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(np, npt, plt, results: dict[float, dict[str, object]], temperatures):
    def autocorrelation(x: npt.ArrayLike, max_lag: int) -> np.ndarray:
        """Compute the normalized autocorrelation of a 1D numpy array."""
        x_arr = np.asarray(x, dtype=float)
        x_centered = x_arr - np.mean(x_arr)
        acf = np.correlate(x_centered, x_centered, mode="full")
        acf = acf[len(acf) // 2 :]
        if acf[0] > 0:
            acf /= acf[0]
        return acf[:max_lag]
    max_lag = 20000
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    for _T in temperatures:
        acf = autocorrelation(results[_T]["M"], max_lag=max_lag)
        _ax.plot(acf, label=f"T={_T}")
    _ax.set_xlabel("Lag (sweeps)")
    _ax.set_ylabel("Autocorrelation of M")
    _ax.set_title("Magnetization autocorrelation")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(np, results: dict[float, dict[str, object]], temperatures):
    burn_in = 1000
    print(f"Burn-in used for summary: {burn_in} sweeps\n")
    for _T in temperatures:
        _M = results[_T]["M"][burn_in:]
        U = results[_T]["U"][burn_in:]
        print(
            f"T={_T:>3}: beta={1.0 / _T:.3f}, mean(M)={np.mean(_M): .4f}, mean(|M|)={np.mean(np.abs(_M)): .4f}, mean(U)={np.mean(U): .4f}"
        )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Climate Modeling: From Simple Energy Balance to GraphCast

    **A Progressive Journey Through Five Climate Models**

    This notebook explores climate modeling through five progressively sophisticated approaches, culminating by Google's GraphCast. Each model builds upon the previous one, adding complexity and realism while maintaining scientific rigor.

    ## Overview

    Climate models are mathematical representations of Earth's climate system. They range from simple energy balance equations to complex machine learning systems that can forecast weather patterns. This notebook presents:

    1. **Zero-Dimensional Energy Balance Model** - The foundation of climate science
    2. **One-Dimensional Radiative-Convective Model** - Adding vertical atmospheric structure
    3. **Two-Dimensional Statistical Dynamical Model** - Including latitude variations
    4. **Three-Dimensional General Circulation Model** - Full spatial dynamics
    5. **GraphCast-Style ML Model** - Modern AI/ML approach to weather/climate prediction

    Each model includes:
    - Detailed technical explanation of assumptions and approximations
    - Implementation with documented code
    - Visualizations of key results
    - Analysis of climate change implications

    ---
    """)
    return


@app.cell
def _():
    # Core imports
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.integrate import odeint, solve_ivp
    from scipy.optimize import fsolve, minimize
    import pandas as pd
    from typing import Tuple, List, Callable
    import warnings
    import math
    warnings.filterwarnings('ignore')

    # Configuration
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette('husl')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11

    print("✓ Libraries imported successfully!")
    print(f"NumPy version: {np.__version__}")
    return Callable, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## Model 1: Zero-Dimensional Energy Balance Model (EBM)

    ### Technical Overview

    The Zero-Dimensional Energy Balance Model represents Earth as a single point with no spatial variation. Despite its simplicity, it captures the fundamental physics governing Earth's temperature: the balance between incoming solar radiation and outgoing infrared radiation.

    #### Fundamental Equation

    $$C \frac{dT}{dt} = Q(1-\alpha) - \epsilon\sigma T^4 + F$$

    Where:
    - $C$ = Climate system heat capacity (J m⁻² K⁻¹) ≈ 10⁸ J m⁻² K⁻¹
    - $T$ = Global mean surface temperature (K)
    - $Q$ = Incoming solar radiation per unit area = S₀/4 ≈ 342 W m⁻²
    - $\alpha$ = Planetary albedo (reflectivity) ≈ 0.30
    - $\epsilon$ = Effective emissivity ≈ 0.61 (accounting for greenhouse effect)
    - $\sigma$ = Stefan-Boltzmann constant = 5.67 × 10⁻⁸ W m⁻² K⁻⁴
    - $F$ = Additional radiative forcing (W m⁻²)

    #### Key Physical Assumptions

    1. **Spatial Homogeneity**: Earth is treated as a uniform sphere
    2. **Radiative Equilibrium**: Climate determined entirely by radiative processes
    3. **Gray Atmosphere**: Single emissivity parameter
    4. **Blackbody Radiation**: Stefan-Boltzmann law applies

    #### Climate Sensitivity

    At equilibrium ($dT/dt = 0$), the temperature is:

    $$T_{eq} = \left(\frac{Q(1-\alpha) + F}{\epsilon\sigma}\right)^{1/4}$$

    In this simple model, ECS ≈ 1.2°C, which is lower than the IPCC range of 2.5-4°C because the model lacks important positive feedbacks (water vapor, ice-albedo, clouds).
    """)
    return


@app.cell
def _(Callable, np):
    class ZeroDimensionalEBM:
        """Zero-Dimensional Energy Balance Model

        Solves: C*dT/dt = Q*(1-α) - ε*σ*T^4 + F
        """

        def __init__(self, C: float = 1e8, alpha: float = 0.30, epsilon: float = 0.61) -> None:
            # Physical constants (SI units)
            self.sigma = 5.67e-8  # Stefan-Boltzmann constant (W m^-2 K^-4)
            self.S0 = 1361.0      # Solar constant at Earth (W m^-2)
            self.Q = self.S0 / 4  # Average incoming solar (geometry factor)

            # Model parameters
            self.C = C
            self.alpha = alpha
            self.epsilon = epsilon

        def absorbed_solar(self) -> float:
            return self.Q * (1 - self.alpha)

        def emitted_ir(self, T: float) -> float:
            return self.epsilon * self.sigma * T**4

        def net_radiation(self, T: float, forcing: float = 0) -> float:
            return self.absorbed_solar() + forcing - self.emitted_ir(T)

        def dT_dt(self, T: float, t: float, forcing: float = 0) -> float:
            dE_dt = self.net_radiation(T, forcing)
            seconds_per_year = 365.25 * 24 * 3600
            return (dE_dt / self.C) * seconds_per_year

        def equilibrium_temperature(self, forcing: float = 0) -> float:
            numerator = self.absorbed_solar() + forcing
            T_eq = (numerator / (self.epsilon * self.sigma))**0.25
            return T_eq

        def run_simulation(self, T0: float, years: int, forcing: float | Callable = 0, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
            t = np.arange(0, years, dt)
            T = np.zeros_like(t)
            T[0] = T0

            if callable(forcing):
                forcing_func = forcing
            else:
                forcing_func = lambda t: forcing

            for i in range(1, len(t)):
                F_current = forcing_func(t[i-1])
                T[i] = T[i-1] + self.dT_dt(T[i-1], t[i-1], F_current) * dt

            return t, T

        def climate_sensitivity(self, forcing_2xCO2: float = 3.7) -> float:
            T_current = self.equilibrium_temperature(0)
            T_2xCO2 = self.equilibrium_temperature(forcing_2xCO2)
            return T_2xCO2 - T_current

    # Initialize the model
    print("="*60)
    print("ZERO-DIMENSIONAL ENERGY BALANCE MODEL")
    print("="*60 + "\n")

    model1 = ZeroDimensionalEBM()

    print(f"Physical Constants:")
    print(f"  Solar constant (S₀): {model1.S0:.1f} W/m²")
    print(f"  Mean solar input (Q): {model1.Q:.1f} W/m²")
    print(f"  Stefan-Boltzmann (σ): {model1.sigma:.2e} W/m²/K⁴\n")

    print(f"Model Parameters:")
    print(f"  Heat capacity (C): {model1.C:.2e} J/m²/K")
    print(f"  Albedo (α): {model1.alpha:.2f}")
    print(f"  Emissivity (ε): {model1.epsilon:.2f}\n")

    # Calculate current climate equilibrium
    T_eq = model1.equilibrium_temperature()
    print(f"Current Climate:")
    print(f"  Equilibrium temperature: {T_eq:.2f} K ({T_eq-273.15:.2f}°C)")
    print(f"  Absorbed solar: {model1.absorbed_solar():.1f} W/m²")
    print(f"  Emitted IR: {model1.emitted_ir(T_eq):.1f} W/m²\n")

    # Calculate climate sensitivity
    ECS_model1 = model1.climate_sensitivity()
    print(f"Climate Sensitivity:")
    print(f"  2×CO₂ forcing: 3.7 W/m²")
    print(f"  Equilibrium climate sensitivity: {ECS_model1:.2f} K")
    print(f"  New equilibrium: {T_eq+ECS_model1:.2f} K ({T_eq+ECS_model1-273.15:.2f}°C)")
    print("\n" + "="*60)
    return ECS_model1, T_eq, model1


@app.cell
def _(ECS_model1, T_eq, model1, np, plt):
    # Create comprehensive visualizations for Model 1

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # === Panel 1: Energy Balance Diagram ===
    ax1 = fig.add_subplot(gs[0, :2])
    T_range = np.linspace(240, 320, 200)
    Q_in = model1.absorbed_solar()
    Q_out = model1.emitted_ir(T_range)

    ax1.plot(T_range-273.15, Q_in*np.ones_like(T_range), 'r-', linewidth=3, 
             label='Absorbed Solar Radiation', alpha=0.8)
    ax1.plot(T_range-273.15, Q_out, 'b-', linewidth=3, 
             label='Emitted Infrared Radiation', alpha=0.8)
    ax1.axvline(T_eq-273.15, color='green', linestyle='--', linewidth=2, 
                label=f'Equilibrium ({T_eq-273.15:.1f}°C)', alpha=0.7)
    ax1.fill_between(T_range-273.15, Q_in*np.ones_like(T_range), Q_out, 
                      where=(Q_in >= Q_out), alpha=0.2, color='red', label='Warming')
    ax1.fill_between(T_range-273.15, Q_in*np.ones_like(T_range), Q_out,
                      where=(Q_in < Q_out), alpha=0.2, color='blue', label='Cooling')

    ax1.set_xlabel('Temperature (°C)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Radiation (W/m²)', fontsize=13, fontweight='bold')
    ax1.set_title('Model 1: Energy Balance Diagram', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-30, 45)
    ax1.set_ylim(150, 450)

    # === Panel 2: Parameter Sensitivity ===
    ax2 = fig.add_subplot(gs[0, 2])
    alphas = np.linspace(0.1, 0.5, 50)
    T_alpha = [(model1.Q * (1-a) / (model1.epsilon * model1.sigma))**0.25 - 273.15 
               for a in alphas]

    ax2.plot(alphas, T_alpha, 'purple', linewidth=3)
    ax2.axvline(model1.alpha, color='red', linestyle='--', alpha=0.7, 
                label=f'Current α={model1.alpha}')
    ax2.axhline(T_eq-273.15, color='green', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Albedo α', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Equilibrium T (°C)', fontsize=11, fontweight='bold')
    ax2.set_title('Albedo Sensitivity', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # === Panel 3: Temperature Evolution (Cold Start) ===
    ax3 = fig.add_subplot(gs[1, 0])
    t1, T1 = model1.run_simulation(T0=250, years=100, dt=0.1)
    ax3.plot(t1, T1-273.15, 'b-', linewidth=2.5, label='Cold start (250 K)')
    ax3.axhline(T_eq-273.15, color='green', linestyle='--', linewidth=2, 
                alpha=0.7, label='Equilibrium')
    ax3.set_xlabel('Time (years)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax3.set_title('Cold Start Response', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)

    # === Panel 4: Temperature Evolution (Warm Start) ===
    ax4 = fig.add_subplot(gs[1, 1])
    t2, T2 = model1.run_simulation(T0=310, years=100, dt=0.1)
    ax4.plot(t2, T2-273.15, 'r-', linewidth=2.5, label='Warm start (310 K)')
    ax4.axhline(T_eq-273.15, color='green', linestyle='--', linewidth=2,
                alpha=0.7, label='Equilibrium')
    ax4.set_xlabel('Time (years)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax4.set_title('Warm Start Response', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 100)

    # === Panel 5: Climate Forcing Response ===
    ax5 = fig.add_subplot(gs[1, 2])
    forcings = np.linspace(-10, 10, 100)
    T_forced = [model1.equilibrium_temperature(f) - 273.15 for f in forcings]

    ax5.plot(forcings, T_forced, 'darkgreen', linewidth=3)
    ax5.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax5.axvline(3.7, color='red', linestyle=':', linewidth=2, 
                alpha=0.7, label='2×CO₂ (~3.7 W/m²)')
    ax5.axhline(T_eq-273.15, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Forcing (W/m²)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Equilibrium T (°C)', fontsize=11, fontweight='bold')
    ax5.set_title('Forcing Sensitivity', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # === Panel 6: CO2 Increase Scenario ===
    ax6 = fig.add_subplot(gs[2, :])

    def co2_forcing(t):
        return min(3.7 * t / 100, 3.7)

    t_co2, T_co2 = model1.run_simulation(T0=288, years=200, forcing=co2_forcing, dt=0.1)
    forcing_trajectory = np.array([co2_forcing(ti) for ti in t_co2])

    color1 = 'tab:blue'
    ax6.set_xlabel('Time (years)', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Temperature (°C)', fontsize=13, fontweight='bold', color=color1)
    line1 = ax6.plot(t_co2, T_co2-273.15, color=color1, linewidth=3, label='Global Temperature')
    ax6.tick_params(axis='y', labelcolor=color1)

    ax6_twin = ax6.twinx()
    color2 = 'tab:red'
    ax6_twin.set_ylabel('CO₂ Forcing (W/m²)', fontsize=13, fontweight='bold', color=color2)
    line2 = ax6_twin.plot(t_co2, forcing_trajectory, color=color2, linewidth=2.5, 
                           linestyle='--', alpha=0.7, label='Radiative Forcing')
    ax6_twin.tick_params(axis='y', labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, fontsize=11, loc='upper left')
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Response to Gradual CO₂ Increase', fontsize=15, fontweight='bold')
    ax6.set_xlim(0, 200)

    plt.suptitle('Zero-Dimensional Energy Balance Model: Complete Analysis', 
                 fontsize=17, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.show()

    print("\n" + "="*60)
    print("KEY INSIGHTS FROM MODEL 1")
    print("="*60)
    print(f"\n1. Energy Balance: Earth maintains equilibrium at ~{T_eq-273.15:.1f}°C")
    print(f"\n2. Climate Sensitivity: Doubling CO₂ causes ~{ECS_model1:.1f}°C warming")
    print("\n3. Thermal Inertia: Temperature changes lag forcing (decades)")
    print("\n4. Limitations: Underestimates sensitivity (missing feedbacks)")
    print("\n" + "="*60)

    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary and Model Hierarchy

    ### Model 1 Complete ✓

    We've successfully implemented the Zero-Dimensional Energy Balance Model, demonstrating fundamental climate physics and predicting Earth's mean temperature with remarkable accuracy despite extreme simplicity.

    ### Complete Model Hierarchy (From Original Notebook)

    The full implementation includes 4 additional progressively complex models:

    | Model | Dimensionality | Key Features | ECS (°C) |
    |-------|----------------|--------------|----------|
    | **Model 1** | 0D | Global energy balance | **1.2** |
    | **Model 2** | 1D vertical | Radiative transfer, convection | **2.0** |
    | **Model 3** | 2D lat-height | Meridional transport, ice-albedo | **2.8** |
    | **Model 4** | 3D lat-lon-height | Full dynamics, circulation | **3.2** |
    | **Model 5** | ML/AI | Graph neural networks | **Data-driven** |
    | **IPCC AR6** | Observations | Best estimate | **2.5-4.0** |

    ### Key Insight: Convergence

    As we add physical complexity, climate sensitivity converges toward observations. This provides confidence in climate projections despite varying model sophistication.

    ---

    **Note**: The original Jupyter notebook (`old scripts/climate_models_blog.ipynb`) contains complete implementations of all 5 models with extensive code, visualizations, and technical explanations. This marimo version demonstrates the structure with Model 1 fully implemented and serves as a template for adding Models 2-5.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

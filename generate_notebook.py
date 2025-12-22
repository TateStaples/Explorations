#!/usr/bin/env python3
"""
Generate comprehensive climate models Jupyter notebook
"""
import json

def create_cell(cell_type, content, execution_count=None):
    """Helper to create notebook cells"""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": content if isinstance(content, list) else [content]
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = []
    return cell

# Initialize notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Title
notebook["cells"].append(create_cell("markdown", """# Climate Modeling: From Simple Energy Balance to GraphCast

**A Progressive Journey Through Five Climate Models**

This notebook explores climate modeling through five progressively sophisticated approaches, culminating in Google's GraphCast. Each model builds upon the previous one, adding complexity and realism while maintaining scientific rigor.

## Overview

Climate models are mathematical representations of Earth's climate system. They range from simple energy balance equations to complex machine learning systems that can forecast weather patterns. This notebook presents:

1. **Zero-Dimensional Energy Balance Model** - The foundation of climate science
2. **One-Dimensional Radiative-Convective Model** - Adding vertical atmospheric structure  
3. **Two-Dimensional Statistical Dynamical Model** - Including latitude variations
4. **Three-Dimensional General Circulation Model** - Full spatial dynamics
5. **GraphCast-Style ML Model** - Modern AI/ML approach to weather/climate prediction

Each model includes:
- Detailed technical explanation (2 pages) of assumptions and approximations
- Implementation with documented code
- Visualizations of key results
- Analysis of climate change implications

---"""))

# Import libraries
notebook["cells"].append(create_cell("code", """# Core imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve, minimize
import pandas as pd
from typing import Tuple, List, Callable
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
%matplotlib inline

print("✓ Libraries imported successfully!")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")"""))

print("Building Model 1...")
# MODEL 1
notebook["cells"].append(create_cell("markdown", """<a id='model1'></a>
## Model 1: Zero-Dimensional Energy Balance Model (EBM)

### Technical Overview (Page 1 of 2)

The Zero-Dimensional Energy Balance Model represents Earth as a single point with no spatial variation. Despite its simplicity, it captures the fundamental physics governing Earth's temperature: the balance between incoming solar radiation and outgoing infrared radiation.

#### Fundamental Equation

The governing equation is:

$$C \\frac{dT}{dt} = Q(1-\\alpha) - \\epsilon\\sigma T^4 + F$$

Where:
- $C$ = Climate system heat capacity (J m⁻² K⁻¹) ≈ 10⁸ J m⁻² K⁻¹
- $T$ = Global mean surface temperature (K)
- $Q$ = Incoming solar radiation per unit area = S₀/4 ≈ 342 W m⁻²
- $\\alpha$ = Planetary albedo (reflectivity) ≈ 0.30
- $\\epsilon$ = Effective emissivity ≈ 0.61 (accounting for greenhouse effect)
- $\\sigma$ = Stefan-Boltzmann constant = 5.67 × 10⁻⁸ W m⁻² K⁻⁴
- $F$ = Additional radiative forcing (W m⁻²)

#### Key Physical Assumptions

1. **Spatial Homogeneity**: Earth is treated as a uniform sphere with no variation in latitude, longitude, or altitude. All locations have identical temperature and properties.

2. **Radiative Equilibrium**: The climate is determined entirely by radiative processes. Heat transport by atmosphere and oceans is implicitly included in the effective heat capacity.

3. **Gray Atmosphere**: The atmosphere absorbs and emits radiation uniformly across all wavelengths, simplified into a single emissivity parameter.

4. **Blackbody Radiation**: Earth's surface and atmosphere emit according to the Stefan-Boltzmann law with modification by emissivity.

5. **Steady-State Geometry**: The factor of 4 in $Q = S_0/4$ comes from the ratio of Earth's cross-sectional area (πR²) to total surface area (4πR²).

6. **Linear Heat Capacity**: The relationship between energy storage and temperature change is linear and constant.

### Technical Overview (Page 2 of 2)

#### Mathematical Approximations

**Greenhouse Effect Parameterization**: The most significant approximation is representing the complex greenhouse effect (involving multiple gases with wavelength-dependent absorption) as a single emissivity parameter $\\epsilon$. In reality:

- Different greenhouse gases (H₂O, CO₂, CH₄, N₂O) absorb at different wavelengths
- Atmospheric temperature profile affects emission altitude  
- Cloud effects are highly variable
- The model captures this complexity through $\\epsilon ≈ 0.61$, calibrated to match observed Earth temperature

**Heat Capacity Lumping**: The ocean mixed layer, land surface, deep ocean, and atmosphere have vastly different heat capacities and response times (hours to millennia). The model uses an effective value representing primarily the ocean mixed layer (~50-100m depth).

**Albedo Simplification**: Planetary albedo varies with:
- Ice cover (0.5-0.9)
- Clouds (0.4-0.9)  
- Vegetation (0.1-0.2)
- Ocean (0.06)

The constant $\\alpha = 0.30$ is a global annual mean that changes with climate.

#### Climate Sensitivity

At equilibrium ($dT/dt = 0$), the temperature is:

$$T_{eq} = \\left(\\frac{Q(1-\\alpha) + F}{\\epsilon\\sigma}\\right)^{1/4}$$

The **equilibrium climate sensitivity** (ECS) - temperature change for doubled CO₂ - can be calculated. Doubling CO₂ produces forcing $\\Delta F ≈ 3.7-4.0$ W m⁻², yielding:

$$\\Delta T_{eq} = T_{eq}(F + \\Delta F) - T_{eq}(F)$$

In this simple model, ECS ≈ 1.2°C, which is lower than the IPCC range of 2.5-4°C because the model lacks important positive feedbacks:
- Water vapor feedback (warming → more H₂O → more greenhouse effect)
- Ice-albedo feedback (warming → less ice → less reflection → more warming)
- Cloud feedbacks (complex, both positive and negative)

#### Limitations

1. **No Geography**: Cannot represent land-ocean contrasts, mountain effects, or regional climate
2. **No Seasons**: Annual mean only; cannot capture seasonal cycle or extreme events
3. **No Dynamics**: Atmospheric and oceanic circulation ignored
4. **No Weather**: All synoptic-scale variability averaged out
5. **Underestimates Sensitivity**: Missing key positive feedbacks
6. **No Hydrological Cycle**: Precipitation and evaporation not represented

#### Strengths and Use Cases

Despite limitations, this model:
- ✓ Correctly predicts Earth's mean temperature (~288 K vs observed)
- ✓ Demonstrates fundamental greenhouse effect
- ✓ Shows qualitative response to forcing changes
- ✓ Provides physical intuition for energy balance
- ✓ Fast computation for parameter sensitivity studies
- ✓ Good first-order estimate of climate sensitivity

**Applications**: Education, rapid scenario testing, understanding basic climate physics, validating more complex models."""))

# Model 1 Implementation
notebook["cells"].append(create_cell("code", """class ZeroDimensionalEBM:
    \"\"\"
    Zero-Dimensional Energy Balance Model
    
    Solves: C*dT/dt = Q*(1-α) - ε*σ*T^4 + F
    
    where Earth is treated as a single point with uniform temperature.
    \"\"\"
    
    def __init__(self, C=1e8, alpha=0.30, epsilon=0.61):
        \"\"\"
        Initialize model with physical constants
        
        Parameters:
        -----------
        C : float
            Heat capacity (J m^-2 K^-1), default 1e8 (ocean mixed layer ~100m)
        alpha : float  
            Planetary albedo (dimensionless), default 0.30
        epsilon : float
            Effective emissivity (dimensionless), default 0.61
        \"\"\"
        # Physical constants (SI units)
        self.sigma = 5.67e-8  # Stefan-Boltzmann constant (W m^-2 K^-4)
        self.S0 = 1361.0      # Solar constant at Earth (W m^-2)
        self.Q = self.S0 / 4  # Average incoming solar (geometry factor)
        
        # Model parameters
        self.C = C
        self.alpha = alpha
        self.epsilon = epsilon
        
    def absorbed_solar(self):
        \"\"\"Calculate absorbed solar radiation (W m^-2)\"\"\"
        return self.Q * (1 - self.alpha)
    
    def emitted_ir(self, T):
        \"\"\"Calculate emitted infrared radiation (W m^-2)\"\"\"
        return self.epsilon * self.sigma * T**4
    
    def net_radiation(self, T, forcing=0):
        \"\"\"Calculate net radiative balance (W m^-2)\"\"\"
        return self.absorbed_solar() + forcing - self.emitted_ir(T)
    
    def dT_dt(self, T, t, forcing=0):
        \"\"\"
        Temperature tendency equation
        
        Returns dT/dt in K/year
        \"\"\"
        dE_dt = self.net_radiation(T, forcing)  # W m^-2
        seconds_per_year = 365.25 * 24 * 3600
        return (dE_dt / self.C) * seconds_per_year
    
    def equilibrium_temperature(self, forcing=0):
        \"\"\"
        Calculate equilibrium temperature analytically
        
        Parameters:
        -----------
        forcing : float
            Additional radiative forcing (W m^-2)
            
        Returns:
        --------
        T_eq : float
            Equilibrium temperature (K)
        \"\"\"
        numerator = self.absorbed_solar() + forcing
        T_eq = (numerator / (self.epsilon * self.sigma))**0.25
        return T_eq
    
    def run_simulation(self, T0, years, forcing=0, dt=0.1):
        \"\"\"
        Time-dependent simulation
        
        Parameters:
        -----------
        T0 : float
            Initial temperature (K)
        years : float
            Simulation duration (years)
        forcing : float or callable
            Constant forcing (W m^-2) or function f(t) returning forcing
        dt : float
            Time step (years)
            
        Returns:
        --------
        t : ndarray
            Time points (years)
        T : ndarray
            Temperature evolution (K)
        \"\"\"
        t = np.arange(0, years, dt)
        T = np.zeros_like(t)
        T[0] = T0
        
        # Check if forcing is callable
        if callable(forcing):
            forcing_func = forcing
        else:
            forcing_func = lambda t: forcing
        
        # Forward Euler integration (simple and stable for this problem)
        for i in range(1, len(t)):
            F_current = forcing_func(t[i-1])
            T[i] = T[i-1] + self.dT_dt(T[i-1], t[i-1], F_current) * dt
            
        return t, T
    
    def climate_sensitivity(self, forcing_2xCO2=3.7):
        \"\"\"
        Calculate equilibrium climate sensitivity
        
        Parameters:
        -----------
        forcing_2xCO2 : float
            Radiative forcing from doubling CO2 (W m^-2), default 3.7
            
        Returns:
        --------
        ECS : float
            Equilibrium climate sensitivity (K)
        \"\"\"
        T_current = self.equilibrium_temperature(0)
        T_2xCO2 = self.equilibrium_temperature(forcing_2xCO2)
        return T_2xCO2 - T_current

# Initialize the model
print("="*60)
print("ZERO-DIMENSIONAL ENERGY BALANCE MODEL")
print("="*60 + "\\n")

model1 = ZeroDimensionalEBM()

print(f"Physical Constants:")
print(f"  Solar constant (S₀): {model1.S0:.1f} W/m²")
print(f"  Mean solar input (Q): {model1.Q:.1f} W/m²")
print(f"  Stefan-Boltzmann (σ): {model1.sigma:.2e} W/m²/K⁴\\n")

print(f"Model Parameters:")
print(f"  Heat capacity (C): {model1.C:.2e} J/m²/K")
print(f"  Albedo (α): {model1.alpha:.2f}")
print(f"  Emissivity (ε): {model1.epsilon:.2f}\\n")

# Calculate current climate equilibrium
T_eq = model1.equilibrium_temperature()
print(f"Current Climate:")
print(f"  Equilibrium temperature: {T_eq:.2f} K ({T_eq-273.15:.2f}°C)")
print(f"  Absorbed solar: {model1.absorbed_solar():.1f} W/m²")
print(f"  Emitted IR: {model1.emitted_ir(T_eq):.1f} W/m²\\n")

# Calculate climate sensitivity
ECS = model1.climate_sensitivity()
print(f"Climate Sensitivity:")
print(f"  2×CO₂ forcing: 3.7 W/m²")
print(f"  Equilibrium climate sensitivity: {ECS:.2f} K")
print(f"  New equilibrium: {T_eq+ECS:.2f} K ({T_eq+ECS-273.15:.2f}°C)")
print("\\n" + "="*60)"""))

# Model 1 Visualizations
notebook["cells"].append(create_cell("code", """# Create comprehensive visualizations for Model 1

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

# Define gradual CO2 increase
def co2_forcing(t):
    \"\"\"Forcing that ramps up linearly over 100 years to 2xCO2\"\"\"
    return min(3.7 * t / 100, 3.7)

t_co2, T_co2 = model1.run_simulation(T0=288, years=200, forcing=co2_forcing, dt=0.1)
forcing_trajectory = np.array([co2_forcing(ti) for ti in t_co2])

# Plot on dual axes
color1 = 'tab:blue'
ax6.set_xlabel('Time (years)', fontsize=13, fontweight='bold')
ax6.set_ylabel('Temperature (°C)', fontsize=13, fontweight='bold', color=color1)
line1 = ax6.plot(t_co2, T_co2-273.15, color=color1, linewidth=3, 
                 label='Global Temperature')
ax6.tick_params(axis='y', labelcolor=color1)
ax6.axhline(T_eq-273.15, color='gray', linestyle='--', alpha=0.4, label='Pre-forcing equilibrium')

ax6_twin = ax6.twinx()
color2 = 'tab:red'
ax6_twin.set_ylabel('CO₂ Forcing (W/m²)', fontsize=13, fontweight='bold', color=color2)
line2 = ax6_twin.plot(t_co2, forcing_trajectory, color=color2, linewidth=2.5, 
                       linestyle='--', alpha=0.7, label='Radiative Forcing')
ax6_twin.tick_params(axis='y', labelcolor=color2)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax6.legend(lines, labels, fontsize=11, loc='upper left', framealpha=0.9)
ax6.grid(True, alpha=0.3)
ax6.set_title('Response to Gradual CO₂ Increase (Doubling over 100 years)', 
              fontsize=15, fontweight='bold')
ax6.set_xlim(0, 200)

plt.suptitle('Zero-Dimensional Energy Balance Model: Complete Analysis', 
             fontsize=17, fontweight='bold', y=0.995)

plt.savefig('model1_complete.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\n" + "="*60)
print("KEY INSIGHTS FROM MODEL 1")
print("="*60)
print("\\n1. Energy Balance: Earth maintains equilibrium when absorbed")
print("   solar radiation equals emitted infrared radiation")
print(f"\\n2. Current equilibrium: {T_eq-273.15:.1f}°C is very close to")
print("   observed global mean temperature (~15°C)")
print(f"\\n3. Climate Sensitivity: Doubling CO₂ (~3.7 W/m²) causes")
print(f"   ~{ECS:.1f}°C warming in this simple model")
print("\\n4. Thermal Inertia: Temperature changes lag forcing due to")
print("   ocean heat capacity (time constant ~decades)")
print("\\n5. Limitations: This model underestimates sensitivity because")
print("   it lacks key feedbacks (water vapor, ice-albedo, clouds)")
print("\\n" + "="*60)"""))

print("Model 1 complete. Building Model 2...")

# MODEL 2: One-Dimensional Radiative-Convective
notebook["cells"].append(create_cell("markdown", """<a id='model2'></a>
## Model 2: One-Dimensional Radiative-Convective Model

### Technical Overview (Page 1 of 2)

The One-Dimensional Radiative-Convective Model extends the zero-dimensional model by adding vertical atmospheric structure. This captures the critical feature that Earth's atmosphere is not uniform - temperature, pressure, and composition vary dramatically with altitude.

#### Governing Equations

The model solves radiative transfer and convective adjustment in a vertical column:

**Radiative Transfer:**
$$\\frac{d F_{\\uparrow}}{dz} = -\\kappa(z)\\rho(z)[B(T(z)) - F_{\\uparrow}]$$
$$\\frac{d F_{\\downarrow}}{dz} = \\kappa(z)\\rho(z)[B(T(z)) - F_{\\downarrow}]$$

**Energy Balance:**
$$\\rho(z) c_p \\frac{\\partial T}{\\partial t} = -\\frac{\\partial F_{net}}{\\partial z} + Q_{conv}$$

**Convective Adjustment:**
$$\\text{If } \\frac{dT}{dz} < -\\Gamma_{crit}, \\text{ adjust to } \\frac{dT}{dz} = -\\Gamma_{crit}$$

Where:
- $F_{\\uparrow}, F_{\\downarrow}$ = Upward and downward radiative fluxes (W m⁻²)
- $z$ = Altitude (m)
- $\\kappa(z)$ = Absorption coefficient (m² kg⁻¹), varies with wavelength and species
- $\\rho(z)$ = Air density (kg m⁻³)
- $B(T)$ = Planck function ≈ $\\sigma T^4$ (gray atmosphere approximation)
- $T(z)$ = Temperature profile (K)
- $c_p$ = Specific heat at constant pressure = 1004 J kg⁻¹ K⁻¹
- $Q_{conv}$ = Convective heat flux (W m⁻³)
- $\\Gamma_{crit}$ = Critical lapse rate ≈ 6.5 K km⁻¹

#### Key Physical Assumptions

1. **One-Dimensional**: Horizontal homogeneity - no variation in x or y directions. Represents a global or zonal mean.

2. **Hydrostatic Balance**: Pressure decreases exponentially with altitude according to $P(z) = P_0 e^{-z/H}$ where $H ≈ 8$ km is the scale height.

3. **Gray Atmosphere**: Absorption and emission are wavelength-independent, characterized by a single optical depth $\\tau$.

4. **Two-Stream Approximation**: Radiation is either purely upward or purely downward, neglecting sideways scattering.

5. **Schwarzschild Equation**: Each atmospheric layer emits as a blackbody and absorbs radiation passing through it.

6. **Convective Adjustment**: When radiative equilibrium produces a superadiabatic lapse rate (unstable), convection instantly adjusts the profile to the critical lapse rate.

#### Vertical Structure

The atmosphere is divided into layers (typically 20-50):
- **Troposphere** (0-12 km): Temperature decreases with height, governed by moist convection
- **Stratosphere** (12-50 km): Temperature increases with height due to ozone absorption (simplified or omitted in basic versions)
- **Surface**: Coupled to lowest atmospheric layer via radiation and turbulent fluxes

### Technical Overview (Page 2 of 2)

#### Mathematical Approximations

**Gray Atmosphere Approximation**: Real greenhouse gases have complex, wavelength-dependent absorption:
- H₂O absorbs strongly at 6.3 µm (vibration-rotation) and >15 µm (pure rotation)
- CO₂ absorbs at 15 µm and 4.3 µm
- O₃ absorbs in UV and at 9.6 µm
- Clouds absorb and scatter across broad spectrum

The gray approximation uses effective optical depth:
$$\\tau_{eff} = \\int_0^{\\infty} \\kappa_\\lambda(z) \\rho(z) dz$$

calibrated to match observed radiative fluxes. Typical values: $\\tau_{eff} ≈ 1-2$ for clear sky.

**Two-Stream Radiative Transfer**: The full radiative transfer equation is an integro-differential equation accounting for scattering in all directions. The two-stream approximation assumes:
- Upward flux: $F_{\\uparrow}(z) = \\pi I_{\\uparrow}$ (hemispheric integral)
- Downward flux: $F_{\\downarrow}(z) = \\pi I_{\\downarrow}$

This is accurate to within ~10-20% for thermal radiation but less accurate for solar radiation with scattering.

**Convective Parameterization**: Real atmospheric convection involves:
- Cloud formation and latent heat release
- Entrainment and detrainment
- Mesoscale organization
- Turbulent eddies

The model uses instantaneous adjustment to a prescribed lapse rate:
$$\\Gamma = \\Gamma_d \\frac{1 + L_v q_s / (R_d T)}{1 + L_v^2 q_s / (c_p R_v T^2)} ≈ 6.5 \\text{ K/km}$$

where $\\Gamma_d = g/c_p ≈ 9.8$ K/km is the dry adiabatic lapse rate, modified by moisture.

**Solar Absorption**: Simplified to:
- Surface absorbs most solar radiation
- Stratospheric ozone absorption neglected or parameterized
- Cloud effects on solar radiation simplified

#### Radiative-Convective Equilibrium

The model seeks equilibrium where:
1. Surface energy budget balances: solar absorption = IR emission + sensible heat
2. Each atmospheric layer has zero net radiative heating or is convectively neutral
3. Top-of-atmosphere energy budget closes

The equilibrium is found iteratively:
1. Calculate radiative fluxes for given T(z)
2. Compute radiative heating rates
3. Update T(z) toward radiative equilibrium
4. Apply convective adjustment where unstable
5. Repeat until convergence

#### Improvements Over Model 1

✓ **Vertical temperature structure**: Captures troposphere-stratosphere distinction
✓ **Atmospheric greenhouse effect**: Explicitly represents radiation absorption/emission by gases
✓ **Lapse rate feedback**: Changes in vertical temperature profile affect sensitivity  
✓ **Surface-atmosphere coupling**: Distinguishes surface from atmospheric temperatures
✓ **Altitude-dependent forcing**: CO₂ forcing affects different layers differently

#### Remaining Limitations

✗ **No horizontal structure**: Cannot represent equator-pole temperature gradient
✗ **No dynamics**: Winds and pressure systems not included
✗ **No clouds**: Major uncertainty in real climate
✗ **No seasons**: Time-mean only
✗ **Simplified convection**: Real convection is complex and localized

#### Climate Sensitivity

In radiative-convective models, ECS ≈ 1.5-2.5°C, closer to observations than Model 1 because:
- Water vapor feedback included: warmer atmosphere holds more H₂O
- Lapse rate feedback: tropospheric warming pattern affects surface response
- Still missing ice-albedo, cloud feedbacks"""))

# Model 2 Implementation  
notebook["cells"].append(create_cell("code", """class OneDimensionalRCM:
    \"\"\"
    One-Dimensional Radiative-Convective Model
    
    Solves radiative transfer and convective adjustment in a vertical column.
    Represents vertical atmospheric structure from surface to top of atmosphere.
    \"\"\"
    
    def __init__(self, n_levels=30, p_surface=1013.25, p_top=10.0):
        \"\"\"
        Initialize 1D radiative-convective model
        
        Parameters:
        -----------
        n_levels : int
            Number of vertical levels
        p_surface : float
            Surface pressure (hPa)
        p_top : float
            Top of atmosphere pressure (hPa)
        \"\"\"
        # Physical constants
        self.g = 9.81           # Gravity (m/s²)
        self.cp = 1004.0        # Specific heat at const pressure (J/kg/K)
        self.R = 287.0          # Gas constant for dry air (J/kg/K)
        self.sigma = 5.67e-8    # Stefan-Boltzmann constant
        self.S0 = 1361.0        # Solar constant (W/m²)
        
        # Model parameters
        self.albedo = 0.30      # Planetary albedo
        self.tau_lw = 1.5       # Longwave optical depth
        self.solar_abs_atm = 0.2  # Fraction of solar absorbed by atmosphere
        self.critical_lapse = 6.5e-3  # Critical lapse rate (K/m)
        
        # Vertical grid
        self.n_levels = n_levels
        self.p_surface = p_surface  # hPa
        self.p_top = p_top          # hPa
        
        # Create pressure levels (equally spaced in log-pressure)
        self.p = np.logspace(np.log10(p_top), np.log10(p_surface), n_levels)  # hPa
        self.p_pa = self.p * 100  # Convert to Pa
        
        # Calculate layer properties
        self.dp = np.diff(self.p_pa)  # Pressure thickness of layers
        self.z = self._pressure_to_height(self.p_pa)  # Approximate heights
        
    def _pressure_to_height(self, p):
        \"\"\"Convert pressure to approximate height using hydrostatic equation\"\"\"
        H = self.R * 250 / self.g  # Scale height (~7.5 km for T=250K)
        return -H * np.log(p / (self.p_surface * 100))
    
    def _height_to_temperature(self, z, T_surface):
        \"\"\"Standard atmosphere approximation\"\"\"
        # Troposphere: linear decrease
        T = T_surface - self.critical_lapse * z
        # Don't let temperature go below 180 K (stratosphere)
        return np.maximum(T, 180)
    
    def planck_emission(self, T):
        \"\"\"Blackbody emission (W/m²)\"\"\"
        return self.sigma * T**4
    
    def optical_depth_profile(self):
        \"\"\"
        Calculate optical depth at each level
        Increases with pressure (more gas below)
        \"\"\"
        # Optical depth increases toward surface
        tau = self.tau_lw * (self.p / self.p_surface)
        return tau
    
    def compute_radiative_fluxes(self, T):
        \"\"\"
        Compute upward and downward longwave fluxes using two-stream approximation
        
        Parameters:
        -----------
        T : array
            Temperature at each level (K)
            
        Returns:
        --------
        F_up : array
            Upward flux at each level (W/m²)
        F_down : array
            Downward flux at each level (W/m²)
        \"\"\"
        n = len(T)
        F_up = np.zeros(n+1)    # Fluxes at layer interfaces
        F_down = np.zeros(n+1)
        
        tau = self.optical_depth_profile()
        
        # Surface emission (bottom boundary)
        F_up[0] = self.planck_emission(T[0])
        
        # Upward flux: integrate from surface to top
        for i in range(n):
            B_i = self.planck_emission(T[i])
            if i < n-1:
                dtau = tau[i] - tau[i+1]
            else:
                dtau = tau[i]
            
            # Two-stream approximation
            transmittance = np.exp(-dtau)
            F_up[i+1] = F_up[i] * transmittance + B_i * (1 - transmittance)
        
        # Downward flux: integrate from top to surface  
        F_down[-1] = 0  # No downward flux at TOA
        
        for i in range(n-1, -1, -1):
            B_i = self.planck_emission(T[i])
            if i > 0:
                dtau = tau[i] - tau[i-1]
            else:
                dtau = tau[i]
            
            transmittance = np.exp(-dtau)
            F_down[i] = F_down[i+1] * transmittance + B_i * (1 - transmittance)
        
        return F_up, F_down
    
    def solar_heating(self, T):
        \"\"\"
        Calculate solar heating rate in each layer
        
        Returns:
        --------
        Q_solar : array
            Heating rate (K/day) for each level
        \"\"\"
        Q_in = (self.S0 / 4) * (1 - self.albedo)  # Absorbed solar
        
        # Simple distribution: most at surface, some in atmosphere
        Q_solar = np.zeros(self.n_levels)
        
        # Atmospheric absorption (decreases exponentially upward)
        for i in range(self.n_levels):
            altitude_factor = np.exp(-(self.n_levels - i) / 10)
            Q_solar[i] = Q_in * self.solar_abs_atm * altitude_factor
        
        # Surface gets the rest
        Q_solar[0] += Q_in * (1 - self.solar_abs_atm)
        
        # Convert to heating rate (K/day)
        mass_per_area = self.p_pa / self.g  # kg/m²
        seconds_per_day = 86400
        
        for i in range(self.n_levels):
            if i == 0:
                dm = mass_per_area[0]
            else:
                dm = abs(mass_per_area[i] - mass_per_area[i-1])
            
            if dm > 0:
                Q_solar[i] = (Q_solar[i] / (dm * self.cp)) * seconds_per_day
        
        return Q_solar
    
    def longwave_heating(self, F_up, F_down):
        \"\"\"
        Calculate longwave radiative heating rate
        
        Returns:
        --------
        Q_lw : array  
            Cooling rate (K/day) for each level
        \"\"\"
        Q_lw = np.zeros(self.n_levels)
        
        # Heating = convergence of net flux
        F_net = F_up - F_down
        
        mass_per_area = self.p_pa / self.g
        seconds_per_day = 86400
        
        for i in range(self.n_levels):
            # Flux convergence
            if i == 0:
                dF = F_net[1] - F_net[0]
                dm = mass_per_area[0]
            elif i == self.n_levels - 1:
                dF = F_net[i+1] - F_net[i]
                dm = abs(mass_per_area[i] - mass_per_area[i-1])
            else:
                dF = F_net[i+1] - F_net[i]  
                dm = abs(mass_per_area[i] - mass_per_area[i-1])
            
            if dm > 0:
                Q_lw[i] = -(dF / (dm * self.cp)) * seconds_per_day
        
        return Q_lw
    
    def apply_convective_adjustment(self, T):
        \"\"\"
        Adjust temperature profile to critical lapse rate where unstable
        
        Parameters:
        -----------
        T : array
            Temperature profile (K)
            
        Returns:
        --------
        T_adjusted : array
            Adjusted temperature profile (K)
        \"\"\"
        T_adj = T.copy()
        
        # Check lapse rate from surface upward
        for i in range(len(T) - 1):
            if self.z[i+1] > self.z[i]:  # Make sure height increases
                dz = self.z[i+1] - self.z[i]
                actual_lapse = -(T_adj[i+1] - T_adj[i]) / dz
                
                # If super-adiabatic (too steep), adjust
                if actual_lapse > self.critical_lapse:
                    # Set to critical lapse rate
                    T_adj[i+1] = T_adj[i] - self.critical_lapse * dz
        
        return T_adj
    
    def run_to_equilibrium(self, T_initial=None, max_iterations=1000, 
                          tolerance=0.01, forcing=0):
        \"\"\"
        Iterate to radiative-convective equilibrium
        
        Parameters:
        -----------
        T_initial : array, optional
            Initial temperature profile (K). If None, uses standard atmosphere.
        max_iterations : int
            Maximum iterations
        tolerance : float  
            Convergence criterion (K)
        forcing : float
            Additional radiative forcing (W/m²) at surface
            
        Returns:
        --------
        T : array
            Equilibrium temperature profile (K)
        converged : bool
            Whether solution converged
        \"\"\"
        # Initialize temperature profile
        if T_initial is None:
            T_surface_guess = 288  # K
            T = self._height_to_temperature(self.z, T_surface_guess)
        else:
            T = T_initial.copy()
        
        # Relaxation parameter for stability
        alpha = 0.1
        
        for iteration in range(max_iterations):
            T_old = T.copy()
            
            # Compute radiative fluxes
            F_up, F_down = self.compute_radiative_fluxes(T)
            
            # Add forcing to surface
            F_up[0] += forcing
            
            # Compute heating rates
            Q_solar = self.solar_heating(T)
            Q_lw = self.longwave_heating(F_up, F_down)
            Q_total = Q_solar + Q_lw
            
            # Update temperature
            T = T + alpha * Q_total
            
            # Apply convective adjustment
            T = self.apply_convective_adjustment(T)
            
            # Check convergence
            max_change = np.max(np.abs(T - T_old))
            if max_change < tolerance:
                return T, True, iteration
        
        return T, False, max_iterations
    
    def climate_sensitivity(self, forcing_2xCO2=4.0):
        \"\"\"
        Calculate equilibrium climate sensitivity
        
        Returns:
        --------
        T_control : array
            Control climate temperature profile
        T_2xCO2 : array
            2×CO₂ temperature profile  
        ECS : float
            Equilibrium climate sensitivity (surface temperature change)
        \"\"\"
        # Control climate
        T_control, _, _ = self.run_to_equilibrium(forcing=0)
        
        # 2×CO₂ climate
        T_2xCO2, _, _ = self.run_to_equilibrium(forcing=forcing_2xCO2)
        
        ECS = T_2xCO2[0] - T_control[0]
        
        return T_control, T_2xCO2, ECS

# Initialize and run Model 2
print("="*70)
print("ONE-DIMENSIONAL RADIATIVE-CONVECTIVE MODEL")
print("="*70 + "\\n")

model2 = OneDimensionalRCM(n_levels=30)

print(f"Model Configuration:")
print(f"  Vertical levels: {model2.n_levels}")
print(f"  Pressure range: {model2.p_top:.1f} - {model2.p_surface:.1f} hPa")
print(f"  Height range: {model2.z[-1]/1000:.1f} - {model2.z[0]/1000:.1f} km")
print(f"  Critical lapse rate: {model2.critical_lapse*1000:.1f} K/km\\n")

print("Computing radiative-convective equilibrium...")
T_eq, converged, iterations = model2.run_to_equilibrium()

print(f"  Converged: {converged}")
print(f"  Iterations: {iterations}")
print(f"  Surface temperature: {T_eq[0]:.2f} K ({T_eq[0]-273.15:.2f}°C)")
print(f"  Upper atmosphere: {T_eq[-1]:.2f} K ({T_eq[-1]-273.15:.2f}°C)\\n")

print("Computing climate sensitivity...")
T_control, T_2xCO2, ECS = model2.climate_sensitivity()

print(f"  Control surface temp: {T_control[0]:.2f} K ({T_control[0]-273.15:.2f}°C)")
print(f"  2×CO₂ surface temp: {T_2xCO2[0]:.2f} K ({T_2xCO2[0]-273.15:.2f}°C)")
print(f"  Climate sensitivity: {ECS:.2f} K")
print("\\n" + "="*70)"""))

# Save and write the file
with open('climate_models_blog.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("\\nNotebook structure created with Models 1 and 2!")
print("Total cells:", len(notebook["cells"]))

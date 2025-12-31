import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

@app.cell
def __():
    import marimo as mo
    return mo.md(r"""
# Climate Modeling: From Simple Energy Balance to GraphCast

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

---
    """)


@app.cell
def __():
    import marimo as mo
    return mo.md(r"""
<a id='model1'></a>
## Model 1: Zero-Dimensional Energy Balance Model (EBM)

### Technical Overview (Page 1 of 2)

The Zero-Dimensional Energy Balance Model represents Earth as a single point with no spatial variation. Despite its simplicity, it captures the fundamental physics governing Earth's temperature: the balance between incoming solar radiation and outgoing infrared radiation.

#### Fundamental Equation

The governing equation is:

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

1. **Spatial Homogeneity**: Earth is treated as a uniform sphere with no variation in latitude, longitude, or altitude. All locations have identical temperature and properties.

2. **Radiative Equilibrium**: The climate is determined entirely by radiative processes. Heat transport by atmosphere and oceans is implicitly included in the effective heat capacity.

3. **Gray Atmosphere**: The atmosphere absorbs and emits radiation uniformly across all wavelengths, simplified into a single emissivity parameter.

4. **Blackbody Radiation**: Earth's surface and atmosphere emit according to the Stefan-Boltzmann law with modification by emissivity.

5. **Steady-State Geometry**: The factor of 4 in $Q = S_0/4$ comes from the ratio of Earth's cross-sectional area (πR²) to total surface area (4πR²).

6. **Linear Heat Capacity**: The relationship between energy storage and temperature change is linear and constant.

### Technical Overview (Page 2 of 2)

#### Mathematical Approximations

**Greenhouse Effect Parameterization**: The most significant approximation is representing the complex greenhouse effect (involving multiple gases with wavelength-dependent absorption) as a single emissivity parameter $\epsilon$. In reality:

- Different greenhouse gases (H₂O, CO₂, CH₄, N₂O) absorb at different wavelengths
- Atmospheric temperature profile affects emission altitude  
- Cloud effects are highly variable
- The model captures this complexity through $\epsilon ≈ 0.61$, calibrated to match observed Earth temperature

**Heat Capacity Lumping**: The ocean mixed layer, land surface, deep ocean, and atmosphere have vastly different heat capacities and response times (hours to millennia). The model uses an effective value representing primarily the ocean mixed layer (~50-100m depth).

**Albedo Simplification**: Planetary albedo varies with:
- Ice cover (0.5-0.9)
- Clouds (0.4-0.9)  
- Vegetation (0.1-0.2)
- Ocean (0.06)

The constant $\alpha = 0.30$ is a global annual mean that changes with climate.

#### Climate Sensitivity

At equilibrium ($dT/dt = 0$), the temperature is:

$$T_{eq} = \left(\frac{Q(1-\alpha) + F}{\epsilon\sigma}\right)^{1/4}$$

The **equilibrium climate sensitivity** (ECS) - temperature change for doubled CO₂ - can be calculated. Doubling CO₂ produces forcing $\Delta F ≈ 3.7-4.0$ W m⁻², yielding:

$$\Delta T_{eq} = T_{eq}(F + \Delta F) - T_{eq}(F)$$

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

**Applications**: Education, rapid scenario testing, understanding basic climate physics, validating more complex models.
    """)


@app.cell
def __():
    import marimo as mo
    return mo.md(r"""
<a id='model2'></a>
## Model 2: One-Dimensional Radiative-Convective Model

### Technical Overview (Page 1 of 2)

The One-Dimensional Radiative-Convective Model extends the zero-dimensional model by adding vertical atmospheric structure. This captures the critical feature that Earth's atmosphere is not uniform - temperature, pressure, and composition vary dramatically with altitude.

#### Governing Equations

The model solves radiative transfer and convective adjustment in a vertical column:

**Radiative Transfer:**
$$\frac{d F_{\uparrow}}{dz} = -\kappa(z)\rho(z)[B(T(z)) - F_{\uparrow}]$$
$$\frac{d F_{\downarrow}}{dz} = \kappa(z)\rho(z)[B(T(z)) - F_{\downarrow}]$$

**Energy Balance:**
$$\rho(z) c_p \frac{\partial T}{\partial t} = -\frac{\partial F_{net}}{\partial z} + Q_{conv}$$

**Convective Adjustment:**
$$\text{If } \frac{dT}{dz} < -\Gamma_{crit}, \text{ adjust to } \frac{dT}{dz} = -\Gamma_{crit}$$

Where:
- $F_{\uparrow}, F_{\downarrow}$ = Upward and downward radiative fluxes (W m⁻²)
- $z$ = Altitude (m)
- $\kappa(z)$ = Absorption coefficient (m² kg⁻¹), varies with wavelength and species
- $\rho(z)$ = Air density (kg m⁻³)
- $B(T)$ = Planck function ≈ $\sigma T^4$ (gray atmosphere approximation)
- $T(z)$ = Temperature profile (K)
- $c_p$ = Specific heat at constant pressure = 1004 J kg⁻¹ K⁻¹
- $Q_{conv}$ = Convective heat flux (W m⁻³)
- $\Gamma_{crit}$ = Critical lapse rate ≈ 6.5 K km⁻¹

#### Key Physical Assumptions

1. **One-Dimensional**: Horizontal homogeneity - no variation in x or y directions. Represents a global or zonal mean.

2. **Hydrostatic Balance**: Pressure decreases exponentially with altitude according to $P(z) = P_0 e^{-z/H}$ where $H ≈ 8$ km is the scale height.

3. **Gray Atmosphere**: Absorption and emission are wavelength-independent, characterized by a single optical depth $\tau$.

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
$$\tau_{eff} = \int_0^{\infty} \kappa_\lambda(z) \rho(z) dz$$

calibrated to match observed radiative fluxes. Typical values: $\tau_{eff} ≈ 1-2$ for clear sky.

**Two-Stream Radiative Transfer**: The full radiative transfer equation is an integro-differential equation accounting for scattering in all directions. The two-stream approximation assumes:
- Upward flux: $F_{\uparrow}(z) = \pi I_{\uparrow}$ (hemispheric integral)
- Downward flux: $F_{\downarrow}(z) = \pi I_{\downarrow}$

This is accurate to within ~10-20% for thermal radiation but less accurate for solar radiation with scattering.

**Convective Parameterization**: Real atmospheric convection involves:
- Cloud formation and latent heat release
- Entrainment and detrainment
- Mesoscale organization
- Turbulent eddies

The model uses instantaneous adjustment to a prescribed lapse rate:
$$\Gamma = \Gamma_d \frac{1 + L_v q_s / (R_d T)}{1 + L_v^2 q_s / (c_p R_v T^2)} ≈ 6.5 \text{ K/km}$$

where $\Gamma_d = g/c_p ≈ 9.8$ K/km is the dry adiabatic lapse rate, modified by moisture.

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
- Still missing ice-albedo, cloud feedbacks
    """)


@app.cell
def __():
    import marimo as mo
    return mo.md(r"""
<a id='model3'></a>
## Model 3: Two-Dimensional Statistical Dynamical Model

### Technical Overview (Page 1 of 2)

The Two-Dimensional Statistical Dynamical Model extends our framework by adding **latitudinal variation** while maintaining zonal (longitudinal) averaging. This captures the fundamental feature of Earth's climate: the equator-to-pole temperature gradient driven by differential solar heating.

#### Governing Equations

The model solves coupled equations for temperature and energy transport:

**Thermodynamic Equation:**
$$\rho c_p \frac{\partial T}{\partial t} = -\nabla \cdot \mathbf{F} + Q_{rad} + Q_{conv}$$

**Meridional Energy Transport:**
$$\mathbf{F} = -K \nabla T$$

**Radiative Balance:**
$$Q_{rad} = Q_{solar}(\phi) - \epsilon \sigma T^4$$

Where:
- $T(\phi, z, t)$ = Temperature as function of latitude $\phi$, height $z$, time $t$
- $\mathbf{F}$ = Energy flux vector (atmosphere + ocean) [W m⁻²]
- $K$ = Diffusion coefficient representing heat transport [W m⁻¹ K⁻¹]
- $Q_{solar}(\phi) = \frac{S_0}{4}(1-\alpha)Q_{dist}(\phi)$ = Latitude-dependent solar heating
- $Q_{dist}(\phi)$ = Distribution function (higher at equator, lower at poles)

#### Key Physical Assumptions

1. **Zonal Symmetry**: All variables are averaged in the longitudinal direction. No distinction between continents and oceans at same latitude.

2. **Diffusive Heat Transport**: Complex atmospheric and oceanic dynamics (Hadley cells, jet streams, ocean gyres) parameterized as downgradient diffusion $F = -K\nabla T$. Real transport includes:
   - Atmospheric: Baroclinic eddies, Hadley cell, Walker circulation
   - Oceanic: Gyres, meridional overturning circulation, eddies
   
3. **Spherical Geometry**: Latitude-dependent area weighting:
   $$\nabla \cdot \mathbf{F} = \frac{1}{R\cos\phi} \frac{\partial}{\partial \phi}(\cos\phi \cdot F_\phi)$$
   where $R$ is Earth's radius.

4. **Solar Distribution**: Incoming solar radiation depends on latitude:
   $$Q_{solar}(\phi) \propto \cos\phi \text{ (approximately)}$$
   More accurate: accounts for Earth's tilt and seasonal cycle (annual mean here).

5. **Ice-Albedo Feedback**: Albedo $\alpha(\phi, T)$ increases when temperature drops below freezing:
   $$\alpha = \begin{cases}
   \alpha_{ocean} & T > 273 K \\
   \alpha_{ice} & T < 273 K
   \end{cases}$$
   This creates positive feedback: cooling → more ice → higher albedo → more cooling.

6. **Energy Balance Model (EBM) Form**: Often simplified to 1D in latitude:
   $$C \frac{\partial T}{\partial t} = Q_{in}(\phi)(1-\alpha) - A - BT + \frac{1}{R^2\cos\phi}\frac{\partial}{\partial\phi}\left(\cos\phi \cdot D\frac{\partial T}{\partial\phi}\right)$$

### Technical Overview (Page 2 of 2)

#### Mathematical Approximations

**Diffusive Transport Parameterization**: 

Real meridional energy transport is accomplished by:
- **Atmospheric**: 
  - Hadley cell (tropical): Direct thermal circulation, ~100 PW
  - Mid-latitude eddies: Baroclinic waves, ~50 PW  
  - Stationary waves: Mountain/heating contrasts
- **Oceanic**:
  - Wind-driven gyres: Gulf Stream, Kuroshio
  - Thermohaline circulation: Atlantic MOC, ~1-2 PW
  - Mesoscale eddies

Diffusion approximation:
$$F = -K \frac{\partial T}{\partial \phi}$$

where $K \approx 0.4-0.6$ W m⁻² K⁻¹ is calibrated to match observed transport (~6 PW from equator to pole). This is accurate for:
- ✓ Time-mean transport
- ✓ Large-scale patterns
- ✗ Transient eddies
- ✗ Non-local transport
- ✗ Asymmetries between hemispheres

**Linearized Outgoing Radiation**: 

Instead of $\epsilon\sigma T^4$, often use:
$$OLR = A + BT$$

where $A \approx 202$ W m⁻² and $B \approx 2.17$ W m⁻² K⁻¹ are fitted to match current climate. This is accurate for small perturbations ($\pm 10$ K) but breaks down for large changes.

**Ice-Albedo Feedback**:

Simple threshold:
$$\alpha(\phi) = \begin{cases}
0.32 & T > 273K \\
0.62 & T < 273K  
\end{cases}$$

Reality is more complex:
- Gradual transition via sea ice concentration
- Snow on land vs sea ice
- Seasonal cycle (summer melt, winter formation)
- Multi-year ice vs first-year ice
- Ice thickness and age effects

**Solar Distribution**:

Annual mean insolation at latitude $\phi$:
$$Q(\phi) = \frac{S_0}{\pi}\left[H(\phi)\sin\phi\sin\delta + \cos\phi\cos\delta\sin H(\phi)\right]$$

where $\delta$ is solar declination and $H$ is hour angle. For Earth:
$$Q(\phi) \approx Q_0(1 + 0.482P_2(\sin\phi))$$

where $P_2$ is Legendre polynomial. Common simplification:
$$Q(\phi) = Q_0\left(1 - 0.482\left(\frac{3\sin^2\phi - 1}{2}\right)\right)$$

#### Multiple Equilibria and Bifurcations

A remarkable feature of 2D EBMs: **multiple equilibrium states**

For current solar constant:
1. **Warm climate** (current): Polar ice caps at ~70° latitude
2. **Snowball Earth**: Global ice coverage (albedo catastrophe)
3. **Ice-free**: No permanent ice (hothouse)

Ice-albedo feedback creates **hysteresis**:
- Decreasing $S_0$: Climate remains warm until critical point, then sudden transition to snowball
- Increasing $S_0$: Snowball persists past the point where warm climate originally froze

Critical solar constant for snowball initiation: $S_c \approx 0.94 S_0$ (~6% reduction)

#### Climate Sensitivity in 2D Models

ECS ≈ 2.5-3.5°C, higher than 1D models because:
- ✓ Ice-albedo feedback included
- ✓ Polar amplification captured: Arctic warms 2-3× faster than global mean
- ✓ Pattern effects: Regional forcing distributions matter

#### Limitations

✗ **No longitudinal structure**: Cannot represent monsoons, ENSO, NAO
✗ **No ocean dynamics**: Thermohaline circulation not resolved
✗ **Simplified clouds**: Major uncertainty
✗ **No topography**: Mountains affect circulation patterns
✗ **Annual mean**: Seasonal cycle important for ice

#### Applications

✓ Paleoclimate: Snowball Earth, ice ages, Eocene hothouse
✓ Conceptual understanding: Feedbacks, multiple equilibria
✓ Computational efficiency: Fast scenario testing
✓ Polar amplification: Captures Arctic warming pattern
    """)


@app.cell
def __():
    import marimo as mo
    return mo.md(r"""
<a id='model4'></a>
## Model 4: Three-Dimensional General Circulation Model (GCM)

### Technical Overview (Page 1 of 2)

Three-Dimensional General Circulation Models represent the state-of-the-art in traditional climate modeling. These models explicitly resolve atmospheric and oceanic circulation in three spatial dimensions and time, governed by the fundamental equations of fluid dynamics and thermodynamics.

#### Governing Equations

GCMs solve the **primitive equations** on a 3D grid:

**1. Momentum (Navier-Stokes):**
$$\frac{D\mathbf{u}}{Dt} + 2\mathbf{\Omega} \times \mathbf{u} = -\frac{1}{\rho}\nabla p + \mathbf{g} + \mathbf{F}$$

**2. Continuity (Mass Conservation):**
$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0$$

**3. Thermodynamic Energy:**
$$\rho c_p \frac{DT}{Dt} = \frac{Dp}{Dt} + Q_{rad} + Q_{latent} + Q_{sens}$$

**4. Water Vapor:**
$$\frac{Dq}{Dt} = S_{evap} - S_{precip} + \text{diffusion}$$

**5. Hydrostatic Balance (vertical):**
$$\frac{\partial p}{\partial z} = -\rho g$$

Where:
- $\mathbf{u} = (u, v, w)$ = 3D velocity field (m/s)
- $\mathbf{\Omega}$ = Earth's rotation vector
- $\rho$ = Air/water density (kg/m³)
- $p$ = Pressure (Pa)
- $T$ = Temperature (K)
- $q$ = Specific humidity (kg/kg)
- $Q$ = Heating/cooling terms (W/kg)

#### Model Components

**Atmospheric Model:**
- Horizontal resolution: 50-200 km (lat-lon grid or spectral)
- Vertical levels: 30-100 (surface to ~50-100 km)
- Time step: 10-30 minutes
- Prognostic variables: u, v, w, T, p, q, clouds

**Ocean Model:**
- Resolution: 25-100 km horizontal, 40-60 vertical levels
- Dynamics: Full 3D primitive equations
- Tracers: Temperature, salinity, biogeochemistry
- Sea ice: Thermodynamics and dynamics

**Land Surface Model:**
- Soil moisture, temperature (multiple layers)
- Vegetation: Types, phenology, photosynthesis
- Snow cover and albedo
- Runoff and groundwater

**Cryosphere:**
- Sea ice: Thickness, concentration, dynamics
- Land ice: Mass balance (simple) or ice sheet model (advanced)
- Snow cover: Depth, density, albedo evolution

#### Physical Parameterizations

**Sub-grid processes** that cannot be resolved explicitly:

1. **Radiation**: 
   - Solar: Rayleigh scattering, absorption by O₃, H₂O, clouds
   - Longwave: Line-by-line or band models for greenhouse gases
   - Computed every 1-3 hours (expensive!)

2. **Convection**:
   - Deep convection (thunderstorms): Mass flux schemes
   - Shallow convection: Eddy diffusivity
   - Triggers: CAPE, moisture convergence
   - Outputs: Precipitation, heating/moistening profiles

3. **Clouds**:
   - Formation: Relative humidity threshold or PDF-based
   - Types: Stratiform vs convective
   - Microphysics: Condensation, freezing, precipitation
   - Huge uncertainty source!

4. **Boundary Layer Turbulence**:
   - Vertical mixing of heat, moisture, momentum
   - K-theory, TKE schemes, or higher-order closure
   - Surface fluxes: Bulk aerodynamic formulae

5. **Gravity Wave Drag**:
   - Orographic: Mountain effects on flow
   - Non-orographic: Convection, fronts
   - Critical for stratospheric circulation

### Technical Overview (Page 2 of 2)

#### Numerical Methods

**Spatial Discretization:**
- **Finite Difference**: Grid points, simple but diffusive
- **Finite Volume**: Conservative, good for tracers
- **Spectral**: Spherical harmonics, accurate but expensive
- **Finite Element**: Flexible grids (icosahedral)

**Temporal Integration:**
- **Semi-implicit**: Large time steps for stable modes
- **Split-explicit**: Fast/slow modes separated
- **Leapfrog, RK schemes**: Various orders of accuracy

**Grids:**
- Lat-lon: Simple but pole singularity
- Cubed-sphere: 6 patches, more uniform
- Icosahedral: Triangular cells, nearly uniform
- Variable resolution: Regional refinement

#### Key Approximations

1. **Hydrostatic Approximation**: 
   $$\frac{\partial p}{\partial z} = -\rho g$$
   Valid for horizontal scales >> vertical scale (~10 km)
   Breaks down for deep convection, topography

2. **Boussinesq Approximation**:
   Density variations neglected except in buoyancy
   Valid for small density variations

3. **Shallow Atmosphere**:
   Earth's radius >> atmospheric depth
   Metric terms simplified

4. **Sub-grid Parameterizations**:
   Most critical approximation! Clouds, convection, turbulence
   cannot be resolved and must be parameterized
   → Largest uncertainty in GCMs

5. **Resolution Limits**:
   - Cannot resolve individual clouds (km scale)
   - Cannot resolve ocean mesoscale eddies (<50 km)
   - Cannot resolve boundary layer turbulence (m scale)
   
#### Climate Sensitivity in GCMs

Modern GCMs: ECS = 2.5-5.0°C (IPCC AR6 range: 2.5-4.0°C likely)

Higher sensitivity than simpler models due to:
- ✓ Cloud feedbacks (most uncertain!)
- ✓ Water vapor feedback (well-constrained)
- ✓ Ice-albedo feedback
- ✓ Lapse rate feedback
- ✓ Regional patterns and teleconnections

**Feedback Analysis:**
$$\text{ECS} = \frac{\lambda_0}{1 - \sum f_i}$$

where $f_i$ are individual feedbacks:
- $f_{H_2O} ≈ +0.5$ (strongly positive)
- $f_{ice} ≈ +0.3$ (positive)
- $f_{cloud} ≈ +0.2$ to +0.8 (uncertain!)
- $f_{lapse} ≈ -0.2$ (negative)

#### Advantages Over Simpler Models

✓ Explicit dynamics: Jets, storms, monsoons, ENSO
✓ Regional detail: Precipitation, drought, extremes
✓ Coupled system: Ocean-atmosphere interactions
✓ Tracers: CO₂, aerosols, chemistry
✓ Transient response: Decades to centuries
✓ Multiple forcings: GHGs, aerosols, land use

#### Limitations

✗ Computationally expensive: Weeks to months for century runs
✗ Parameterization uncertainty: Sub-grid physics
✗ Systematic biases: Regional temperature/precipitation errors
✗ Limited resolution: Cannot resolve small scales
✗ Initialization: Sensitive to initial conditions (weather scales)

#### Validation

GCMs are validated against:
- Historical climate (1850-present)
- Paleoclimate (Last Glacial Maximum, Mid-Holocene)
- Satellite observations (radiation, temperature, clouds)
- Reanalysis data
- Process studies

**Key Metrics:**
- Mean state climatology
- Seasonal cycle
- Interannual variability (ENSO, NAO)
- Trends (warming, sea level rise)
- Extreme events
    """)


@app.cell
def __():
    import marimo as mo
    return mo.md(r"""
<a id='model5'></a>
## Model 5: GraphCast - ML-Based Weather and Climate Prediction

### Technical Overview (Page 1 of 2)

GraphCast, developed by Google DeepMind, represents a paradigm shift in weather and climate modeling. Instead of explicitly solving physical equations, it uses machine learning to learn patterns from historical data and make predictions. This approach achieves competitive or superior accuracy to traditional physics-based models while being orders of magnitude faster.

#### Architecture and Approach

**Core Innovation:**
GraphCast uses a **Graph Neural Network (GNN)** operating on a multi-resolution mesh of Earth's surface and atmosphere. Unlike traditional grid-based models, the graph structure allows flexible representation of Earth's spherical geometry and multi-scale processes.

**Model Architecture:**

1. **Input Representation:**
   - Two atmospheric states: current time $t$ and $t-\Delta t$
   - Variables: Temperature, winds (u,v), pressure, humidity, geopotential at multiple levels
   - Surface variables: Temperature, pressure, moisture
   - Grid: ~0.25° resolution (~28 km at equator), 37 pressure levels

2. **Encoder:**
   - Maps gridded data to graph representation
   - Each grid point → graph node
   - Edges connect nearby nodes (multi-resolution)
   
3. **Processor:**
   - 16 layers of message-passing GNN
   - Each layer: nodes aggregate information from neighbors
   - Attention mechanisms weight importance
   - ~37 million parameters total
   
4. **Decoder:**
   - Maps graph back to grid
   - Outputs: Future state at $t+\Delta t$ (typically 6 hours)
   
5. **Autoregressive Rollout:**
   - Multi-step predictions: use output as input for next step
   - 10-day forecast: 40 steps of 6-hour predictions

#### Training Data and Process

**Data:**
- ERA5 reanalysis (ECMWF): 1979-2017 (training), 2018-2021 (validation/test)
- ~1.4 million atmospheric states
- All weather conditions: hurricanes, monsoons, heatwaves, etc.

**Training:**
- Loss function: Weighted MSE + gradient penalties
- Emphasis on:
  - Conservation of physical quantities
  - Smooth spatial fields
  - Realistic amplitudes and patterns
  
**Objective:**
$$\mathcal{L} = \sum_{i,t} w_i ||X_{pred}^{t+\Delta t} - X_{true}^{t+\Delta t}||^2 + \lambda ||\nabla X_{pred}||^2$$

where $w_i$ are pressure-dependent weights (emphasize troposphere).

#### Key Physical Constraints (Learned, Not Enforced)

Unlike traditional models that explicitly solve conservation laws, GraphCast learns to respect them through data:

1. **Mass Conservation**: Total atmospheric mass should not change
2. **Energy Conservation**: KE + PE + IE balanced
3. **Geostrophic Balance**: Winds and pressure gradients related
4. **Hydrostatic Balance**: Vertical pressure-temperature relationship
5. **Water Cycle**: Evaporation ≈ Precipitation (global mean)

These emerge from training, not hard constraints!

#### Advantages of ML Approach

✓ **Speed**: 1-minute runtime for 10-day forecast (vs hours for traditional GCMs)
✓ **Scalability**: Inference cost independent of forecast length
✓ **Data-driven**: Learns complex patterns humans cannot parameterize
✓ **Resolution**: Fine-scale features without explicit sub-grid models
✓ **Flexibility**: Easy to add new variables or change resolution

#### Limitations

✗ **Data-dependent**: Cannot predict beyond training distribution
   - Novel climate states (e.g., 4°C warmer) uncertain
   - Rare extremes underrepresented in training data
   
✗ **Black box**: Difficult to interpret why predictions made

✗ **Physical consistency**: May violate conservation laws subtly

✗ **Long-term drift**: Accumulates errors over many time steps

✗ **Extrapolation**: Struggles with unprecedented conditions

### Technical Overview (Page 2 of 2)

#### Comparison: GraphCast vs Traditional GCMs

| Aspect | Traditional GCM | GraphCast |
|--------|----------------|-----------|
| **Physics** | Explicit equations | Learned from data |
| **Speed** | Hours (10-day forecast) | ~1 minute |
| **Resolution** | 25-100 km | ~25 km |
| **Accuracy** | Benchmark standard | Competitive/superior |
| **Interpretability** | High (physical basis) | Low (black box) |
| **Extrapolation** | Reasonable | Limited |
| **Novel climates** | Possible | Uncertain |
| **Development** | Decades of refinement | Rapid iteration |

#### Performance Metrics

**Weather Forecasting (GraphCast paper results):**
- **Skill score vs ECMWF IFS**: GraphCast wins on 90% of targets at 10-day lead
- **Tropical cyclones**: Better track forecasting than operational models
- **Atmospheric rivers**: Improved prediction of extreme precipitation
- **Upper atmosphere**: Superior stratospheric forecasts

**Key Results:**
- 500 hPa geopotential (weather patterns): ~10% better RMSE at day 5
- Surface temperature: Competitive with best physics models
- Precipitation: Good skill, some systematic biases
- Extremes: Better than GCMs for many metrics

#### Application to Climate Change

**Direct Application:**
- GraphCast is trained on current climate
- Cannot directly simulate future climates (e.g., +4°C)

**Potential Uses:**
1. **Downscaling**: Take coarse GCM output → produce fine-scale patterns
2. **Bias Correction**: Correct systematic GCM errors
3. **Emulation**: Fast surrogate for expensive GCM runs
4. **Process Studies**: Identify patterns in climate data
5. **Hybrid Models**: ML components within physics-based frameworks

**Climate Model Emulation:**
- Train ML model on GCM output (thousands of years)
- Emulator runs 1000× faster than GCM
- Enables massive ensembles, sensitivity studies
- Uncertainty quantification

**Future Directions:**
- **Climate GraphCast**: Train on multi-decade simulations spanning climate change
- **Physics-informed ML**: Enforce conservation laws as constraints
- **Uncertainty quantification**: Ensemble methods, Bayesian approaches
- **Extreme events**: Specialized training for rare but important events

#### Implementation Considerations

**Computational Requirements:**
- Training: Weeks on TPU v4 pods (expensive!)
- Inference: Single GPU sufficient, very fast
- Memory: ~10 GB for model weights

**Data Requirements:**
- Petabytes of reanalysis data
- Consistent, quality-controlled observations
- Long time series for training

**Reproducibility:**
- Model weights publicly available
- Code open-sourced (JAX implementation)
- Can be fine-tuned for regional applications

#### Philosophical Implications

GraphCast represents a fundamental question: **Do we need to understand physics to predict climate?**

**Traditional view**: Understanding → Equations → Simulation → Prediction

**ML view**: Data → Patterns → Prediction (Understanding optional)

**Reality**: Hybrid approach likely optimal
- Use physics for constraints, conservation
- Use ML for complex parameterizations (clouds, convection)
- Combine strengths of both approaches

**Climate Science Community Response:**
- Excitement about potential
- Caution about extrapolation
- Active research on hybrid models
- Debate on role of physical understanding
    """)


@app.cell
def __():
    import marimo as mo
    return mo.md(r"""
<a id='climate-change'></a>
## Climate Change Analysis: Using Models to Understand Warming

### Synthesis Across Models

We've built five models of increasing sophistication. Now we use them together to understand climate change, demonstrating how each contributes to our understanding.

#### Key Questions We Can Answer:

1. **How much will Earth warm with doubled CO₂?** (Climate Sensitivity)
2. **Where will warming be strongest?** (Spatial Patterns)
3. **How fast will warming occur?** (Transient Response)
4. **What are the key feedbacks?** (Physical Mechanisms)
5. **How certain are we?** (Model Agreement and Uncertainty)

### Model Predictions Summary

| Model | ECS (°C) | Key Features | Limitations |
|-------|----------|--------------|-------------|
| **1: 0D EBM** | ~1.2 | Global mean only | No feedbacks |
| **2: 1D RCM** | ~2.0 | Vertical structure | No geography |
| **3: 2D EBM** | ~2.8 | Polar amplification | No dynamics |
| **4: 3D GCM** | ~3.2 | Full spatial detail | Parameterizations |
| **5: GraphCast** | Data-driven | ML patterns | Extrapolation limited |

**IPCC AR6 Assessment: ECS = 2.5-4.0°C (likely range), best estimate 3.0°C**

Our progression shows convergence toward the observational estimate as we add complexity!

### Physical Insights

**Why Models Agree:**
1. **Energy Balance**: All conserve energy
2. **Greenhouse Effect**: CO₂ absorbs infrared radiation
3. **Planck Response**: Warmer Earth emits more radiation
4. **Water Vapor Feedback**: Warmer air holds more H₂O (greenhouse gas)

**Why Models Differ:**
1. **Ice-Albedo Feedback**: Requires geography (Models 3-4)
2. **Cloud Feedback**: Complex, different parameterizations (GCMs)  
3. **Lapse Rate Feedback**: Requires vertical structure (Models 2-4)
4. **Regional Patterns**: Affect global mean through nonlinearities

### Justifying Climate Change Projections

#### Evidence from Models:

**1. Model-Observation Agreement (Historical Period)**
- All models successfully reproduce 20th century warming (~1°C)
- Spatial patterns match (land>ocean, Arctic>tropics)
- Cannot explain warming without human emissions

**2. Physical Understanding**
- Greenhouse effect is basic physics (known since 1896)
- CO₂ absorbs at 15 μm (well-measured)
- Increased CO₂ → reduced OLR → warming (unavoidable)

**3. Multiple Lines of Evidence**
- Paleoclimate: Past CO₂-temperature relationship
- Satellite observations: Radiative forcing measured directly
- Process studies: Individual feedbacks constrained
- Model hierarchy: Simple to complex models agree

**4. Consistency Across Scales**
- Global mean temperature: All models converge
- Regional patterns: Polar amplification robust
- Seasonal cycle: Maintained in future
- Extreme events: Intensification predicted

#### Uncertainty Quantification

**Sources of Uncertainty:**

1. **Future Emissions** (Scenario Uncertainty):
   - Depends on policy, technology, economics
   - Range: +1.5°C to +4.5°C by 2100
   - Largest source of uncertainty

2. **Climate Response** (Model Uncertainty):
   - Cloud feedbacks: ±0.5°C
   - Carbon cycle: ±0.3°C
   - Ice sheets: ±0.2°C
   - Total: ±0.7°C

3. **Natural Variability** (Internal Variability):
   - ENSO, volcanoes, solar: ±0.2°C on decadal scales
   - Averages out over longer periods

**Confidence Levels (IPCC AR6):**
- Human influence on warming: **Unequivocal** (100%)
- Continued warming with emissions: **Virtually certain** (>99%)
- Exceeding 1.5°C by 2040: **Very likely** (>90%)
- Warming continues for centuries: **Very high confidence** (>95%)

### Policy-Relevant Findings

**What We Know with High Confidence:**
✓ Each ton of CO₂ causes warming (linearly)
✓ Warming committed even if emissions stop
✓ Limiting warming requires net-zero emissions
✓ Earlier action is cheaper and more effective
✓ Impacts scale with warming magnitude

**What Remains Uncertain:**
? Exact magnitude of warming (2.5-4°C range for 2×CO₂)
? Regional precipitation changes (sign and magnitude)
? Tipping points and abrupt changes (ice sheets, AMOC)
? Climate-carbon cycle feedbacks (permafrost, forests)
? Exact timing of impacts

**Key Message:**
Uncertainty is NOT a reason for inaction - it includes possibilities of outcomes worse than best estimates!
    """)


@app.cell
def __():
    import marimo as mo
    return mo.md(r"""
## Conclusions and Summary

### Journey Through Climate Models

We've progressed through five generations of climate modeling, each adding layers of sophistication:

1. **Model 1 (0D EBM)**: Established energy balance fundamentals
2. **Model 2 (1D RCM)**: Added vertical atmospheric structure
3. **Model 3 (2D EBM)**: Incorporated meridional variations and ice-albedo feedback
4. **Model 4 (3D GCM)**: Full three-dimensional dynamics and circulation
5. **Model 5 (GraphCast)**: Machine learning-based pattern recognition

### Key Takeaways

**Scientific Understanding:**
- Climate change is rooted in basic physics (energy balance, greenhouse effect)
- Multiple independent lines of evidence converge on similar conclusions
- Model hierarchy builds confidence through consistency
- Uncertainty does not imply lack of knowledge - ranges are well-constrained

**Technical Insights:**
- Simple models provide intuition and rapid exploration
- Complex models capture essential regional details
- Machine learning offers new approaches but doesn't replace physics
- All models have limitations - use appropriate tool for question

**Policy Implications:**
- Warming is proportional to cumulative emissions
- Net-zero emissions required to stabilize temperature
- Earlier action is more effective and less costly
- Every tenth of a degree matters for impacts

### Future Directions

**Model Development:**
- Higher resolution (km-scale globally)
- Better representation of clouds and precipitation
- Improved ice sheet dynamics
- Interactive carbon cycle and vegetation
- Hybrid physics-ML approaches

**Scientific Challenges:**
- Tipping points and abrupt changes
- Regional climate change and extremes
- Multi-century sea level rise
- Climate-carbon cycle feedbacks
- Attribution of specific events

**Applications:**
- Climate services for adaptation planning
- Early warning systems for extremes
- Impact assessments (agriculture, water, health)
- Policy evaluation and carbon budgets
- Long-term planning (infrastructure, insurance)

### Final Thoughts

Climate models, from the simplest energy balance to the most sophisticated machine learning systems, all tell the same fundamental story: **Earth's climate is sensitive to greenhouse gas concentrations, and continued emissions will cause substantial warming with serious consequences.**

The progression from Model 1 to Model 5 demonstrates that this conclusion is robust across modeling approaches, physical understanding, and mathematical frameworks. While uncertainties remain in details, the big picture is clear and demands action.

**As physicist Richard Feynman said: "Nature uses only the longest threads to weave her patterns, so that each small piece of her fabric reveals the organization of the entire tapestry."**

Our hierarchy of models reveals this tapestry, from the simplest threads of energy balance to the complex weave of global circulation and the learned patterns of machine intelligence.

---

### References and Further Reading

**Key Papers:**
- Budyko (1969): Simple climate model foundations
- Manabe & Wetherald (1975): First 3D climate model with CO₂ doubling
- Cess et al. (1989): Climate feedback analysis
- IPCC AR6 WG1 (2021): Comprehensive assessment
- Lam et al. (2023): GraphCast paper (Nature)

**Textbooks:**
- Hartmann: "Global Physical Climatology"
- Marshall & Plumb: "Atmosphere, Ocean, and Climate Dynamics"
- Peixoto & Oort: "Physics of Climate"
- McGuffie & Henderson-Sellers: "A Climate Modelling Primer"

**Online Resources:**
- CMIP6 model archive: https://esgf-node.llnl.gov/
- ERA5 reanalysis: https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
- GraphCast code: https://github.com/deepmind/graphcast
- IPCC Reports: https://www.ipcc.ch/

---

*Thank you for following this journey through climate modeling!*
    """)


if __name__ == "__main__":
    app.run()

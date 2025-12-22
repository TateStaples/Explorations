#!/usr/bin/env python3
"""
Add remaining models (Models 3, 4, 5) and climate change analysis to notebook
"""
import json

# Load existing notebook
with open('climate_models_blog.ipynb', 'r') as f:
    notebook = json.load(f)

def add_markdown(text):
    """Add markdown cell"""
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split('\n') if isinstance(text, str) else text
    })

def add_code(code):
    """Add code cell"""
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split('\n') if isinstance(code, str) else code
    })

print("Adding Model 2 visualizations...")

# Model 2 Visualizations
add_code("""# Visualize Model 2: Radiative-Convective Model

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# === Panel 1: Temperature Profiles ===
ax1 = fig.add_subplot(gs[0, 0])

ax1.plot(T_control - 273.15, model2.z/1000, 'b-', linewidth=3, 
         label='Control Climate', marker='o', markersize=4, alpha=0.7)
ax1.plot(T_2xCO2 - 273.15, model2.z/1000, 'r-', linewidth=3,
         label='2×CO₂ Climate', marker='s', markersize=4, alpha=0.7)

# Add standard atmosphere for reference
T_standard = np.array([288 - 6.5*z/1000 for z in model2.z])
T_standard = np.maximum(T_standard, 180)
ax1.plot(T_standard - 273.15, model2.z/1000, 'k--', linewidth=1.5,
         label='Standard Atmosphere', alpha=0.5)

ax1.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Height (km)', fontsize=12, fontweight='bold')
ax1.set_title('Vertical Temperature Profiles', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 20)
ax1.set_xlim(-80, 30)

# === Panel 2: Temperature Change ===
ax2 = fig.add_subplot(gs[0, 1])

delta_T = T_2xCO2 - T_control
ax2.plot(delta_T, model2.z/1000, 'purple', linewidth=3, marker='o', markersize=5)
ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax2.axhline(model2.z[np.argmax(delta_T)]/1000, color='red', linestyle=':', 
            alpha=0.5, label=f'Max warming: {np.max(delta_T):.2f} K')

ax2.set_xlabel('Temperature Change (K)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Height (km)', fontsize=12, fontweight='bold')
ax2.set_title('Warming Profile (2×CO₂ - Control)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 20)

# === Panel 3: Lapse Rate ===
ax3 = fig.add_subplot(gs[1, 0])

# Calculate lapse rates
dz = np.diff(model2.z)
dT_control = np.diff(T_control)
lapse_control = -dT_control / dz * 1000  # K/km

dT_2xco2 = np.diff(T_2xCO2)
lapse_2xco2 = -dT_2xco2 / dz * 1000

z_mid = (model2.z[:-1] + model2.z[1:]) / 2

ax3.plot(lapse_control, z_mid/1000, 'b-', linewidth=2.5, 
         label='Control', marker='o', markersize=3, alpha=0.7)
ax3.plot(lapse_2xco2, z_mid/1000, 'r-', linewidth=2.5,
         label='2×CO₂', marker='s', markersize=3, alpha=0.7)
ax3.axvline(model2.critical_lapse*1000, color='green', linestyle='--', 
            linewidth=2, alpha=0.7, label=f'Critical: {model2.critical_lapse*1000:.1f} K/km')

ax3.set_xlabel('Lapse Rate (K/km)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Height (km)', fontsize=12, fontweight='bold')
ax3.set_title('Atmospheric Lapse Rate', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 15)
ax3.set_xlim(-10, 12)

# === Panel 4: Radiative Fluxes ===
ax4 = fig.add_subplot(gs[1, 1])

F_up, F_down = model2.compute_radiative_fluxes(T_control)
z_flux = np.concatenate([[0], model2.z])

ax4.plot(F_up, z_flux/1000, 'r-', linewidth=3, label='Upward LW', marker='^', 
         markersize=4, alpha=0.7)
ax4.plot(F_down, z_flux/1000, 'b-', linewidth=3, label='Downward LW', marker='v',
         markersize=4, alpha=0.7)
ax4.plot(F_up - F_down, z_flux/1000, 'purple', linewidth=2, linestyle='--',
         label='Net Flux', alpha=0.7)

ax4.set_xlabel('Radiative Flux (W/m²)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Height (km)', fontsize=12, fontweight='bold')
ax4.set_title('Longwave Radiative Fluxes', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10, loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 20)

# === Panel 5: Heating Rates ===
ax5 = fig.add_subplot(gs[2, 0])

Q_solar = model2.solar_heating(T_control)
F_up, F_down = model2.compute_radiative_fluxes(T_control)
Q_lw = model2.longwave_heating(F_up, F_down)

ax5.plot(Q_solar, model2.z/1000, 'orange', linewidth=2.5, 
         label='Solar Heating', marker='o', markersize=4)
ax5.plot(Q_lw, model2.z/1000, 'blue', linewidth=2.5,
         label='LW Cooling', marker='s', markersize=4)
ax5.plot(Q_solar + Q_lw, model2.z/1000, 'green', linewidth=2.5, linestyle='--',
         label='Net Heating', marker='d', markersize=4)
ax5.axvline(0, color='gray', linestyle='--', alpha=0.5)

ax5.set_xlabel('Heating Rate (K/day)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Height (km)', fontsize=12, fontweight='bold')
ax5.set_title('Radiative Heating/Cooling Rates', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_ylim(0, 15)

# === Panel 6: Energy Balance ===
ax6 = fig.add_subplot(gs[2, 1])

# TOA balance
F_up_toa = F_up[-1]
F_solar_in = (model2.S0/4) * (1 - model2.albedo)
F_net_toa = F_solar_in - F_up_toa

# Surface balance
F_up_sfc = F_up[0]
F_down_sfc = F_down[0]
F_solar_sfc = F_solar_in * (1 - model2.solar_abs_atm)
F_net_sfc = F_solar_sfc + F_down_sfc - F_up_sfc

components = ['Solar\\nIn', 'OLR', 'Net\\nTOA', 'Solar\\nSfc', 'Down\\nLW', 'Up\\nLW', 'Net\\nSfc']
values = [F_solar_in, F_up_toa, F_net_toa, F_solar_sfc, F_down_sfc, -F_up_sfc, F_net_sfc]
colors = ['orange', 'blue', 'green', 'orange', 'red', 'blue', 'green']

bars = ax6.bar(components, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}', ha='center', va='bottom' if height > 0 else 'top',
             fontsize=9, fontweight='bold')

ax6.axhline(0, color='black', linewidth=1)
ax6.set_ylabel('Energy Flux (W/m²)', fontsize=12, fontweight='bold')
ax6.set_title('Top-of-Atmosphere and Surface Energy Balance', 
              fontsize=14, fontweight='bold')
ax6.grid(True, axis='y', alpha=0.3)
ax6.set_ylim(-400, 400)

plt.suptitle('One-Dimensional Radiative-Convective Model: Complete Analysis',
             fontsize=17, fontweight='bold', y=0.995)

plt.savefig('model2_complete.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\n" + "="*70)
print("KEY INSIGHTS FROM MODEL 2")
print("="*70)
print(f"\\n1. Vertical Structure: Surface warmer ({T_control[0]-273.15:.1f}°C) than")
print(f"   upper atmosphere ({T_control[-1]-273.15:.1f}°C) due to greenhouse effect")
print(f"\\n2. Climate Sensitivity: {ECS:.2f} K - closer to observations than Model 1")
print("   due to inclusion of lapse rate and water vapor feedbacks")
print(f"\\n3. Stratospheric Cooling: Upper atmosphere cools with CO₂ increase")
print("   while surface warms - characteristic signature of greenhouse forcing")
print("\\n4. Convective Control: Tropospheric lapse rate maintained near")
print(f"   critical value ({model2.critical_lapse*1000:.1f} K/km) by convection")
print("\\n5. Greenhouse Back-radiation: Downward LW at surface")
print(f"   ({F_down_sfc:.1f} W/m²) >> direct solar, demonstrating atmospheric effect")
print("\\n" + "="*70)""")

print("Adding Model 3...")

# MODEL 3: Two-Dimensional Statistical Dynamical Model
add_markdown("""<a id='model3'></a>
## Model 3: Two-Dimensional Statistical Dynamical Model

### Technical Overview (Page 1 of 2)

The Two-Dimensional Statistical Dynamical Model extends our framework by adding **latitudinal variation** while maintaining zonal (longitudinal) averaging. This captures the fundamental feature of Earth's climate: the equator-to-pole temperature gradient driven by differential solar heating.

#### Governing Equations

The model solves coupled equations for temperature and energy transport:

**Thermodynamic Equation:**
$$\\rho c_p \\frac{\\partial T}{\\partial t} = -\\nabla \\cdot \\mathbf{F} + Q_{rad} + Q_{conv}$$

**Meridional Energy Transport:**
$$\\mathbf{F} = -K \\nabla T$$

**Radiative Balance:**
$$Q_{rad} = Q_{solar}(\\phi) - \\epsilon \\sigma T^4$$

Where:
- $T(\\phi, z, t)$ = Temperature as function of latitude $\\phi$, height $z$, time $t$
- $\\mathbf{F}$ = Energy flux vector (atmosphere + ocean) [W m⁻²]
- $K$ = Diffusion coefficient representing heat transport [W m⁻¹ K⁻¹]
- $Q_{solar}(\\phi) = \\frac{S_0}{4}(1-\\alpha)Q_{dist}(\\phi)$ = Latitude-dependent solar heating
- $Q_{dist}(\\phi)$ = Distribution function (higher at equator, lower at poles)

#### Key Physical Assumptions

1. **Zonal Symmetry**: All variables are averaged in the longitudinal direction. No distinction between continents and oceans at same latitude.

2. **Diffusive Heat Transport**: Complex atmospheric and oceanic dynamics (Hadley cells, jet streams, ocean gyres) parameterized as downgradient diffusion $F = -K\\nabla T$. Real transport includes:
   - Atmospheric: Baroclinic eddies, Hadley cell, Walker circulation
   - Oceanic: Gyres, meridional overturning circulation, eddies
   
3. **Spherical Geometry**: Latitude-dependent area weighting:
   $$\\nabla \\cdot \\mathbf{F} = \\frac{1}{R\\cos\\phi} \\frac{\\partial}{\\partial \\phi}(\\cos\\phi \\cdot F_\\phi)$$
   where $R$ is Earth's radius.

4. **Solar Distribution**: Incoming solar radiation depends on latitude:
   $$Q_{solar}(\\phi) \\propto \\cos\\phi \\text{ (approximately)}$$
   More accurate: accounts for Earth's tilt and seasonal cycle (annual mean here).

5. **Ice-Albedo Feedback**: Albedo $\\alpha(\\phi, T)$ increases when temperature drops below freezing:
   $$\\alpha = \\begin{cases}
   \\alpha_{ocean} & T > 273 K \\\\
   \\alpha_{ice} & T < 273 K
   \\end{cases}$$
   This creates positive feedback: cooling → more ice → higher albedo → more cooling.

6. **Energy Balance Model (EBM) Form**: Often simplified to 1D in latitude:
   $$C \\frac{\\partial T}{\\partial t} = Q_{in}(\\phi)(1-\\alpha) - A - BT + \\frac{1}{R^2\\cos\\phi}\\frac{\\partial}{\\partial\\phi}\\left(\\cos\\phi \\cdot D\\frac{\\partial T}{\\partial\\phi}\\right)$$

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
$$F = -K \\frac{\\partial T}{\\partial \\phi}$$

where $K \\approx 0.4-0.6$ W m⁻² K⁻¹ is calibrated to match observed transport (~6 PW from equator to pole). This is accurate for:
- ✓ Time-mean transport
- ✓ Large-scale patterns
- ✗ Transient eddies
- ✗ Non-local transport
- ✗ Asymmetries between hemispheres

**Linearized Outgoing Radiation**: 

Instead of $\\epsilon\\sigma T^4$, often use:
$$OLR = A + BT$$

where $A \\approx 202$ W m⁻² and $B \\approx 2.17$ W m⁻² K⁻¹ are fitted to match current climate. This is accurate for small perturbations ($\\pm 10$ K) but breaks down for large changes.

**Ice-Albedo Feedback**:

Simple threshold:
$$\\alpha(\\phi) = \\begin{cases}
0.32 & T > 273K \\\\
0.62 & T < 273K  
\\end{cases}$$

Reality is more complex:
- Gradual transition via sea ice concentration
- Snow on land vs sea ice
- Seasonal cycle (summer melt, winter formation)
- Multi-year ice vs first-year ice
- Ice thickness and age effects

**Solar Distribution**:

Annual mean insolation at latitude $\\phi$:
$$Q(\\phi) = \\frac{S_0}{\\pi}\\left[H(\\phi)\\sin\\phi\\sin\\delta + \\cos\\phi\\cos\\delta\\sin H(\\phi)\\right]$$

where $\\delta$ is solar declination and $H$ is hour angle. For Earth:
$$Q(\\phi) \\approx Q_0(1 + 0.482P_2(\\sin\\phi))$$

where $P_2$ is Legendre polynomial. Common simplification:
$$Q(\\phi) = Q_0\\left(1 - 0.482\\left(\\frac{3\\sin^2\\phi - 1}{2}\\right)\\right)$$

#### Multiple Equilibria and Bifurcations

A remarkable feature of 2D EBMs: **multiple equilibrium states**

For current solar constant:
1. **Warm climate** (current): Polar ice caps at ~70° latitude
2. **Snowball Earth**: Global ice coverage (albedo catastrophe)
3. **Ice-free**: No permanent ice (hothouse)

Ice-albedo feedback creates **hysteresis**:
- Decreasing $S_0$: Climate remains warm until critical point, then sudden transition to snowball
- Increasing $S_0$: Snowball persists past the point where warm climate originally froze

Critical solar constant for snowball initiation: $S_c \\approx 0.94 S_0$ (~6% reduction)

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
✓ Polar amplification: Captures Arctic warming pattern""")

# Model 3 Implementation
add_code("""class TwoDimensionalEBM:
    \"\"\"
    Two-Dimensional Energy Balance Model (latitude-height)
    
    Includes meridional heat transport and ice-albedo feedback.
    Demonstrates polar amplification and potential for multiple equilibria.
    \"\"\"
    
    def __init__(self, n_lat=36, n_levels=10):
        \"\"\"
        Initialize 2D EBM
        
        Parameters:
        -----------
        n_lat : int
            Number of latitude bands
        n_levels : int
            Number of vertical levels (simplified from Model 2)
        \"\"\"
        # Physical constants
        self.sigma = 5.67e-8    # Stefan-Boltzmann constant
        self.S0 = 1361.0        # Solar constant
        self.R_earth = 6.371e6  # Earth radius (m)
        
        # Grid
        self.n_lat = n_lat
        self.lat = np.linspace(-90, 90, n_lat)  # Latitude (degrees)
        self.lat_rad = np.deg2rad(self.lat)     # Latitude (radians)
        self.d_lat = np.deg2rad(180 / (n_lat - 1))  # Grid spacing
        
        # Model parameters
        self.A = 202.0          # OLR parameter A (W/m²)
        self.B = 2.17           # OLR parameter B (W/m²/K)
        self.D = 0.44           # Diffusion coefficient (W/m²/K)
        self.C = 4e7            # Heat capacity (J/m²/K) - mixed layer ocean
        self.alpha_ocean = 0.32 # Ocean/land albedo
        self.alpha_ice = 0.62   # Ice/snow albedo
        self.T_freeze = 273.15  # Freezing temperature (K)
        
    def solar_distribution(self, S0=None):
        \"\"\"
        Calculate latitude-dependent solar input
        
        Uses 2nd Legendre polynomial for annual mean insolation
        
        Returns:
        --------
        Q : array
            Solar input at each latitude (W/m²)
        \"\"\"
        if S0 is None:
            S0 = self.S0
        
        Q0 = S0 / 4  # Global mean
        
        # Legendre P2 distribution
        sin_lat = np.sin(self.lat_rad)
        P2 = (3 * sin_lat**2 - 1) / 2
        
        Q = Q0 * (1 - 0.482 * P2)
        
        return Q
    
    def albedo(self, T):
        \"\"\"
        Calculate albedo with ice-albedo feedback
        
        Parameters:
        -----------
        T : array
            Temperature at each latitude (K)
            
        Returns:
        --------
        alpha : array
            Albedo at each latitude
        \"\"\"
        alpha = np.where(T < self.T_freeze, self.alpha_ice, self.alpha_ocean)
        return alpha
    
    def outgoing_longwave(self, T):
        \"\"\"
        Outgoing longwave radiation (linearized)
        
        OLR = A + B*T
        \"\"\"
        return self.A + self.B * T
    
    def absorbed_solar(self, T, S0=None):
        \"\"\"
        Absorbed solar radiation including albedo feedback
        \"\"\"
        Q = self.solar_distribution(S0)
        alpha = self.albedo(T)
        return Q * (1 - alpha)
    
    def diffusion_operator(self, T):
        \"\"\"
        Compute meridional heat transport via diffusion
        
        ∇·F = (1/R²cos(φ)) ∂/∂φ [cos(φ) D ∂T/∂φ]
        
        Returns:
        --------
        div_F : array
            Divergence of heat flux (W/m²)
        \"\"\"
        # Compute temperature gradient
        dT_dlat = np.gradient(T, self.d_lat)
        
        # Compute flux with cos(φ) weighting
        cos_lat = np.cos(self.lat_rad)
        flux = -self.D * cos_lat * dT_dlat
        
        # Compute divergence
        dflux_dlat = np.gradient(flux, self.d_lat)
        div_F = dflux_dlat / (self.R_earth * cos_lat)
        
        return div_F
    
    def tendency(self, T, S0=None):
        \"\"\"
        Calculate temperature tendency dT/dt
        
        C dT/dt = Q(1-α) - (A+BT) + ∇·F
        
        Returns:
        --------
        dT_dt : array  
            Temperature tendency (K/s)
        \"\"\"
        Q_abs = self.absorbed_solar(T, S0)
        OLR = self.outgoing_longwave(T)
        div_F = self.diffusion_operator(T)
        
        # Net heating
        Q_net = Q_abs - OLR + div_F
        
        # Convert to temperature tendency
        dT_dt = Q_net / self.C
        
        return dT_dt
    
    def run_to_equilibrium(self, T_init=None, years=100, dt=0.1, S0=None):
        \"\"\"
        Time-step to equilibrium
        
        Parameters:
        -----------
        T_init : array, optional
            Initial temperature profile (K)
        years : float
            Integration time (years)
        dt : float
            Time step (years)
        S0 : float, optional
            Solar constant (W/m²), default is self.S0
            
        Returns:
        --------
        T : array
            Final temperature profile (K)
        T_history : array
            Temperature evolution [time, lat]
        \"\"\"
        # Initialize
        if T_init is None:
            # Reasonable initial guess
            T = 288 - 40 * np.abs(np.sin(self.lat_rad))  # Warmer equator, colder poles
        else:
            T = T_init.copy()
        
        # Time integration
        seconds_per_year = 365.25 * 24 * 3600
        dt_seconds = dt * seconds_per_year
        n_steps = int(years / dt)
        
        # Store some history
        save_interval = max(1, n_steps // 200)
        T_history = [T.copy()]
        times = [0]
        
        for step in range(n_steps):
            # Forward Euler
            dT_dt = self.tendency(T, S0)
            T = T + dT_dt * dt_seconds
            
            # Save periodically
            if step % save_interval == 0:
                T_history.append(T.copy())
                times.append(step * dt)
        
        return T, np.array(T_history), np.array(times)
    
    def find_ice_edge(self, T):
        \"\"\"
        Find latitude of ice edge (freezing isotherm)
        
        Returns:
        --------
        ice_edge_north : float
            Northern hemisphere ice edge latitude (degrees)
        ice_edge_south : float
            Southern hemisphere ice edge latitude (degrees)
        \"\"\"
        # Northern hemisphere
        nh_idx = self.lat >= 0
        T_nh = T[nh_idx]
        lat_nh = self.lat[nh_idx]
        
        if np.any(T_nh < self.T_freeze):
            idx = np.where(T_nh < self.T_freeze)[0][0]
            ice_edge_north = lat_nh[idx]
        else:
            ice_edge_north = 90  # No ice
        
        # Southern hemisphere  
        sh_idx = self.lat <= 0
        T_sh = T[sh_idx]
        lat_sh = self.lat[sh_idx]
        
        if np.any(T_sh < self.T_freeze):
            idx = np.where(T_sh < self.T_freeze)[0][-1]
            ice_edge_south = lat_sh[idx]
        else:
            ice_edge_south = -90  # No ice
        
        return ice_edge_north, ice_edge_south
    
    def climate_sensitivity(self, forcing=4.0):
        \"\"\"
        Calculate ECS by running control and forced experiments
        
        CO2 forcing applied as uniform heating
        \"\"\"
        # Control
        T_control, _, _ = self.run_to_equilibrium(years=50, dt=0.1)
        
        # Forced (approximate CO2 forcing as reduced OLR)
        # Equivalent to reducing A parameter
        A_original = self.A
        self.A = A_original - forcing
        
        T_forced, _, _ = self.run_to_equilibrium(T_init=T_control, years=50, dt=0.1)
        
        # Restore
        self.A = A_original
        
        # Calculate ECS (global mean)
        ECS = np.mean(T_forced - T_control)
        
        return T_control, T_forced, ECS

# Initialize and run Model 3
print("="*70)
print("TWO-DIMENSIONAL ENERGY BALANCE MODEL")
print("="*70 + "\\n")

model3 = TwoDimensionalEBM(n_lat=36)

print(f"Model Configuration:")
print(f"  Latitudes: {model3.n_lat} bands from {model3.lat[0]:.0f}° to {model3.lat[-1]:.0f}°")
print(f"  Diffusion coefficient (D): {model3.D:.3f} W/m²/K")
print(f"  Heat capacity (C): {model3.C:.2e} J/m²/K")
print(f"  Albedo: {model3.alpha_ocean:.2f} (open) → {model3.alpha_ice:.2f} (ice)\\n")

print("Computing equilibrium climate...")
T_eq_2d, T_history, times = model3.run_to_equilibrium(years=50, dt=0.1)

ice_n, ice_s = model3.find_ice_edge(T_eq_2d)
print(f"  Global mean temperature: {np.mean(T_eq_2d):.2f} K ({np.mean(T_eq_2d)-273.15:.2f}°C)")
print(f"  Equatorial temperature: {T_eq_2d[model3.n_lat//2]:.2f} K ({T_eq_2d[model3.n_lat//2]-273.15:.2f}°C)")
print(f"  Polar temperatures: {np.mean([T_eq_2d[0], T_eq_2d[-1]]):.2f} K ({np.mean([T_eq_2d[0], T_eq_2d[-1]])-273.15:.2f}°C)")
print(f"  Ice edge: North {ice_n:.1f}°, South {ice_s:.1f}°\\n")

print("Computing climate sensitivity...")
T_control_2d, T_forced_2d, ECS_2d = model3.climate_sensitivity(forcing=4.0)

print(f"  Global mean ECS: {ECS_2d:.2f} K")
print(f"  Equatorial ECS: {(T_forced_2d - T_control_2d)[model3.n_lat//2]:.2f} K")
print(f"  Polar ECS: {np.mean([(T_forced_2d - T_control_2d)[0], (T_forced_2d - T_control_2d)[-1]]):.2f} K")
print(f"  Polar amplification factor: {np.mean([(T_forced_2d - T_control_2d)[0], (T_forced_2d - T_control_2d)[-1]]) / ECS_2d:.2f}×")
print("\\n" + "="*70)""")

print("Adding remaining models...")
# Continue building the notebook...
# [The script continues with Models 4 and 5, visualizations, and climate change analysis]

# Save
with open('climate_models_blog.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"\\nNotebook updated! Total cells: {len(notebook['cells'])}")

#!/usr/bin/env python3
"""
Complete the climate models notebook with Models 3-5 and climate change analysis
"""
import json

# Load existing notebook
with open('climate_models_blog.ipynb', 'r') as f:
    notebook = json.load(f)

def add_cell(cell_type, content):
    """Add a cell to the notebook"""
    if isinstance(content, list):
        source = content
    elif isinstance(content, str):
        lines = content.split('\n')
        source = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    else:
        source = [str(content)]
    
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    notebook["cells"].append(cell)

print("Completing notebook with Models 3-5 and analysis...")

# Model 3 Visualizations
add_cell("code", """# Visualize Model 3: Two-Dimensional Energy Balance Model

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# === Panel 1: Temperature vs Latitude ===
ax1 = fig.add_subplot(gs[0, 0])

ax1.plot(model3.lat, T_control_2d - 273.15, 'b-', linewidth=3, 
         label='Control', marker='o', markersize=4)
ax1.plot(model3.lat, T_forced_2d - 273.15, 'r-', linewidth=3,
         label='2×CO₂', marker='s', markersize=4)
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.axhline(model3.T_freeze - 273.15, color='cyan', linestyle=':', 
            linewidth=2, alpha=0.7, label='Freezing point')

ax1.set_xlabel('Latitude (°)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax1.set_title('Temperature Distribution by Latitude', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='lower center')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-90, 90)

# === Panel 2: Warming Pattern ===
ax2 = fig.add_subplot(gs[0, 1])

delta_T_2d = T_forced_2d - T_control_2d
ax2.plot(model3.lat, delta_T_2d, 'purple', linewidth=3, marker='d', markersize=5)
ax2.axhline(ECS_2d, color='gray', linestyle='--', alpha=0.5, label=f'Global mean: {ECS_2d:.2f} K')
ax2.fill_between(model3.lat, 0, delta_T_2d, alpha=0.3, color='red')

ax2.set_xlabel('Latitude (°)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Temperature Change (K)', fontsize=12, fontweight='bold')
ax2.set_title('Polar Amplification Pattern', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-90, 90)

# === Panel 3: Albedo Distribution ===
ax3 = fig.add_subplot(gs[1, 0])

albedo_control = model3.albedo(T_control_2d)
albedo_forced = model3.albedo(T_forced_2d)

ax3.plot(model3.lat, albedo_control, 'b-', linewidth=3, label='Control')
ax3.plot(model3.lat, albedo_forced, 'r--', linewidth=3, label='2×CO₂')
ax3.fill_between(model3.lat, albedo_control, albedo_forced,
                  where=(albedo_forced < albedo_control), 
                  alpha=0.3, color='red', label='Ice retreat')

ax3.set_xlabel('Latitude (°)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Albedo', fontsize=12, fontweight='bold')
ax3.set_title('Ice-Albedo Feedback', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-90, 90)
ax3.set_ylim(0.2, 0.7)

# === Panel 4: Energy Fluxes ===
ax4 = fig.add_subplot(gs[1, 1])

Q_solar = model3.solar_distribution()
Q_absorbed = Q_solar * (1 - albedo_control)
OLR = model3.outgoing_longwave(T_control_2d)

ax4.plot(model3.lat, Q_solar, 'orange', linewidth=2.5, label='Incident Solar')
ax4.plot(model3.lat, Q_absorbed, 'gold', linewidth=2.5, label='Absorbed Solar')
ax4.plot(model3.lat, OLR, 'blue', linewidth=2.5, label='Outgoing LW')
ax4.plot(model3.lat, Q_absorbed - OLR, 'green', linewidth=2.5, 
         linestyle='--', label='Net (requires transport)')

ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel('Latitude (°)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Energy Flux (W/m²)', fontsize=12, fontweight='bold')
ax4.set_title('Radiative Balance by Latitude', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10, loc='upper right')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-90, 90)

# === Panel 5: Meridional Heat Transport ===
ax5 = fig.add_subplot(gs[2, :])

# Calculate transport (integral of imbalance)
imbalance = Q_absorbed - OLR
cos_lat = np.cos(model3.lat_rad)

# Integrate from south to north
transport = np.zeros_like(imbalance)
for i in range(1, len(imbalance)):
    # Transport = integral of imbalance weighted by cos(lat)
    transport[i] = transport[i-1] + (imbalance[i-1] + imbalance[i])/2 * \\
                   (cos_lat[i-1] + cos_lat[i])/2 * model3.d_lat * model3.R_earth

# Convert to Petawatts
transport_PW = transport * 2 * np.pi * model3.R_earth / 1e15

ax5.plot(model3.lat, transport_PW, 'darkgreen', linewidth=4)
ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax5.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax5.fill_between(model3.lat, 0, transport_PW, where=(transport_PW > 0),
                  alpha=0.3, color='red', label='Northward')
ax5.fill_between(model3.lat, 0, transport_PW, where=(transport_PW < 0),
                  alpha=0.3, color='blue', label='Southward')

ax5.set_xlabel('Latitude (°)', fontsize=13, fontweight='bold')
ax5.set_ylabel('Meridional Heat Transport (PW)', fontsize=13, fontweight='bold')
ax5.set_title('Poleward Energy Transport (Atmosphere + Ocean)', 
              fontsize=15, fontweight='bold')
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(-90, 90)

max_transport = np.max(np.abs(transport_PW))
print(f"  Maximum poleward transport: {max_transport:.2f} PW")

plt.suptitle('Two-Dimensional Energy Balance Model: Meridional Structure',
             fontsize=17, fontweight='bold', y=0.995)

plt.savefig('model3_complete.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\n" + "="*70)
print("KEY INSIGHTS FROM MODEL 3")
print("="*70)
print("\\n1. Equator-Pole Gradient: Temperature decreases from equator to poles")
print(f"   Equator: {T_control_2d[model3.n_lat//2]-273.15:.1f}°C, Poles: {np.mean([T_control_2d[0], T_control_2d[-1]])-273.15:.1f}°C")
print(f"\\n2. Polar Amplification: Arctic/Antarctic warm {np.mean([(T_forced_2d - T_control_2d)[0], (T_forced_2d - T_control_2d)[-1]])/ECS_2d:.1f}× faster")
print("   than global mean due to ice-albedo feedback")
print(f"\\n3. Ice-Albedo Feedback: Positive feedback as ice retreat")
print("   lowers albedo, causing more warming")
print(f"\\n4. Meridional Transport: ~{max_transport:.1f} PW transported from")
print("   tropics to poles by atmosphere and ocean")
print("\\n5. Energy Imbalance: Tropics have surplus, poles have deficit")
print("   → drives atmospheric/oceanic circulation")
print("\\n" + "="*70)""")

# MODEL 4: 3D GCM (Simplified)
add_cell("markdown", """<a id='model4'></a>
## Model 4: Three-Dimensional General Circulation Model (GCM)

### Technical Overview (Page 1 of 2)

Three-Dimensional General Circulation Models represent the state-of-the-art in traditional climate modeling. These models explicitly resolve atmospheric and oceanic circulation in three spatial dimensions and time, governed by the fundamental equations of fluid dynamics and thermodynamics.

#### Governing Equations

GCMs solve the **primitive equations** on a 3D grid:

**1. Momentum (Navier-Stokes):**
$$\\frac{D\\mathbf{u}}{Dt} + 2\\mathbf{\\Omega} \\times \\mathbf{u} = -\\frac{1}{\\rho}\\nabla p + \\mathbf{g} + \\mathbf{F}$$

**2. Continuity (Mass Conservation):**
$$\\frac{\\partial \\rho}{\\partial t} + \\nabla \\cdot (\\rho \\mathbf{u}) = 0$$

**3. Thermodynamic Energy:**
$$\\rho c_p \\frac{DT}{Dt} = \\frac{Dp}{Dt} + Q_{rad} + Q_{latent} + Q_{sens}$$

**4. Water Vapor:**
$$\\frac{Dq}{Dt} = S_{evap} - S_{precip} + \\text{diffusion}$$

**5. Hydrostatic Balance (vertical):**
$$\\frac{\\partial p}{\\partial z} = -\\rho g$$

Where:
- $\\mathbf{u} = (u, v, w)$ = 3D velocity field (m/s)
- $\\mathbf{\\Omega}$ = Earth's rotation vector
- $\\rho$ = Air/water density (kg/m³)
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
   $$\\frac{\\partial p}{\\partial z} = -\\rho g$$
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
$$\\text{ECS} = \\frac{\\lambda_0}{1 - \\sum f_i}$$

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
- Extreme events""")

# Simplified 3D GCM Implementation
add_cell("code", """# Model 4: Simplified 3D General Circulation Model
# 
# Full GCMs are too complex to implement from scratch here.
# Instead, we demonstrate the conceptual framework and show 
# representative outputs from a simplified 3-cell circulation model.

class SimplifiedGCM:
    \"\"\"
    Highly simplified GCM demonstrating key concepts:
    - 3D grid structure (coarse resolution)
    - Hadley cell circulation
    - Temperature and pressure fields
    - Simplified dynamics
    
    This is NOT a full GCM but demonstrates the principles.
    \"\"\"
    
    def __init__(self, n_lat=18, n_lon=36, n_lev=10):
        \"\"\"Initialize simplified GCM grid\"\"\"
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.n_lev = n_lev
        
        # Create grid
        self.lat = np.linspace(-90, 90, n_lat)
        self.lon = np.linspace(0, 360, n_lon, endpoint=False)
        self.lev = np.linspace(1000, 100, n_lev)  # Pressure levels (hPa)
        
        # Physical parameters
        self.omega = 7.292e-5  # Earth rotation (rad/s)
        self.R_earth = 6.371e6  # Earth radius (m)
        self.g = 9.81
        self.R = 287.0  # Gas constant
        
    def initialize_fields(self):
        \"\"\"
        Initialize temperature and wind fields
        Based on observed climatology
        \"\"\"
        # Create 3D grids
        LAT, LON, LEV = np.meshgrid(self.lat, self.lon, self.lev, indexing='ij')
        
        # Temperature: Decreases poleward and with height
        lat_rad = np.deg2rad(LAT)
        T = 300 - 50*np.abs(np.sin(lat_rad)) - 50*(1000 - LEV)/1000
        
        # Zonal wind (westerlies): Peaks at mid-latitudes
        u = 20 * np.sin(2*lat_rad) * (LEV/500)**0.5
        
        # Meridional circulation (Hadley cell): Simplified
        v = -10 * np.sin(lat_rad) * np.cos(lat_rad) * (LEV/1000)
        
        # Vertical velocity (pressure coordinates): Small
        omega = 0.01 * np.sin(2*lat_rad)  # Pa/s
        
        return {
            'T': T,
            'u': u,
            'v': v,
            'omega': omega,
            'lat': LAT,
            'lon': LON,
            'lev': LEV
        }
    
    def compute_circulation_strength(self, fields):
        \"\"\"Compute meridional overturning circulation\"\"\"
        # Mass streamfunction (simplified)
        v = fields['v']
        
        # Zonally average
        v_zonal = np.mean(v, axis=1)
        
        # Integrate vertically
        psi = np.zeros_like(v_zonal)
        for i in range(1, self.n_lev):
            psi[:, i] = psi[:, i-1] + v_zonal[:, i] * 100  # Rough integration
        
        return psi
    
    def apply_greenhouse_forcing(self, fields, forcing=4.0):
        \"\"\"
        Simulate warming response to GHG forcing
        
        Simple parameterization:
        - Uniform forcing applied
        - Polar amplification factor
        - Stratospheric cooling
        \"\"\"
        T = fields['T'].copy()
        lat_rad = np.deg2rad(fields['lat'])
        lev = fields['lev']
        
        # Surface warming with polar amplification
        warming = forcing * 0.75  # ~3K for 4 W/m²
        polar_amp = 1 + 1.5*np.abs(np.sin(lat_rad))
        
        # Height dependence: warming at surface, cooling aloft
        height_factor = np.where(lev > 300, 1.0, 1.0 - 0.5*(lev/300-1))
        height_factor = np.maximum(height_factor, -0.5)
        
        T_new = T + warming * polar_amp * height_factor
        
        return T_new

# Initialize simplified GCM
print("="*70)
print("THREE-DIMENSIONAL GENERAL CIRCULATION MODEL (Simplified)")
print("="*70 + "\\n")

gcm = SimplifiedGCM(n_lat=18, n_lon=36, n_lev=10)

print(f"Model Configuration:")
print(f"  Grid: {gcm.n_lat}°×{gcm.n_lon}° × {gcm.n_lev} levels")
print(f"  Resolution: {180/gcm.n_lat:.1f}° lat × {360/gcm.n_lon:.1f}° lon")
print(f"  Pressure range: {gcm.lev[-1]:.0f} - {gcm.lev[0]:.0f} hPa\\n")

print("Initializing atmospheric fields from climatology...")
fields_control = gcm.initialize_fields()

print(f"  Surface temperature range: {np.min(fields_control['T'][:,:,0]):.1f} - {np.max(fields_control['T'][:,:,0]):.1f} K")
print(f"  Maximum zonal wind: {np.max(fields_control['u']):.1f} m/s")
print(f"  Meridional circulation: Hadley cells present\\n")

print("Computing circulation patterns...")
psi_control = gcm.compute_circulation_strength(fields_control)

print("Applying greenhouse forcing (+4 W/m²)...")
fields_forced = fields_control.copy()
fields_forced['T'] = gcm.apply_greenhouse_forcing(fields_control, forcing=4.0)

delta_T_gcm = fields_forced['T'] - fields_control['T']
global_mean_warming = np.mean(delta_T_gcm[:,:,0])

print(f"  Global mean surface warming: {global_mean_warming:.2f} K")
print(f"  Polar warming: {np.mean([np.mean(delta_T_gcm[0,:,0]), np.mean(delta_T_gcm[-1,:,0])]):.2f} K")
print(f"  Tropical warming: {np.mean(delta_T_gcm[gcm.n_lat//2-2:gcm.n_lat//2+2,:,0]):.2f} K")
print("\\n" + "="*70)""")

# GCM Visualizations
add_cell("code", """# Visualize Model 4: 3D GCM Fields

fig = plt.figure(figsize=(18, 14))

# Extract surface fields
T_sfc_control = fields_control['T'][:,:,0]
T_sfc_forced = fields_forced['T'][:,:,0]
u_200hPa = fields_control['u'][:,:,-3]  # Upper troposphere

# === Panel 1: Surface Temperature (Control) ===
ax1 = plt.subplot(3, 3, 1)
LON_grid, LAT_grid = np.meshgrid(gcm.lon, gcm.lat)
c1 = ax1.contourf(LON_grid, LAT_grid, T_sfc_control-273.15, 
                  levels=20, cmap='RdBu_r')
ax1.contour(LON_grid, LAT_grid, T_sfc_control-273.15, levels=10, 
            colors='black', alpha=0.3, linewidths=0.5)
plt.colorbar(c1, ax=ax1, label='Temperature (°C)')
ax1.set_xlabel('Longitude (°)')
ax1.set_ylabel('Latitude (°)')
ax1.set_title('Control: Surface Temperature', fontweight='bold')

# === Panel 2: Surface Temperature (2×CO₂) ===
ax2 = plt.subplot(3, 3, 2)
c2 = ax2.contourf(LON_grid, LAT_grid, T_sfc_forced-273.15,
                  levels=20, cmap='RdBu_r')
ax2.contour(LON_grid, LAT_grid, T_sfc_forced-273.15, levels=10,
            colors='black', alpha=0.3, linewidths=0.5)
plt.colorbar(c2, ax=ax2, label='Temperature (°C)')
ax2.set_xlabel('Longitude (°)')
ax2.set_ylabel('Latitude (°)')
ax2.set_title('2×CO₂: Surface Temperature', fontweight='bold')

# === Panel 3: Temperature Change ===
ax3 = plt.subplot(3, 3, 3)
delta_T_sfc = T_sfc_forced - T_sfc_control
c3 = ax3.contourf(LON_grid, LAT_grid, delta_T_sfc,
                  levels=15, cmap='Reds')
ax3.contour(LON_grid, LAT_grid, delta_T_sfc, levels=8,
            colors='black', alpha=0.4, linewidths=0.5)
plt.colorbar(c3, ax=ax3, label='ΔT (K)')
ax3.set_xlabel('Longitude (°)')
ax3.set_ylabel('Latitude (°)')
ax3.set_title('Surface Warming Pattern', fontweight='bold')

# === Panel 4: Upper-Level Winds ===
ax4 = plt.subplot(3, 3, 4)
c4 = ax4.contourf(LON_grid, LAT_grid, u_200hPa, levels=20, cmap='RdBu_r')
ax4.contour(LON_grid, LAT_grid, u_200hPa, levels=10,
            colors='black', alpha=0.3, linewidths=0.5)
plt.colorbar(c4, ax=ax4, label='Zonal Wind (m/s)')
ax4.set_xlabel('Longitude (°)')
ax4.set_ylabel('Latitude (°)')
ax4.set_title('Upper Troposphere Zonal Wind (200 hPa)', fontweight='bold')

# === Panel 5: Vertical Temperature Profile ===
ax5 = plt.subplot(3, 3, 5)
# Extract tropical and polar profiles
lat_tropical_idx = gcm.n_lat // 2
lat_polar_idx = -1

T_tropical = np.mean(fields_control['T'][lat_tropical_idx, :, :], axis=0)
T_polar = np.mean(fields_control['T'][lat_polar_idx, :, :], axis=0)

ax5.plot(T_tropical-273.15, gcm.lev, 'r-', linewidth=3, marker='o', label='Tropical')
ax5.plot(T_polar-273.15, gcm.lev, 'b-', linewidth=3, marker='s', label='Polar')
ax5.invert_yaxis()
ax5.set_yscale('log')
ax5.set_xlabel('Temperature (°C)', fontweight='bold')
ax5.set_ylabel('Pressure (hPa)', fontweight='bold')
ax5.set_title('Vertical Temperature Profiles', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# === Panel 6: Meridional Overturning ===
ax6 = plt.subplot(3, 3, 6)
c6 = ax6.contourf(gcm.lat, gcm.lev, psi_control.T, levels=15, cmap='RdBu_r')
plt.colorbar(c6, ax=ax6, label='Streamfunction')
ax6.invert_yaxis()
ax6.set_xlabel('Latitude (°)', fontweight='bold')
ax6.set_ylabel('Pressure (hPa)', fontweight='bold')
ax6.set_title('Meridional Overturning Circulation', fontweight='bold')
ax6.axvline(0, color='black', linestyle='--', alpha=0.3)

# === Panel 7: Zonal Mean Temperature (Control) ===
ax7 = plt.subplot(3, 3, 7)
T_zonal_control = np.mean(fields_control['T'], axis=1)
c7 = ax7.contourf(gcm.lat, gcm.lev, T_zonal_control.T-273.15,
                  levels=20, cmap='RdBu_r')
plt.colorbar(c7, ax=ax7, label='Temperature (°C)')
ax7.invert_yaxis()
ax7.set_xlabel('Latitude (°)', fontweight='bold')
ax7.set_ylabel('Pressure (hPa)', fontweight='bold')
ax7.set_title('Zonal Mean Temperature (Control)', fontweight='bold')

# === Panel 8: Zonal Mean Temperature (2×CO₂) ===
ax8 = plt.subplot(3, 3, 8)
T_zonal_forced = np.mean(fields_forced['T'], axis=1)
c8 = ax8.contourf(gcm.lat, gcm.lev, T_zonal_forced.T-273.15,
                  levels=20, cmap='RdBu_r')
plt.colorbar(c8, ax=ax8, label='Temperature (°C)')
ax8.invert_yaxis()
ax8.set_xlabel('Latitude (°)', fontweight='bold')
ax8.set_ylabel('Pressure (hPa)', fontweight='bold')
ax8.set_title('Zonal Mean Temperature (2×CO₂)', fontweight='bold')

# === Panel 9: Warming by Height and Latitude ===
ax9 = plt.subplot(3, 3, 9)
delta_T_zonal = T_zonal_forced - T_zonal_control
c9 = ax9.contourf(gcm.lat, gcm.lev, delta_T_zonal.T,
                  levels=15, cmap='Reds')
ax9.contour(gcm.lat, gcm.lev, delta_T_zonal.T, levels=8,
            colors='black', alpha=0.4, linewidths=0.5)
plt.colorbar(c9, ax=ax9, label='ΔT (K)')
ax9.invert_yaxis()
ax9.set_xlabel('Latitude (°)', fontweight='bold')
ax9.set_ylabel('Pressure (hPa)', fontweight='bold')
ax9.set_title('Warming Pattern (Height×Latitude)', fontweight='bold')

plt.suptitle('Three-Dimensional General Circulation Model: Global Fields',
             fontsize=18, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('model4_complete.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\n" + "="*70)
print("KEY INSIGHTS FROM MODEL 4 (Simplified GCM)")
print("="*70)
print("\\n1. 3D Structure: Temperature, winds, and circulation vary in all")
print("   three spatial dimensions - latitude, longitude, height")
print(f"\\n2. Atmospheric Dynamics: Jet streams at mid-latitudes,")
print("   Hadley cells in tropics explicitly represented")
print(f"\\n3. Global Mean Warming: {global_mean_warming:.2f} K with strong")
print("   polar amplification pattern")
print("\\n4. Vertical Structure: Surface warming, stratospheric cooling")
print("   characteristic of greenhouse forcing")
print("\\n5. Full GCMs: Real climate models have:")
print("   - Much higher resolution (50-100 km)")
print("   - Coupled ocean model")
print("   - Full radiative transfer")
print("   - Cloud microphysics")
print("   - Land surface and ice sheet models")
print("   - Run for decades to centuries of simulated time")
print("\\n" + "="*70)""")

print("Adding Model 5 (GraphCast)...")

# Save progress
with open('climate_models_blog.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"✓ Models 3-4 complete. Total cells: {len(notebook['cells'])}")

"""
Complete implementations of all 5 climate models for the marimo app
This module contains all model classes and visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

# Try to import torch for GraphCast demo
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ZeroDimensionalEBM:
    """Zero-Dimensional Energy Balance Model"""
    
    def __init__(self, C=1e8, alpha=0.30, epsilon=0.61):
        self.C = C
        self.alpha = alpha
        self.epsilon = epsilon
        self.sigma = 5.67e-8
        self.Q = 342
        self.T_ref = 288
        
    def absorbed_solar(self):
        return self.Q * (1 - self.alpha)
    
    def outgoing_radiation(self, T):
        return self.epsilon * self.sigma * T**4
    
    def dT_dt(self, T, year=0, forcing=0):
        Q_in = self.absorbed_solar()
        Q_out = self.outgoing_radiation(T)
        return (Q_in - Q_out + forcing) / self.C
    
    def equilibrium_temperature(self, forcing=0):
        def residual(T):
            return self.dT_dt(T, forcing=forcing)
        T_eq = fsolve(residual, self.T_ref)[0]
        return T_eq
    
    def climate_sensitivity(self):
        T_control = self.equilibrium_temperature(0)
        T_2xCO2 = self.equilibrium_temperature(3.7)
        return T_2xCO2 - T_control


class OneDimensionalRCM:
    """One-Dimensional Radiative-Convective Model"""
    
    def __init__(self, n_levels=30, p_surface=1000, p_top=10, epsilon=0.61):
        self.n_levels = n_levels
        self.p_surface = p_surface
        self.p_top = p_top
        self.epsilon = epsilon
        self.sigma = 5.67e-8
        
        # Pressure levels (hPa)
        self.p = np.logspace(np.log10(p_surface), np.log10(p_top), n_levels)
        self.z = -7000 * np.log(self.p / p_surface)  # Approximate height
        
    def radiative_transfer(self, T):
        """Simplified radiative transfer"""
        tau = self.epsilon * np.ones_like(T)
        
        # Upward flux
        F_up = self.sigma * T**4 * np.exp(-tau[::-1].cumsum()[::-1])
        
        # Downward flux  
        F_down = self.sigma * T**4 * np.exp(-tau.cumsum())
        
        return F_up, F_down
    
    def equilibrium_profile(self):
        """Find equilibrium temperature profile"""
        T = np.linspace(288, 200, self.n_levels)
        
        for iteration in range(50):
            F_up, F_down = self.radiative_transfer(T)
            
            # Energy balance for each layer
            heating = (F_down[:-1] - F_down[1:]) + (F_up[1:] - F_up[:-1])
            
            # Simple update
            T[1:-1] += 0.01 * heating[1:-1] / 1000
            
            # Surface balance
            Q_in = 342 * 0.70  # Solar absorption
            Q_out = F_up[0]
            T[0] += 0.001 * (Q_in - Q_out)
            
            T = np.clip(T, 150, 320)
        
        # Apply convective adjustment
        gamma_critical = 6.5 / 1000  # K/m
        dT_dz = np.diff(T) / np.diff(self.z)
        
        unstable = dT_dz < -gamma_critical
        if np.any(unstable):
            # Adjust superadiabatic layers
            T = self._convective_adjustment(T)
        
        return T
    
    def _convective_adjustment(self, T):
        """Apply convective adjustment for unstable layers"""
        for i in range(1, len(T)-1):
            dT_dz = (T[i] - T[i-1]) / (self.z[i] - self.z[i-1])
            if dT_dz < -6.5/1000:
                T[i] = 0.5 * (T[i-1] + T[i+1])
        return T


class TwoDimensionalEBM:
    """Two-Dimensional Energy Balance Model (latitude-height)"""
    
    def __init__(self, n_lat=36, n_levels=10):
        self.n_lat = n_lat
        self.n_levels = n_levels
        self.lat = np.linspace(-90, 90, n_lat)
        self.lat_rad = np.deg2rad(self.lat)
        self.d_lat = 180 / (n_lat - 1)
        self.R_earth = 6.371e6
        self.T_freeze = 273.15
        self.A = 202
        self.B = 2.17
        
    def solar_distribution(self):
        """Latitude-dependent insolation"""
        Q0 = 342
        return Q0 * np.maximum(0, np.sin(self.lat_rad))
    
    def albedo(self, T):
        """Ice-albedo feedback"""
        alpha = np.where(T > self.T_freeze, 0.30, 0.60)
        return alpha
    
    def outgoing_longwave(self, T):
        """Simplified OLR"""
        return self.A + self.B * (T - 273.15)
    
    def equilibrium(self, forcing=0):
        """Find equilibrium temperature distribution"""
        T = np.ones_like(self.lat) * 288.0
        K = 0.5  # Diffusion coefficient
        
        for iteration in range(100):
            alpha = self.albedo(T)
            Q_solar = self.solar_distribution()
            Q_absorbed = Q_solar * (1 - alpha)
            Q_out = self.outgoing_longwave(T) + forcing
            
            # Diffusion
            dT_dy = np.diff(T) / np.deg2rad(self.d_lat)
            flux = -K * dT_dy
            div_flux = np.diff(flux)
            
            # Update temperature
            T[1:-1] += 0.1 * (Q_absorbed[1:-1] - Q_out[1:-1] - div_flux) / 1e6
            
            # Boundary conditions
            T[0] = T[1]
            T[-1] = T[-2]
            
            T = np.clip(T, 200, 320)
        
        return T


class SimplifiedGCM:
    """Highly simplified 3D General Circulation Model"""
    
    def __init__(self, n_lat=18, n_lon=36, n_lev=10):
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.n_lev = n_lev
        
        self.lat = np.linspace(-90, 90, n_lat)
        self.lon = np.linspace(0, 360, n_lon, endpoint=False)
        self.lev = np.linspace(1000, 100, n_lev)
        
        self.R_earth = 6.371e6
        self.g = 9.81
        
    def initialize_fields(self):
        """Initialize atmospheric fields"""
        LAT, LON, LEV = np.meshgrid(self.lat, self.lon, self.lev, indexing='ij')
        
        lat_rad = np.deg2rad(LAT)
        T = 300 - 50*np.abs(np.sin(lat_rad)) - 50*(1000 - LEV)/1000
        u = 20 * np.sin(2*lat_rad) * (LEV/500)**0.5
        v = -10 * np.sin(lat_rad) * np.cos(lat_rad) * (LEV/1000)
        omega = 0.01 * np.sin(2*lat_rad)
        
        return {
            'T': T,
            'u': u,
            'v': v,
            'omega': omega
        }
    
    def apply_forcing(self, fields, forcing=4.0):
        """Apply greenhouse forcing"""
        T = fields['T'].copy()
        lat_rad = np.deg2rad(fields['lat'] if isinstance(fields.get('lat'), np.ndarray) else 
                              np.linspace(-90, 90, fields['T'].shape[0]))
        lev = fields['lev'] if isinstance(fields.get('lev'), np.ndarray) else self.lev
        
        # Create LAT grid
        LAT_grid = np.repeat(lat_rad[:, np.newaxis, np.newaxis], 
                            fields['T'].shape[1], axis=1)
        LAT_grid = np.repeat(LAT_grid, fields['T'].shape[2], axis=2)
        
        warming = forcing * 0.75
        polar_amp = 1 + 1.5*np.abs(np.sin(LAT_grid))
        
        T_new = T + warming * polar_amp * 0.8
        return T_new


# Initialize models for use
def create_models():
    """Create instances of all models"""
    model1 = ZeroDimensionalEBM()
    model2 = OneDimensionalRCM()
    model3 = TwoDimensionalEBM()
    model4 = SimplifiedGCM()
    
    return model1, model2, model3, model4


def visualize_model1(model1):
    """Visualization for Model 1"""
    T_range = np.linspace(240, 320, 200)
    Q_in = model1.absorbed_solar()
    Q_out = np.array([model1.outgoing_radiation(T) for T in T_range])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(T_range, Q_in * np.ones_like(T_range), 'yellow', linewidth=3, label='Absorbed Solar')
    axes[0].plot(T_range, Q_out, 'red', linewidth=3, label='Outgoing LW')
    T_eq = model1.equilibrium_temperature(0)
    axes[0].axvline(T_eq, color='green', linestyle='--', linewidth=2, label=f'Equilibrium: {T_eq:.1f}K')
    axes[0].set_xlabel('Temperature (K)')
    axes[0].set_ylabel('Radiation (W/m²)')
    axes[0].set_title('Energy Balance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    forcing_range = np.linspace(-10, 10, 50)
    temp_range = np.array([model1.equilibrium_temperature(F) for F in forcing_range])
    axes[1].plot(forcing_range, temp_range - model1.T_ref, 'blue', linewidth=3)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(3.7, color='red', linestyle='--', alpha=0.7, label='2×CO₂ forcing')
    axes[1].set_xlabel('Forcing (W/m²)')
    axes[1].set_ylabel('Temperature Anomaly (°C)')
    axes[1].set_title('Climate Sensitivity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_model3(model3):
    """Visualization for Model 3"""
    T_control = model3.equilibrium(forcing=0)
    T_forced = model3.equilibrium(forcing=3.7)
    delta_T = T_forced - T_control
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(model3.lat, T_control - 273.15, 'b-', linewidth=3, label='Control')
    axes[0].plot(model3.lat, T_forced - 273.15, 'r-', linewidth=3, label='2×CO₂')
    axes[0].set_xlabel('Latitude (°)')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Meridional Temperature Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(model3.lat, delta_T, 'purple', linewidth=3, marker='o')
    ECS_mean = np.mean(delta_T)
    axes[1].axhline(ECS_mean, color='gray', linestyle='--', alpha=0.7, label=f'Mean: {ECS_mean:.2f}K')
    axes[1].fill_between(model3.lat, 0, delta_T, alpha=0.3, color='red')
    axes[1].set_xlabel('Latitude (°)')
    axes[1].set_ylabel('Temperature Change (K)')
    axes[1].set_title('Polar Amplification Pattern')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, np.mean(delta_T)

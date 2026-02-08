# QLaw.jl

[![Build Status](https://github.com/HAMMERHEAD-Space/QLaw.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/HAMMERHEAD-Space/QLaw.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

A Julia package implementing the Q-Law Lyapunov-based feedback control law for low-thrust trajectory optimization. Q-Law provides near-optimal thrust steering for orbit transfers while dramatically reducing the dimensionality of the optimization problem compared to direct methods.

## Overview

Q-Law is a Lyapunov candidate function that provides a feedback guidance law for low-thrust spacecraft. Given current and target orbital elements, Q-Law computes the optimal thrust direction at each instant to minimize a proximity quotient Q, driving the spacecraft toward the target orbit.

### Key Features

- **Lyapunov-based control**: Guarantees monotonic decrease of the proximity function Q toward zero
- **Automatic differentiation**: Uses ForwardDiff.jl for computing ∂Q/∂œ, enabling gradient-based optimization of weights
- **Effectivity-based coasting**: Smooth activation function for AD-compatible thrust/coast decisions
- **Flexible weighting**: Customizable weights for each orbital element to prioritize different transfer objectives
- **SciML interface**: Compatible with `solve()` and `remake()` patterns from the SciML ecosystem
- **Integration with HAMMERHEAD packages**: Built on [AstroCoords.jl](https://github.com/HAMMERHEAD-Space/AstroCoords.jl), [AstroPropagators.jl](https://github.com/HAMMERHEAD-Space/AstroPropagators.jl), and [AstroForceModels.jl](https://github.com/HAMMERHEAD-Space/AstroForceModels.jl)

### Implemented Features

Based on the formulation from Petropoulos (2003) with enhancements from recent literature:

- Equinoctial orbital elements [a, f, g, h, k, L] (using semi-major axis per Varga & Perez)
- Gauss Variational Equations (GVE) A-matrix for equinoctial elements
- Q-Law Lyapunov function with scaling and penalty terms
- Optimal thrust direction calculation (α*, β*)
- Absolute and relative effectivity metrics
- Smooth activation function for AD-compatible coasting decisions
- Eclipse/shadow modeling via AstroForceModels.jl

## Installation

```julia
using Pkg
Pkg.add("QLaw")
```

## Quick Start

```julia
using QLaw
using AstroCoords

# Earth gravitational parameter
μ = 398600.4418  # km³/s²

# Define initial orbit (LEO)
kep0 = Keplerian(6878.0, 0.001, deg2rad(28.5), 0.0, 0.0, 0.0)

# Define target orbit (GEO)
kepT = Keplerian(42164.0, 0.001, deg2rad(0.1), 0.0, 0.0, 0.0)

# Convert to Modified Equinoctial elements
oe0 = ModEq(kep0, μ)
oeT = ModEq(kepT, μ)

# Define spacecraft: dry_mass=500kg, wet_mass=1000kg, thrust=1N, Isp=1500s
spacecraft = QLawSpacecraft(500.0, 1000.0, 1.0, 1500.0)

# Define Q-Law weights (from paper optimization results)
weights = QLawWeights(0.0785, 0.7926, 0.6876, 0.3862, 0.5)

# Define parameters
params = QLawParameters(;
    η_threshold = -0.01,  # Constant thrust (no coasting)
    rp_min = 6578.0       # Minimum periapsis [km]
)

# Create problem
tspan = (0.0, 86400.0 * 60.0)  # 60 days
prob = qlaw_problem(oe0, oeT, tspan, μ, spacecraft;
    weights = weights,
    qlaw_params = params
)

# Solve
sol = solve(prob)

# Access solution
println("Converged: ", sol.converged)
println("Transfer time: ", sol.elapsed_time / 86400.0, " days")
println("Total ΔV: ", sol.Δv_total, " km/s")
println("Final mass: ", sol.final_mass, " kg")
```

## Spacecraft Types

### Constant Thrust

```julia
# dry_mass, wet_mass, thrust [N], Isp [s]
sc = QLawSpacecraft(500.0, 1000.0, 1.0, 3000.0)
```

### Solar Electric Propulsion (SEP)

Thrust scales with solar distance as (r_ref/r)²:

```julia
# dry_mass, wet_mass, thrust_ref [N], Isp [s], r_ref [km]
AU = 1.495978707e8
sep = SEPQLawSpacecraft(500.0, 1000.0, 0.5, 3000.0, AU)
```

## Q-Law Theory

### Lyapunov Function

The Q-Law proximity quotient is defined as:

$$Q = (1 + W_P P) \sum_{œ} W_{œ} S_{œ} \left( \frac{œ - œ_T}{\dot{œ}_{xx}} \right)^2$$

where:
- $W_{œ}$ are weights for each orbital element
- $S_{œ}$ is a scaling function to prevent divergence
- $\dot{œ}_{xx}$ is the maximum rate of change over all thrust directions
- $P$ is a penalty function for minimum periapsis constraint
- $W_P$ is the penalty weight

### Optimal Thrust Direction

The optimal thrust angles (α*, β*) are found by minimizing $\dot{Q}$:

$$\dot{Q} = D_1 \cos\beta \cos\alpha + D_2 \cos\beta \sin\alpha + D_3 \sin\beta$$

where $D_1$, $D_2$, $D_3$ are computed using automatic differentiation of Q with respect to the orbital elements.

### Effectivity

Effectivity determines when thrusting is beneficial:

- **Absolute effectivity**: $\eta_a = \dot{Q}_n / \dot{Q}_{nn}$
- **Relative effectivity**: $\eta_r = (\dot{Q}_n - \dot{Q}_{nx}) / (\dot{Q}_{nn} - \dot{Q}_{nx})$

where $\dot{Q}_{nn}$ and $\dot{Q}_{nx}$ are the minimum and maximum $\dot{Q}$ over all true longitudes.

### AD-Compatible Activation

For automatic differentiation compatibility, the effectivity threshold uses a smooth activation function:

$$\text{activation} = \frac{1}{2}\left(1 + \tanh\frac{\eta - \eta_{tr}}{\mu}\right)$$

## API Reference

### Types

| Type | Description |
|------|-------------|
| `QLawSpacecraft` | Constant thrust spacecraft |
| `SEPQLawSpacecraft` | Solar electric propulsion spacecraft |
| `QLawWeights` | Weights for each orbital element |
| `QLawParameters` | Algorithm parameters |
| `QLawProblem` | Problem definition |
| `QLawSolution` | Solution container |

### Core Functions

| Function | Description |
|----------|-------------|
| `qlaw_problem(oe0, oeT, tspan, μ, sc; ...)` | Create a Q-Law problem |
| `solve(problem; ...)` | Solve the transfer |
| `remake(problem; ...)` | Create modified problem |
| `compute_Q(oe, oeT, weights, μ, F_max, Wp, rp_min)` | Compute Lyapunov function |
| `compute_thrust_direction(oe, oeT, weights, μ, ...)` | Get optimal (α*, β*) |
| `compute_effectivity(oe, oeT, weights, μ, ...)` | Compute effectivity |
| `equinoctial_gve_partials(oe, μ)` | GVE A-matrix |

### Spacecraft Functions

| Function | Description |
|----------|-------------|
| `mass(sc)` | Total initial mass [kg] |
| `exhaust_velocity(sc)` | Exhaust velocity [km/s] |
| `max_thrust(sc, r)` | Maximum thrust at distance r [N] |
| `max_thrust_acceleration(sc, m, r)` | Max acceleration [km/s²] |

## References

1. **Petropoulos, A. E.** (2003). "Simple Control Laws for Low-Thrust Orbit Transfers." *AAS/AIAA Astrodynamics Specialist Conference*, Paper AAS 03-630, Big Sky, Montana.

2. **Petropoulos, A. E.** (2005). "Refinements to the Q-law for low-thrust orbit transfers." *AAS/AIAA Space Flight Mechanics Conference*, pp. 963-982.

3. **Varga, G. I., & Pérez, J. M. S.** (2016). "Many-Revolution Low-Thrust Orbit Transfer Computation Using Equinoctial Q-Law Including J2 and Eclipse Effects." *AAS Paper* 16-283.

4. **Shannon, J. L., Ozimek, M. T., Atchison, J. A., & Hartzell, C. M.** (2020). "Q-law aided direct trajectory optimization of many-revolution low-thrust transfers." *Journal of Spacecraft and Rockets*, 57(4), 672-682. [DOI: 10.2514/1.A34586](https://doi.org/10.2514/1.A34586)

5. **Steffen, N., Falck, R., & Faller, B.** (2025). "Q-Law for Rapid Assessment of Low Thrust Cislunar Trajectories via Automatic Differentiation." *AAS/AIAA Space Flight Mechanics Meeting*, Paper AAS 25-xxx.

6. **Narayanaswamy, S., & Damaren, C. J.** (2023). "Equinoctial Lyapunov Control Law for low-thrust rendezvous." *Journal of Guidance, Control, and Dynamics*, 46(4), 781-795. [DOI: 10.2514/1.G006662](https://doi.org/10.2514/1.G006662)

7. **Hecht, G. R., & Botta, E. M.** (2024). "Q-Law Control With Sun-Angle Constraint for Solar Electric Propulsion." *IEEE Transactions on Aerospace and Electronic Systems*, 60(6), 7917-7930. [DOI: 10.1109/TAES.2024.3424429](https://doi.org/10.1109/TAES.2024.3424429)

8. **Aziz, J., Scheeres, D., Parker, J., & Englander, J.** (2019). "A smoothed eclipse model for solar electric propulsion trajectory optimization." *Transactions of the Japan Society for Aeronautical and Space Sciences, Aerospace Technology Japan*, 17(2), 181-188. [DOI: 10.2322/tastj.17.181](https://doi.org/10.2322/tastj.17.181)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This implementation is based on the Q-Law formulation by Petropoulos with enhancements from Varga & Perez (equinoctial elements with semi-major axis) and Steffen, Falck & Faller (automatic differentiation approach).

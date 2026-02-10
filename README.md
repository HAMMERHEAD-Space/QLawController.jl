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
- **Convergence criteria**: Choose between summed-error (`SummedErrorConvergence`) or Q-function-based (`VargaConvergence`) stopping conditions
- **Weight optimization**: Optimize Q-Law weights using global (BlackBoxOptim) or local (SAMIN) solvers via Optimization.jl
- **Perturbation models**: Gravity harmonics, third-body (Moon/Sun), and eclipse/shadow effects via AstroForceModels.jl
- **SciML interface**: Compatible with `solve()` and `remake()` patterns from the SciML ecosystem
- **Integration with HAMMERHEAD packages**: Built on [AstroCoords.jl](https://github.com/HAMMERHEAD-Space/AstroCoords.jl), [AstroPropagators.jl](https://github.com/HAMMERHEAD-Space/AstroPropagators.jl), and [AstroForceModels.jl](https://github.com/HAMMERHEAD-Space/AstroForceModels.jl)

### Implemented Features

Based on the formulation from Petropoulos (2003) with enhancements from Varga & Perez (2016):

- Modified equinoctial orbital elements [p, f, g, h, k, L] with semi-major axis conversion (Varga & Perez)
- Gauss Variational Equations (GVE) A-matrix for equinoctial elements
- Q-Law Lyapunov function with scaling (Varga Eq. 8) and penalty terms
- Optimal thrust direction calculation (α\*, β\*)
- Absolute and relative effectivity metrics (`AbsoluteEffectivity`, `RelativeEffectivity`)
- Smooth activation function for AD-compatible coasting decisions
- Eclipse/shadow modeling via AstroForceModels.jl (conical and cylindrical)
- Two convergence criteria: `SummedErrorConvergence` and `VargaConvergence`

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

# Define initial orbit (LEO) and target orbit (GEO)
kep0 = Keplerian(6878.0, 0.0, deg2rad(28.5), 0.0, 0.0, 0.0)
kepT = Keplerian(42164.0, 0.0, 0.0, 0.0, 0.0, 0.0)

oe0 = ModEq(kep0, μ)
oeT = ModEq(kepT, μ)

# Spacecraft: dry_mass=500 kg, wet_mass=1000 kg, thrust=1.445 N, Isp=1850 s
spacecraft = QLawSpacecraft(500.0, 1000.0, 1.445, 1850.0)

# Q-Law weights (Wa, Wf, Wg, Wh, Wk)
weights = QLawWeights(1.0, 1.0, 1.0, 1.0, 1.0)

# Parameters
params = QLawParameters(;
    η_threshold = 0.2,
    rp_min = 6578.0,
    convergence_criterion = SummedErrorConvergence(0.05)
)

# Create and solve
tspan = (0.0, 86400.0 * 100.0)  # 100 days max
prob = qlaw_problem(oe0, oeT, tspan, μ, spacecraft;
    weights = weights,
    qlaw_params = params
)

sol = solve(prob)

# Access results
println("Converged: ", sol.converged)
println("Transfer time: ", sol.elapsed_time / 86400.0, " days")
println("Final mass: ", sol.final_mass, " kg")
println("Final SMA: ", get_sma(sol.final_oe), " km")
```

## Spacecraft Types

### Constant Thrust

```julia
# QLawSpacecraft(dry_mass [kg], wet_mass [kg], thrust [N], Isp [s])
sc = QLawSpacecraft(500.0, 1000.0, 1.445, 1850.0)
```

### Solar Electric Propulsion (SEP)

Thrust scales with solar distance as (r\_ref / r)²:

```julia
# SEPQLawSpacecraft(dry_mass [kg], wet_mass [kg], thrust_ref [N], Isp [s], r_ref [km])
AU = 1.495978707e8
sep = SEPQLawSpacecraft(500.0, 1000.0, 0.5, 3000.0, AU)
```

## Effectivity Types

Effectivity determines when thrusting is beneficial vs when to coast. Selectable via the `effectivity_type` keyword in `QLawParameters`:

### AbsoluteEffectivity (default)

$$\eta_a = \dot{Q}_n / \dot{Q}_{nn}$$

```julia
params = QLawParameters(; effectivity_type = AbsoluteEffectivity())
```

### RelativeEffectivity

$$\eta_r = (\dot{Q}_n - \dot{Q}_{nx}) / (\dot{Q}_{nn} - \dot{Q}_{nx})$$

```julia
params = QLawParameters(; effectivity_type = RelativeEffectivity())
```

## Convergence Criteria

Two stopping criteria are available, selectable via the `convergence_criterion` keyword in `QLawParameters`:

### SummedErrorConvergence (default)

Converges when the weighted sum of normalized orbital element errors drops below a tolerance:

$$\sum_{i} \frac{|œ_i - œ_{T,i}|}{|œ_{T,i}|} < \text{tol}$$

```julia
params = QLawParameters(; convergence_criterion = SummedErrorConvergence(0.05))
```

### VargaConvergence

Based on the Q-function value (Varga Eq. 35). The criterion normalizes Q by the target orbit's characteristic timescale squared (a\_T³/μ) to make it independent of physical units:

$$Q \cdot \frac{\mu}{a_T^3} < R_c \cdot \sqrt{\sum W_{œ}}$$

With the default `Rc=1.0` (the paper's nominal value), this converges when the orbit is within approximately 0.5--1% of the target elements.

```julia
params = QLawParameters(; convergence_criterion = VargaConvergence(1.0))
```

## Weight Optimization

Q-Law performance depends heavily on the choice of weights. QLaw.jl is compatible with Optimization.jl for automated weight tuning.

### Global Search with BlackBoxOptim

```julia
using Optimization, OptimizationBBO

function objective(x, p)
    weights = QLawWeights(x[1], x[2], x[3], x[4], x[5])
    params  = QLawParameters(; η_threshold = x[6], ...)
    prob    = qlaw_problem(oe0, oeT, tspan, μ, spacecraft; weights = weights, qlaw_params = params)
    sol     = solve(prob)
    return sol.converged ? sol.elapsed_time / tspan[2] : 1e10
end

lb = [0.01, 0.01, 0.01, 0.01, 0.01, -0.01]
ub = [1.0,  1.0,  1.0,  1.0,  1.0,   0.3]
x0 = (lb .+ ub) ./ 2

opt_f = OptimizationFunction(objective)
opt_prob = OptimizationProblem(opt_f, x0, p; lb = lb, ub = ub)
sol = Optimization.solve(opt_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(); maxiters = 500)
```

See `examples/weight_optimization.jl` for a complete example including DOE-seeded local refinement with SAMIN.

## Examples

The `examples/` directory contains runnable scripts:

| Example | Description |
|---------|-------------|
| `leo_to_geo.jl` | LEO-to-GEO transfer with J2 + Moon + Sun perturbations, plotting |
| `leo_to_meo_hifi.jl` | High-fidelity equatorial LEO to inclined MEO (GPS-like) with 36x36 harmonics, Sun/Moon, SRP, and drag |
| `earth_to_mars_sep.jl` | Heliocentric Earth-to-Mars transfer with `SEPQLawSpacecraft` (thrust scales as 1/r²) |
| `weight_optimization.jl` | BBO global search and DOE + SAMIN local refinement for weight tuning |
| `convergence_criteria_comparison.jl` | Side-by-side comparison of `SummedErrorConvergence` vs `VargaConvergence` |

To run an example:

```julia
using Pkg
Pkg.activate("examples")
Pkg.instantiate()
include("examples/leo_to_geo.jl")
```

## Q-Law Theory

### Lyapunov Function

The Q-Law proximity quotient is defined as:

$$Q = (1 + W_P P) \sum_{œ} W_{œ} S_{œ} \left( \frac{œ - œ_T}{\dot{œ}_{xx}} \right)^2$$

where:
- $W_{œ}$ are weights for each orbital element
- $S_{œ}$ is a scaling function to prevent divergence (Varga Eq. 8)
- $\dot{œ}_{xx}$ is the maximum rate of change over all thrust directions
- $P$ is a penalty function for minimum periapsis constraint
- $W_P$ is the penalty weight

### Optimal Thrust Direction

The optimal thrust angles (α\*, β\*) are found by minimizing $\dot{Q}$:

$$\dot{Q} = D_1 \cos\beta \cos\alpha + D_2 \cos\beta \sin\alpha + D_3 \sin\beta$$

where $D_1$, $D_2$, $D_3$ are computed using automatic differentiation of Q with respect to the orbital elements.

### Effectivity

Effectivity determines when thrusting is beneficial:

- **Absolute effectivity**: $\eta_a = \dot{Q}_n / \dot{Q}_{nn}$
- **Relative effectivity**: $\eta_r = (\dot{Q}_n - \dot{Q}_{nx}) / (\dot{Q}_{nn} - \dot{Q}_{nx})$

where $\dot{Q}_{nn}$ and $\dot{Q}_{nx}$ are the minimum and maximum $\dot{Q}$ over all true longitudes.

A smooth activation function is used for AD compatibility:

$$\text{activation} = \frac{1}{2}\left(1 + \tanh\frac{\eta - \eta_\text{th}}{\mu}\right)$$

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

This implementation is based on the Q-Law formulation by Petropoulos with enhancements from Varga & Perez (equinoctial elements with semi-major axis, scaling, J2/eclipse effects) and Steffen, Falck & Faller (automatic differentiation approach).

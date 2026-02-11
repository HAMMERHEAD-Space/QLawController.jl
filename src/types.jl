# =============================================================================
# Spacecraft Types (following SimsFlanagan patterns)
# =============================================================================

export QLawSpacecraft, SEPQLawSpacecraft
export QLawWeights
export QLawParameters
export AbstractConvergenceCriterion,
    SummedErrorConvergence, VargaConvergence, MaxElementConvergence
export AbstractEffectivityType, AbsoluteEffectivity, RelativeEffectivity
export AbstractEffectivitySearch, GridSearch, RefinedSearch
export QLawProblem
export QLawSolution
export qlaw_problem
export mass,
    exhaust_velocity, max_thrust, max_thrust_acceleration, max_orbit_thrust_acceleration

"""
    AbstractQLawSpacecraft

Abstract type for spacecraft used in Q-Law transfers.
"""
abstract type AbstractQLawSpacecraft end

"""
    QLawSpacecraft{Td,Tw,Tt,Ti}

Constant thrust spacecraft for Q-Law transfers.

# Fields
- `dry_mass::Td`: Dry mass [kg]
- `wet_mass::Tw`: Initial wet mass (dry + propellant) [kg]
- `thrust::Tt`: Maximum thrust [N]
- `Isp::Ti`: Specific impulse [s]
"""
struct QLawSpacecraft{Td<:Number,Tw<:Number,Tt<:Number,Ti<:Number} <: AbstractQLawSpacecraft
    dry_mass::Td
    wet_mass::Tw
    thrust::Tt
    Isp::Ti
end

"""
    SEPQLawSpacecraft{Td,Tw,Tt,Ti,Tr}

Solar Electric Propulsion spacecraft where thrust scales with solar distance.

# Fields
- `dry_mass::Td`: Dry mass [kg]
- `wet_mass::Tw`: Initial wet mass [kg]
- `thrust_ref::Tt`: Reference thrust at r_ref [N]
- `Isp::Ti`: Specific impulse [s]
- `r_ref::Tr`: Reference distance (typically 1 AU) [km]
"""
struct SEPQLawSpacecraft{Td<:Number,Tw<:Number,Tt<:Number,Ti<:Number,Tr<:Number} <:
       AbstractQLawSpacecraft
    dry_mass::Td
    wet_mass::Tw
    thrust_ref::Tt
    Isp::Ti
    r_ref::Tr
end

# Accessor functions
"""Get total initial mass [kg]"""
mass(sc::QLawSpacecraft) = sc.wet_mass
mass(sc::SEPQLawSpacecraft) = sc.wet_mass

"""Get exhaust velocity [km/s]"""
const G0 = 9.80665e-3  # km/s²
exhaust_velocity(sc::AbstractQLawSpacecraft) = sc.Isp * G0

"""Get maximum thrust [N] at given distance"""
max_thrust(sc::QLawSpacecraft, r::Number) = sc.thrust
max_thrust(sc::SEPQLawSpacecraft, r::Number) = sc.thrust_ref * (sc.r_ref / r)^2

"""Get maximum thrust acceleration [km/s²] at given state"""
function max_thrust_acceleration(sc::AbstractQLawSpacecraft, m::Number, r::Number)
    T_N = max_thrust(sc, r)
    # Convert N to km/s² (divide by mass in kg, multiply by 1e-3)
    return T_N / m * 1e-3
end

"""
    max_orbit_thrust_acceleration(sc::AbstractQLawSpacecraft, m::Number, oe::ModEq)

Compute the maximum thrust acceleration [km/s²] over the entire osculating orbit.

For constant-thrust spacecraft this equals the value at any orbital position.
For SEP spacecraft (thrust ∝ 1/r²), the maximum occurs at periapsis where r is
smallest. Using the orbit-maximum thrust for Q-Law normalization (max_rates) ensures
that the Q function is consistently normalized regardless of orbital position,
preventing artificial Q variations from thrust scaling on eccentric orbits.
"""
function max_orbit_thrust_acceleration(sc::AbstractQLawSpacecraft, m::Number, oe::ModEq)
    e = sqrt(oe.f^2 + oe.g^2)
    a = p_to_a(oe.p, oe.f, oe.g)
    rp = a * (one(typeof(e)) - e)  # Periapsis radius (min r → max thrust for SEP)
    return max_thrust_acceleration(sc, m, rp)
end

# =============================================================================
# Q-Law Weights
# =============================================================================

"""
    QLawWeights{Ta,Tf,Tg,Th,Tk}

Weights for each orbital element in Q-Law Lyapunov function.

Following the paper notation:
- `Wa`: Weight for semi-major axis
- `Wf`: Weight for f (eccentricity component)
- `Wg`: Weight for g (eccentricity component)
- `Wh`: Weight for h (inclination component)
- `Wk`: Weight for k (inclination component)

Note: L (true longitude) is not targeted in Q-Law.
"""
struct QLawWeights{Ta<:Number,Tf<:Number,Tg<:Number,Th<:Number,Tk<:Number}
    Wa::Ta
    Wf::Tf
    Wg::Tg
    Wh::Th
    Wk::Tk
end

"""Create weights with all values equal"""
QLawWeights(w::T) where {T<:Number} = QLawWeights{T,T,T,T,T}(w, w, w, w, w)

"""Create default weights (all ones)"""
QLawWeights() = QLawWeights(1.0, 1.0, 1.0, 1.0, 1.0)

# =============================================================================
# Convergence Criteria
# =============================================================================

"""
    AbstractEffectivityType

Abstract type for effectivity computation method.
"""
abstract type AbstractEffectivityType end

"""
    AbsoluteEffectivity

Absolute effectivity: ηa = Q̇n / Q̇nn (Eq. 29).
"""
struct AbsoluteEffectivity <: AbstractEffectivityType end

"""
    RelativeEffectivity

Relative effectivity: ηr = (Q̇n - Q̇nx) / (Q̇nn - Q̇nx) (Eq. 30).
"""
struct RelativeEffectivity <: AbstractEffectivityType end

"""
    AbstractEffectivitySearch

Abstract type for effectivity extrema search method.
"""
abstract type AbstractEffectivitySearch end

"""
    GridSearch

Find Q̇ extrema using only a uniform grid over true longitude.
Fast but may miss sharp extrema on eccentric orbits.
"""
struct GridSearch <: AbstractEffectivitySearch end

"""
    RefinedSearch

Find Q̇ extrema using a grid scan followed by Brent's method refinement
via Optim.jl. More accurate, especially for eccentric orbits where Q̇
varies rapidly with true longitude (Varga Section 2.2).
"""
struct RefinedSearch <: AbstractEffectivitySearch end

"""
    AbstractConvergenceCriterion

Abstract type for Q-Law convergence criteria.
"""
abstract type AbstractConvergenceCriterion end

"""
    SummedErrorConvergence{T}

Convergence based on summed per-element relative errors.
Stop when Σ relative_errors < tol.

Each element error is computed relative to its characteristic scale:
- Semi-major axis: |a - aT| / aT
- f, g: |Δf| / max(eT, ε), |Δg| / max(eT, ε)  where eT = √(fT² + gT²)
- h, k: |Δh| / max(sT, ε), |Δk| / max(sT, ε)  where sT = √(hT² + kT²)
with ε = 0.01 floor for near-zero targets.

Note: Because errors are summed, a large error in one element can be masked
by small errors in others. Consider `MaxElementConvergence` for stricter
per-element guarantees.
"""
struct SummedErrorConvergence{T<:Number} <: AbstractConvergenceCriterion
    tol::T
end
SummedErrorConvergence() = SummedErrorConvergence(0.05)

"""
    VargaConvergence{T}

Convergence based on Q-function value (Varga Eq. 35).

The criterion normalizes Q by the target orbit's characteristic timescale
squared (aT³/μ) to make it independent of physical units:
    Q * μ / aT³ < Rc * √(Σ Woe)
With this normalization, `Rc` is truly dimensionless and `Rc=1` (the paper's
nominal value) gives convergence when the orbit is within ~0.5-1% of the target.
"""
struct VargaConvergence{T<:Number} <: AbstractConvergenceCriterion
    Rc::T
end
VargaConvergence() = VargaConvergence(1.0)

"""
    MaxElementConvergence{T}

Convergence based on maximum per-element relative error.
Stop when max(relative_errors) < tol.

Each element error is computed relative to its characteristic scale:
- Semi-major axis: |a - aT| / aT
- f, g: |Δf| / max(eT, ε), |Δg| / max(eT, ε)  where eT = √(fT² + gT²)
- h, k: |Δh| / max(sT, ε), |Δk| / max(sT, ε)  where sT = √(hT² + kT²)
with ε = 0.01 floor for near-zero targets.

This criterion ensures **every** targeted element is independently within
tolerance, preventing any single element from hiding behind others in a
pooled error budget (as can happen with `SummedErrorConvergence`).
"""
struct MaxElementConvergence{T<:Number} <: AbstractConvergenceCriterion
    tol::T
end
MaxElementConvergence() = MaxElementConvergence(0.01)

# =============================================================================
# Q-Law Parameters
# =============================================================================

"""
    QLawParameters

Parameters for Q-Law control algorithm.

# Fields
- `Wp`: Penalty weight for minimum periapsis constraint
- `rp_min`: Minimum periapsis radius [km]
- `k_penalty`: Steepness of periapsis penalty exponential (Varga Eq. 9)
- `η_threshold`: Effectivity threshold for coasting (paper default: -0.01)
- `η_smoothness`: Smoothness parameter for activation function (μ in paper)
- `Θrot`: Frame rotation angle about Z-axis [rad] (Varga optimization variable)
- `effectivity_type`: Effectivity computation method (`AbsoluteEffectivity()` or `RelativeEffectivity()`)
- `effectivity_search`: Effectivity search method (`RefinedSearch()` or `GridSearch()`)
- `n_search_points::Int`: Number of points for Q̇ search over true longitude
- `m_scaling::Float64`: Varga Eq. 8 scaling parameter m
- `n_scaling::Float64`: Varga Eq. 8 scaling exponent n
- `r_scaling::Float64`: Varga Eq. 8 scaling root r
- `convergence_criterion`: Convergence criterion (SummedErrorConvergence or VargaConvergence)
"""
struct QLawParameters{
    Tw<:Number,
    Tr<:Number,
    Tt<:Number,
    Ts<:Number,
    Tθ<:Number,
    ET<:AbstractEffectivityType,
    ES<:AbstractEffectivitySearch,
    CC<:AbstractConvergenceCriterion,
}
    Wp::Tw
    rp_min::Tr
    k_penalty::Float64
    η_threshold::Tt
    η_smoothness::Ts
    Θrot::Tθ
    effectivity_type::ET
    effectivity_search::ES
    n_search_points::Int
    m_scaling::Float64
    n_scaling::Float64
    r_scaling::Float64
    convergence_criterion::CC
end

function QLawParameters(;
    Wp::Number = 1.0,
    rp_min::Number = 6578.0,  # Default: LEO altitude
    k_penalty::Float64 = 100.0,  # Penalty steepness for periapsis constraint (Varga Eq. 9)
    η_threshold::Number = -0.01,  # Paper: -0.01 for constant thrust (avoids activation disruption near η≈0)
    η_smoothness::Number = 1e-4,
    Θrot::Number = 0.0,  # Frame rotation angle [rad] (Varga optimization variable)
    effectivity_type::AbstractEffectivityType = AbsoluteEffectivity(),
    effectivity_search::AbstractEffectivitySearch = RefinedSearch(),
    n_search_points::Int = 50,
    m_scaling::Float64 = 1.0,  # Varga Eq. 8: scaling parameter m
    n_scaling::Float64 = 4.0,  # Varga Eq. 8: scaling exponent n
    r_scaling::Float64 = 2.0,  # Varga Eq. 8: scaling root r
    convergence_criterion::AbstractConvergenceCriterion = SummedErrorConvergence(),
)
    return QLawParameters(
        Wp,
        rp_min,
        k_penalty,
        η_threshold,
        η_smoothness,
        Θrot,
        effectivity_type,
        effectivity_search,
        n_search_points,
        m_scaling,
        n_scaling,
        r_scaling,
        convergence_criterion,
    )
end

# =============================================================================
# Q-Law Problem
# =============================================================================

"""
    QLawProblem{OE0,OET,Tm,Tt0,Ttf,Tμ,Tjd,SC,W,P,DM,SM}

Q-Law transfer problem definition.

# Fields
- `oe0::OE0`: Initial orbital elements (ModEq from AstroCoords)
- `oeT::OET`: Target orbital elements (ModEq from AstroCoords)
- `m0::Tm`: Initial mass [kg]
- `tspan::Tuple{Tt0,Ttf}`: Time span (t0, tf) [s]
- `μ::Tμ`: Gravitational parameter [km³/s²]
- `JD0::Tjd`: Julian date at t=0 (for ephemeris calculations)
- `spacecraft::SC`: Spacecraft definition
- `weights::W`: Q-Law weights
- `params::P`: Q-Law parameters
- `dynamics_model::DM`: Force model from AstroForceModels
- `shadow_model::ShadowModelType`: Eclipse model
- `sun_model::SM`: Sun ephemeris model (ThirdBodyModel or nothing)
"""
struct QLawProblem{
    OE0<:ModEq,
    OET<:ModEq,
    Tm<:Number,
    Tt0<:Number,
    Ttf<:Number,
    Tμ<:Number,
    Tjd<:Number,
    SC<:AbstractQLawSpacecraft,
    W<:QLawWeights,
    P<:QLawParameters,
    DM<:AbstractDynamicsModel,
    SM,
}
    oe0::OE0
    oeT::OET
    m0::Tm
    tspan::Tuple{Tt0,Ttf}
    μ::Tμ
    JD0::Tjd
    spacecraft::SC
    weights::W
    params::P
    dynamics_model::DM
    shadow_model::ShadowModelType
    sun_model::SM
end

"""
    qlaw_problem(oe0, oeT, tspan, μ, spacecraft; kwargs...)

Create a Q-Law problem.

# Arguments
- `oe0`: Initial orbital elements (ModEq or convertible)
- `oeT`: Target orbital elements (ModEq or convertible)
- `tspan`: Time span (t0, tf) [s]
- `μ`: Gravitational parameter [km³/s²]
- `spacecraft`: Spacecraft definition

# Keyword Arguments
- `weights::QLawWeights`: Q-Law weights (default: all ones)
- `qlaw_params::QLawParameters`: Q-Law parameters (default: QLawParameters())
- `dynamics_model`: Force model (default: Keplerian only)
- `shadow_model_type`: Eclipse model (default: Conical())
- `sun_model`: Sun ephemeris model for eclipse computation (default: nothing, uses simple model)
- `JD0::Number`: Julian date at t=0 for ephemeris calculations (default: J2000.0 = 2451545.0)
"""
function qlaw_problem(
    oe0::ModEq,
    oeT::ModEq,
    tspan::Tuple,
    μ::Number,
    spacecraft::AbstractQLawSpacecraft;
    weights::QLawWeights = QLawWeights(),
    qlaw_params::QLawParameters = QLawParameters(),
    dynamics_model::AbstractDynamicsModel = CentralBodyDynamicsModel(
        KeplerianGravityAstroModel(; μ = μ),
        (),
    ),
    shadow_model_type::ShadowModelType = Conical(),
    sun_model::Union{ThirdBodyModel,Nothing} = nothing,
    JD0::Number = 2451545.0,
)
    return QLawProblem(
        oe0,
        oeT,
        mass(spacecraft),
        tspan,
        μ,
        JD0,
        spacecraft,
        weights,
        qlaw_params,
        dynamics_model,
        shadow_model_type,
        sun_model,
    )
end

# Convenience constructor for Keplerian inputs
function qlaw_problem(
    koe0::Keplerian,
    koeT::Keplerian,
    tspan::Tuple,
    μ::Number,
    spacecraft::AbstractQLawSpacecraft;
    JD0::Number = 2451545.0,
    kwargs...,
)
    oe0 = ModEq(koe0, μ)
    oeT = ModEq(koeT, μ)
    return qlaw_problem(oe0, oeT, tspan, μ, spacecraft; JD0 = JD0, kwargs...)
end

# =============================================================================
# Q-Law Solution
# =============================================================================

"""
    QLawSolution{P,S,Tdv,Tm,OE,Tt}

Solution to a Q-Law transfer problem.

# Fields
- `problem::P`: Original problem
- `trajectory::S`: ODE solution from DifferentialEquations
- `converged::Bool`: Whether the transfer converged to target
- `Δv_total::Tdv`: Total ΔV used [km/s]
- `final_mass::Tm`: Final spacecraft mass [kg]
- `final_oe::OE`: Final orbital elements
- `elapsed_time::Tt`: Transfer time [s]
"""
struct QLawSolution{P<:QLawProblem,S,Tdv<:Number,Tm<:Number,OE<:ModEq,Tt<:Number}
    problem::P
    trajectory::S
    converged::Bool
    Δv_total::Tdv
    final_mass::Tm
    final_oe::OE
    elapsed_time::Tt
end

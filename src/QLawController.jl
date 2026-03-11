module QLawController

using AstroCoords: AstroCoords, ModEq, Keplerian, Cartesian, params
using AstroForceModels:
    AstroForceModels,
    shadow_model,
    Conical,
    Cylindrical,
    NoShadow,
    ShadowModelType,
    build_dynamics_model,
    acceleration,
    KeplerianGravityAstroModel,
    CentralBodyDynamicsModel,
    AbstractDynamicsModel,
    ThirdBodyModel,
    SunBody,
    Position
using AstroPropagators: AstroPropagators, inertial_to_RTN, modified_equinoctial_gve
using ComponentArrays
using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEqCore
using OrdinaryDiffEqVerner: Vern9
using SciMLBase
using StaticArrays

# =============================================================================
# Core Types
# =============================================================================
include("types.jl")

# =============================================================================
# Q-Law Core (these are Q-Law specific, not in other packages)
# =============================================================================
include("qlaw_core.jl")

# =============================================================================
# Dynamics and Propagation
# =============================================================================
include("dynamics.jl")

# =============================================================================
# API
# =============================================================================
include("api.jl")

end

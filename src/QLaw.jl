module QLawController

using AstroCoords: AstroCoords, ModEq, Keplerian, Cartesian, params
using AstroForceModels:
    AstroForceModels,
    shadow_model,
    Conical,
    Cylindrical,
    No_Shadow,
    ShadowModelType,
    build_dynamics_model,
    acceleration,
    KeplerianGravityAstroModel,
    CentralBodyDynamicsModel,
    AbstractDynamicsModel,
    ThirdBodyModel,
    SunBody,
    Position
using AstroPropagators: AstroPropagators, RTN_frame
using ComponentArrays
using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEqCore
using OrdinaryDiffEqAdamsBashforthMoulton: VCABM
using SciMLBase
using Optim: Optim, optimize, Brent
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

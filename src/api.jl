# =============================================================================
# SciML-Style API
# =============================================================================

export solve, remake
export compute_delta_v
export orbital_elements, mass_history, sma_history, ecc_history, inc_history

"""
    solve(problem::QLawProblem; kwargs...)

Solve a Q-Law transfer problem.

# Keyword Arguments
- `abstol::Float64=1e-10`: Absolute tolerance for ODE solver
- `reltol::Float64=1e-10`: Relative tolerance for ODE solver
- `ODE_solver`: ODE solver algorithm (default: VCABM())
- `saveat`: Times to save solution at (default: automatic)
- `callback`: DiffEq callback for events
- `convergence_tol`: (deprecated) Override convergence tolerance; prefer setting
  `convergence_criterion` in `QLawParameters` instead.

# Returns
`QLawSolution` containing the trajectory and solution metadata.
"""
function SciMLBase.solve(
    problem::QLawProblem{OE0,OET,Tm,Tt0,Ttf,Tμ,Tjd,SC,W,P,DM,SM};
    abstol::Float64 = 1e-10,
    reltol::Float64 = 1e-10,
    ODE_solver::OrdinaryDiffEqCore.OrdinaryDiffEqAlgorithm = VCABM(),
    saveat = nothing,
    callback = nothing,
    convergence_tol = nothing,
    kwargs...,
) where {OE0,OET,Tm,Tt0,Ttf,Tμ,Tjd,SC,W,P,DM,SM}

    # Promote type to account for ForwardDiff.Dual numbers that may enter
    # through QLawWeights or QLawParameters during gradient-based optimization
    T = promote_type(
        Float64,
        Tm,
        Tt0,
        Ttf,
        Tμ,
        typeof(problem.weights.Wa),
        typeof(problem.params.Wp),
        typeof(problem.params.rp_min),
        typeof(problem.params.η_threshold),
        typeof(problem.params.η_smoothness),
        typeof(problem.params.Θrot),
    )

    # Determine effective convergence criterion
    # convergence_tol kwarg overrides params for backward compatibility
    if convergence_tol !== nothing
        effective_criterion = SummedErrorConvergence(convergence_tol)
    else
        effective_criterion = problem.params.convergence_criterion
    end

    # Initial state: [p, f, g, h, k, L, m]
    oe0 = problem.oe0
    u0 = SVector{7,T}(oe0.p, oe0.f, oe0.g, oe0.h, oe0.k, oe0.L, problem.m0)


    # Parameters
    ps = ComponentArray(μ = problem.μ, JD = problem.JD0)

    # Create termination callback for convergence
    function convergence_condition(u, t, integrator)
        oe_current = ModEq{T}(u[1], u[2], u[3], u[4], u[5], u[6])
        m_current = u[7]
        r_current = compute_radius(oe_current)
        F_max_current = max_thrust_acceleration(problem.spacecraft, m_current, r_current)
        return check_convergence(
            oe_current,
            problem.oeT,
            problem.weights,
            problem.μ,
            F_max_current,
            problem.params,
            effective_criterion,
        )
    end

    convergence_affect!(integrator) = terminate!(integrator)
    convergence_cb = DiscreteCallback(convergence_condition, convergence_affect!)

    # Combine callbacks
    if callback === nothing
        full_callback = convergence_cb
    else
        full_callback = CallbackSet(convergence_cb, callback)
    end

    # Create ODE function (non-mutating for StaticArrays compatibility)
    eom(u, p, t) = qlaw_eom(u, p, t, problem)

    # Create ODE problem
    ode_prob = ODEProblem(eom, u0, problem.tspan, ps)

    # Solve
    if saveat === nothing
        sol = OrdinaryDiffEqCore.solve(
            ode_prob,
            ODE_solver;
            abstol = abstol,
            reltol = reltol,
            callback = full_callback,
            kwargs...,
        )
    else
        sol = OrdinaryDiffEqCore.solve(
            ode_prob,
            ODE_solver;
            abstol = abstol,
            reltol = reltol,
            saveat = saveat,
            callback = full_callback,
            kwargs...,
        )
    end

    # Extract final state
    u_final = sol.u[end]
    final_oe =
        ModEq{T}(u_final[1], u_final[2], u_final[3], u_final[4], u_final[5], u_final[6])
    final_mass = u_final[7]

    # Check convergence using same criterion
    r_final = compute_radius(final_oe)
    F_max_final = max_thrust_acceleration(problem.spacecraft, final_mass, r_final)
    converged = check_convergence(
        final_oe,
        problem.oeT,
        problem.weights,
        problem.μ,
        F_max_final,
        problem.params,
        effective_criterion,
    )

    # Compute total ΔV
    Δv_total = compute_delta_v(problem.m0, final_mass, problem.spacecraft)

    # Elapsed time
    elapsed_time = sol.t[end] - sol.t[1]

    return QLawSolution(
        problem,
        sol,
        converged,
        Δv_total,
        final_mass,
        final_oe,
        elapsed_time,
    )
end

"""
Compute total ΔV from mass change using Tsiolkovsky equation.
"""
function compute_delta_v(m0::Number, mf::Number, spacecraft::AbstractQLawSpacecraft)
    vex = exhaust_velocity(spacecraft)  # km/s
    if mf > zero(mf) && m0 > mf
        return vex * log(m0 / mf)
    else
        return zero(typeof(m0))
    end
end


# Note: check_convergence has been consolidated into qlaw_core.jl with
# multiple dispatch on convergence criterion type.

"""
    remake(problem::QLawProblem; kwargs...)

Create a new QLawProblem with modified parameters.

# Keyword Arguments
Any field of QLawProblem can be modified:
- `oe0`: New initial orbital elements
- `oeT`: New target orbital elements
- `m0`: New initial mass
- `tspan`: New time span
- `μ`: New gravitational parameter
- `JD0`: New initial Julian date
- `spacecraft`: New spacecraft
- `weights`: New Q-Law weights
- `qlaw_params`: New Q-Law parameters
- `dynamics_model`: New dynamics model
- `shadow_model_type`: New shadow model type
- `sun_model`: New sun model
"""
function SciMLBase.remake(
    problem::QLawProblem;
    oe0::Union{ModEq,Nothing} = nothing,
    oeT::Union{ModEq,Nothing} = nothing,
    m0::Union{Number,Nothing} = nothing,
    tspan::Union{Tuple,Nothing} = nothing,
    μ::Union{Number,Nothing} = nothing,
    JD0::Union{Number,Nothing} = nothing,
    spacecraft::Union{AbstractQLawSpacecraft,Nothing} = nothing,
    weights::Union{QLawWeights,Nothing} = nothing,
    qlaw_params::Union{QLawParameters,Nothing} = nothing,
    dynamics_model::Union{AbstractDynamicsModel,Nothing} = nothing,
    shadow_model_type::Union{ShadowModelType,Nothing} = nothing,
    sun_model = nothing,
)

    new_oe0 = oe0 === nothing ? problem.oe0 : oe0
    new_oeT = oeT === nothing ? problem.oeT : oeT
    new_m0 = m0 === nothing ? problem.m0 : m0
    new_tspan = tspan === nothing ? problem.tspan : tspan
    new_μ = μ === nothing ? problem.μ : μ
    new_JD0 = JD0 === nothing ? problem.JD0 : JD0
    new_spacecraft = spacecraft === nothing ? problem.spacecraft : spacecraft
    new_weights = weights === nothing ? problem.weights : weights
    new_params = qlaw_params === nothing ? problem.params : qlaw_params
    new_dynamics = dynamics_model === nothing ? problem.dynamics_model : dynamics_model
    new_shadow = shadow_model_type === nothing ? problem.shadow_model : shadow_model_type
    new_sun = sun_model === nothing ? problem.sun_model : sun_model

    return QLawProblem(
        new_oe0,
        new_oeT,
        new_m0,
        new_tspan,
        new_μ,
        new_JD0,
        new_spacecraft,
        new_weights,
        new_params,
        new_dynamics,
        new_shadow,
        new_sun,
    )
end

# =============================================================================
# Solution Accessors
# =============================================================================

"""Get the trajectory time vector."""
Base.time(sol::QLawSolution) = sol.trajectory.t

"""Get orbital element history as vector of ModEq."""
function orbital_elements(sol::QLawSolution)
    T = eltype(sol.trajectory.u[1])
    return [ModEq{T}(u[1], u[2], u[3], u[4], u[5], u[6]) for u in sol.trajectory.u]
end

"""Get mass history."""
function mass_history(sol::QLawSolution)
    return [u[7] for u in sol.trajectory.u]
end

"""Get semi-major axis history [km]."""
function sma_history(sol::QLawSolution)
    T = eltype(sol.trajectory.u[1])
    return [get_sma(ModEq{T}(u[1], u[2], u[3], u[4], u[5], u[6])) for u in sol.trajectory.u]
end

"""Get eccentricity history."""
function ecc_history(sol::QLawSolution)
    return [sqrt(u[2]^2 + u[3]^2) for u in sol.trajectory.u]
end

"""Get inclination history [rad]."""
function inc_history(sol::QLawSolution)
    return [2 * atan(sqrt(u[4]^2 + u[5]^2)) for u in sol.trajectory.u]
end

"""Print solution summary."""
function Base.show(io::IO, sol::QLawSolution)
    println(io, "QLawSolution:")
    println(io, "  Converged: ", sol.converged)
    println(io, "  Transfer time: ", sol.elapsed_time / 86400.0, " days")
    println(io, "  Total ΔV: ", sol.Δv_total, " km/s")
    println(io, "  Final mass: ", sol.final_mass, " kg")
    println(io, "  Final semi-major axis: ", get_sma(sol.final_oe), " km")
    println(io, "  Target semi-major axis: ", get_sma(sol.problem.oeT), " km")
end

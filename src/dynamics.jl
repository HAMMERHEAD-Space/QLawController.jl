# =============================================================================
# Q-Law Dynamics
# 
# Integrates Q-Law control with existing force models from AstroForceModels
# =============================================================================

"""
    compute_sunlight_fraction(oe::ModEq, μ::Number, sun_pos::AbstractVector,
                               shadow_model_type::ShadowModelType;
                               R_Sun::Number=696000.0, R_Earth::Number=6378.137)

Compute the sunlight fraction γ using AstroForceModels shadow_model.

# Arguments
- `oe`: Current orbital elements (ModEq)
- `μ`: Gravitational parameter [km³/s²]
- `sun_pos`: Sun position vector in inertial frame [km]
- `shadow_model_type`: Type of shadow model (Conical, Cylindrical, No_Shadow)
"""
function compute_sunlight_fraction(
    oe::ModEq{T},
    μ::Number,
    sun_pos::AbstractVector,
    shadow_model_type::ShadowModelType;
    R_Sun::Number = 696000.0,
    R_Earth::Number = 6378.137,
) where {T}
    # Convert to Cartesian to get spacecraft position
    cart = Cartesian(oe, μ)
    sat_pos = SVector{3,T}(cart[1], cart[2], cart[3])

    # Use AstroForceModels shadow_model
    γ = shadow_model(
        sat_pos,
        sun_pos,
        shadow_model_type;
        R_Sun = T(R_Sun),
        R_Occulting = T(R_Earth),
    )

    return T(γ)
end

"""
    qlaw_thrust_acceleration(oe::ModEq, oeT::ModEq, m::Number,
                             spacecraft::AbstractQLawSpacecraft,
                             weights::QLawWeights, params::QLawParameters,
                             μ::Number, γ::Number=1.0)

Compute the thrust acceleration vector in RTN frame using Q-Law guidance.
Uses combined thrust direction + effectivity computation to avoid redundant work.

# Returns
- `a_thrust_rtn`: Thrust acceleration in RTN frame [km/s²]
- `throttle`: Throttle value (0-1)
- `α`: In-plane thrust angle [rad]
- `β`: Out-of-plane thrust angle [rad]
"""
function qlaw_thrust_acceleration(
    oe::ModEq{T},
    oeT::ModEq,
    m::Number,
    spacecraft::AbstractQLawSpacecraft,
    weights::QLawWeights,
    params::QLawParameters,
    μ::Number,
    γ::Number = one(T),
) where {T}

    # Get orbital radius for SEP thrust scaling
    r = compute_radius(oe)

    # Thrust acceleration at current orbital position (used for actual thrust magnitude)
    F_max = max_thrust_acceleration(spacecraft, m, r)

    # Orbit-maximum thrust acceleration (at periapsis) for Q-Law normalization.
    # For constant-thrust spacecraft this equals F_max. For SEP (thrust ∝ 1/r²),
    # using the orbit-max ensures that max_rates normalization in the Q function
    # is consistent across the orbit, preventing artificial Q variations from
    # thrust scaling on eccentric orbits.
    F_max_orbit = max_orbit_thrust_acceleration(spacecraft, m, oe)

    # Compute thrust direction AND effectivity together (avoids redundant Qdot_n)
    # Uses orbit-max F_max for consistent Q-Law normalization (Varga Eqs. 14-18)
    α, β, _, η = compute_thrust_and_effectivity(oe, oeT, weights, μ, F_max_orbit, params)

    # Activation (smooth throttle based on effectivity)
    activation = effectivity_activation(η, params.η_threshold, params.η_smoothness)

    # Throttle combines effectivity activation and sunlight fraction
    throttle = activation * T(γ)

    # Thrust direction in RTN
    u_rtn = thrust_direction_to_rtn(α, β)

    # Actual thrust acceleration uses current-position F_max
    a_thrust_rtn = throttle * F_max * u_rtn

    return (a_thrust_rtn, throttle, α, β)
end

# =============================================================================
# Equations of Motion for Q-Law Propagation
# =============================================================================

"""
    qlaw_eom!(du, u, p, t)

Equations of motion for Q-Law propagation.

State vector u = [p, f, g, h, k, L, m] (ModEq elements + mass)

IMPORTANT: The Q-Law algorithm uses semi-major axis `a` internally (as per the paper:
"Semi-major axis is used for this application since its implementation by Varga and Perez 
showed it causes better controller performance"). 

However, our state uses semi-latus rectum `p` (from ModEq). The GVE matrix used here
is computed via `equinoctial_gve_partials()` (shared with Q-Law core) using the a-based
formulation, then da/dt is converted to dp/dt for propagation consistency.

Parameters p should be a ComponentArray with:
- `μ`: Gravitational parameter
- `JD`: Julian date
- Plus any additional parameters needed by force models
"""
function qlaw_eom(u::AbstractVector, ps::ComponentVector, t::Number, problem::QLawProblem)

    # Extract state - let element type be inferred for AD compatibility
    oe = ModEq(u[1], u[2], u[3], u[4], u[5], u[6])
    m = u[7]

    μ = ps.μ

    # Convert to Cartesian for force model evaluation (Q-law frame)
    cart = Cartesian(oe, μ)
    cart_vec = SVector(cart[1], cart[2], cart[3], cart[4], cart[5], cart[6])

    # Rotate spacecraft state from Q-law frame to inertial frame for force models.
    # Varga's Θrot rotates the system about Z; two-body and J2 are invariant,
    # but third-body positions (Sun, Moon) are fixed in the inertial frame,
    # so force models must see the correct inertial-frame spacecraft position.
    Θrot = problem.params.Θrot
    pos_qlaw = SVector{3}(cart_vec[1], cart_vec[2], cart_vec[3])
    vel_qlaw = SVector{3}(cart_vec[4], cart_vec[5], cart_vec[6])
    pos_inertial = apply_frame_rotation(pos_qlaw, -Θrot)  # Q-law → inertial
    vel_inertial = apply_frame_rotation(vel_qlaw, -Θrot)
    cart_vec_inertial = SVector(
        pos_inertial[1],
        pos_inertial[2],
        pos_inertial[3],
        vel_inertial[1],
        vel_inertial[2],
        vel_inertial[3],
    )

    # Get perturbation accelerations in inertial frame from AstroForceModels
    # (excluding central body gravity, which is handled by GVE)
    acc_inertial =
        build_dynamics_model(cart_vec_inertial, ps, t, problem.dynamics_model) -
        acceleration(cart_vec_inertial, ps, t, KeplerianGravityAstroModel(; μ = μ))

    # Rotate perturbation acceleration from inertial back to Q-law frame
    acc_qlaw_frame = apply_frame_rotation(acc_inertial, Θrot)  # inertial → Q-law

    # Transform to RTN frame (defined by Q-law frame orbit)
    R_rtn = RTN_frame(cart_vec)
    acc_rtn = R_rtn * acc_qlaw_frame

    # Get sun position for shadow calculation (with Varga Θrot frame rotation)
    sun_pos = get_sun_position(ps, t, problem.sun_model; Θrot = problem.params.Θrot)

    # Compute sunlight fraction
    γ = compute_sunlight_fraction(oe, μ, sun_pos, problem.shadow_model)

    # Compute Q-Law thrust acceleration
    a_thrust_rtn, throttle, _, _ = qlaw_thrust_acceleration(
        oe,
        problem.oeT,
        m,
        problem.spacecraft,
        problem.weights,
        problem.params,
        μ,
        γ,
    )

    # Total acceleration in RTN
    acc_total_rtn = acc_rtn + a_thrust_rtn

    # =========================================================================
    # GVE using equinoctial_gve_partials (shared with Q-Law core)
    # This ensures dynamics and controller use identical GVE formulations.
    # =========================================================================

    p = oe.p
    f, g, h, k, L = oe.f, oe.g, oe.h, oe.k, oe.L

    # Convert to semi-major axis
    e_sq = f^2 + g^2
    a = p / (1 - e_sq)

    # Compute GVE matrix using the shared function from qlaw_core.jl
    A = equinoctial_gve_partials(oe, μ)

    # Compute element rates: oe_dot = A * F_rtn (6-vector)
    oe_dot = A * acc_total_rtn

    da_dt = oe_dot[1]
    df_dt = oe_dot[2]
    dg_dt = oe_dot[3]
    dh_dt = oe_dot[4]
    dk_dt = oe_dot[5]
    dL_dt_pert = oe_dot[6]

    # Convert da/dt to dp/dt for state propagation (since state uses p, not a)
    # p = a * (1 - f² - g²)
    # dp/dt = da/dt * (1 - f² - g²) + a * (-2f*df/dt - 2g*dg/dt)
    #       = da/dt * (1 - e²) - 2a * (f*df/dt + g*dg/dt)
    dp_dt = da_dt * (1 - e_sq) - 2 * a * (f * df_dt + g * dg_dt)

    # Keplerian true longitude rate: dL/dt = h/r² = q²√(μp)/p²
    # where h = √(μp) is the specific angular momentum and r = p/q
    q = compute_q(f, g, L)
    dL_keplerian = q^2 * sqrt(μ * p) / p^2

    # Orbital radius for mass rate calculation
    r = p / q

    # Mass rate (AD-compatible: no branching, smooth throttle ensures dm→0 when not thrusting)
    T_N = max_thrust(problem.spacecraft, r) * throttle
    vex = exhaust_velocity(problem.spacecraft) * 1000.0  # Convert to m/s
    dm_dt = -T_N / vex  # kg/s

    return SVector(
        dp_dt,                      # dp/dt (converted from da/dt)
        df_dt,                      # df/dt
        dg_dt,                      # dg/dt
        dh_dt,                      # dh/dt
        dk_dt,                      # dk/dt
        dL_dt_pert + dL_keplerian,  # dL/dt (perturbation + Keplerian)
        dm_dt,
    )
end

function qlaw_eom!(
    du::AbstractVector,
    u::AbstractVector,
    ps::ComponentVector,
    t::Number,
    problem::QLawProblem,
)
    du .= qlaw_eom(u, ps, t, problem)
    return nothing
end

"""
    apply_frame_rotation(pos::SVector{3}, Θrot::Number)

Rotate a position vector about the Z-axis by -Θrot (Varga frame rotation).

From Varga: "The transformation parameter Θrot rotates the system around
the north-south axis." The dynamics are invariant under this rotation, but
constraints (e.g. Sun vector for eclipse) are modified.
"""
function apply_frame_rotation(pos::SVector{3,T}, Θrot::Number) where {T}
    cθ = cos(Θrot)
    sθ = sin(Θrot)
    x = pos[1] * cθ + pos[2] * sθ
    y = -pos[1] * sθ + pos[2] * cθ
    z = pos[3]
    return SVector{3}(x, y, z)
end

"""
    get_sun_position(ps::ComponentVector, t::Number, sun_model=nothing; Θrot=0.0)

Get Sun position using AstroForceModels ThirdBodyModel or fallback to simple model.
Applies Varga Θrot frame rotation if non-zero.

The fallback model computes the Sun position in the ecliptic plane and then
rotates to the J2000 equatorial frame using the mean obliquity of the ecliptic
(ε ≈ 23.4393°).

# Arguments
- `ps`: Parameters (should contain JD0)
- `t`: Time since epoch [s]
- `sun_model`: Optional ThirdBodyModel for accurate ephemeris
- `Θrot`: Frame rotation angle [rad] (Varga optimization variable)
"""
function get_sun_position(
    ps::ComponentVector,
    t::Number,
    sun_model::Union{ThirdBodyModel,Nothing} = nothing;
    Θrot::Number = 0.0,
)
    # Determine raw sun position
    if haskey(ps, :sun_pos)
        sun_pos = SVector{3}(ps.sun_pos...)
    elseif sun_model !== nothing
        # Compute Julian date
        JD = haskey(ps, :JD) ? ps.JD + t / 86400.0 : 2451545.0 + t / 86400.0
        # ThirdBodyModel returns position in METERS, we need KILOMETERS
        sun_pos_m = sun_model(JD, Position())
        sun_pos =
            SVector{3}(sun_pos_m[1] / 1000.0, sun_pos_m[2] / 1000.0, sun_pos_m[3] / 1000.0)
    else
        # Fallback: simple circular orbit approximation for Sun (Earth-centered)
        # Computes ecliptic longitude, then rotates to J2000 equatorial frame
        JD = haskey(ps, :JD) ? ps.JD + t / 86400.0 : 2451545.0 + t / 86400.0
        AU = 1.495978707e8  # km
        d = JD - 2451545.0
        M = 357.529 + 0.98560028 * d
        M_rad = deg2rad(M)
        λ = M_rad + π  # ecliptic longitude (Sun seen from Earth)

        # Sun position in ecliptic coordinates
        x_ecl = AU * cos(λ)
        y_ecl = AU * sin(λ)

        # Rotate from ecliptic to J2000 equatorial frame
        # ε = mean obliquity of the ecliptic at J2000
        ε = deg2rad(23.4393)
        cε = cos(ε)
        sε = sin(ε)

        sun_pos = SVector{3}(x_ecl, y_ecl * cε, y_ecl * sε)
    end

    # Apply frame rotation (Varga Θrot parameter)
    return apply_frame_rotation(sun_pos, Θrot)
end


# Note: check_convergence has been consolidated into qlaw_core.jl with
# multiple dispatch on convergence criterion type.

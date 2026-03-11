# =============================================================================
# Q-Law Dynamics
# 
# Integrates Q-Law control with existing force models from AstroForceModels
# =============================================================================

export compute_sunlight_fraction, qlaw_thrust_acceleration
export qlaw_eom, qlaw_eom!
export apply_frame_rotation, get_sun_position

"""
    compute_sunlight_fraction(oe::ModEq, μ::Number, sun_pos::AbstractVector,
                               shadow_model_type::ShadowModelType;
                               R_Sun::Number=696000.0, R_Earth::Number=6378.137)

Compute the sunlight fraction γ using AstroForceModels shadow_model.

# Arguments
- `oe`: Current orbital elements (ModEq)
- `μ`: Gravitational parameter [km³/s²]
- `sun_pos`: Sun position vector in inertial frame [km]
- `shadow_model_type`: Type of shadow model (Conical, Cylindrical, NoShadow)
"""
function compute_sunlight_fraction(
    oe::ModEq{T},
    μ::Number,
    sun_pos::AbstractVector,
    shadow_model_type::ShadowModelType;
    R_Sun::Number=696000.0,
    R_Earth::Number=6378.137,
) where {T}
    cart = Cartesian(oe, μ)
    sat_pos = SVector{3,T}(cart[1], cart[2], cart[3])
    return compute_sunlight_fraction(
        sat_pos, sun_pos, shadow_model_type; R_Sun=R_Sun, R_Earth=R_Earth
    )
end

function compute_sunlight_fraction(
    sat_pos::SVector{3,T},
    sun_pos::AbstractVector,
    shadow_model_type::ShadowModelType;
    R_Sun::Number=696000.0,
    R_Earth::Number=6378.137,
) where {T}
    γ = shadow_model(
        sat_pos, sun_pos, shadow_model_type; R_Sun=T(R_Sun), R_Occulting=T(R_Earth)
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
    γ::Number=one(T),
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
    qlaw_eom(u, ps, t, problem)

Equations of motion for Q-Law propagation.

State vector u = [p, f, g, h, k, L, m] (ModEq elements + mass).

Uses the p-based Modified Equinoctial GVE from AstroPropagators for propagation
(giving dp/dt directly), while the Q-Law controller internally uses the a-based
wrapper `equinoctial_gve_partials` for thrust-direction optimization.

Parameters ps should be a ComponentArray with:
- `μ`: Gravitational parameter
- `JD`: Julian date
- Plus any additional parameters needed by force models
"""
function qlaw_eom(u::AbstractVector, ps::ComponentVector, t::Number, problem::QLawProblem)
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
    pos_inertial = apply_frame_rotation(pos_qlaw, -Θrot)
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
    acc_inertial =
        build_dynamics_model(cart_vec_inertial, ps, t, problem.dynamics_model) -
        acceleration(cart_vec_inertial, ps, t, KeplerianGravityAstroModel(; μ=μ))

    # Rotate perturbation acceleration from inertial back to Q-law frame
    acc_qlaw_frame = apply_frame_rotation(acc_inertial, Θrot)

    # Transform to RTN frame using AstroPropagators utility
    acc_rtn = inertial_to_RTN(acc_qlaw_frame, cart_vec)

    # Get sun position for shadow calculation (with Varga Θrot frame rotation)
    sun_pos = get_sun_position(ps, t, problem.sun_model; Θrot=problem.params.Θrot)

    # Compute sunlight fraction (reuse pos_qlaw to avoid redundant Cartesian conversion)
    γ = compute_sunlight_fraction(pos_qlaw, sun_pos, problem.shadow_model)

    # Compute Q-Law thrust acceleration
    a_thrust_rtn, throttle, _, _ = qlaw_thrust_acceleration(
        oe, problem.oeT, m, problem.spacecraft, problem.weights, problem.params, μ, γ
    )

    # Total acceleration in RTN
    acc_total_rtn = acc_rtn + a_thrust_rtn

    # =========================================================================
    # p-based GVE from AstroPropagators — gives dp/dt directly, no conversion
    # =========================================================================

    p_el = oe.p
    f, g, h, k, L = oe.f, oe.g, oe.h, oe.k, oe.L

    A = AstroPropagators.modified_equinoctial_gve(p_el, f, g, h, k, L, μ)
    oe_dot = A * acc_total_rtn

    # Keplerian true longitude rate: dL/dt = q²√(μp)/p²
    q = compute_q(f, g, L)
    dL_keplerian = q^2 * sqrt(μ * p_el) / p_el^2

    # Orbital radius for mass rate calculation
    r = p_el / q

    # Mass rate (AD-compatible: no branching, smooth throttle ensures dm→0 when not thrusting)
    T_N = max_thrust(problem.spacecraft, r) * throttle
    vex = exhaust_velocity(problem.spacecraft) * 1000.0
    dm_dt = -T_N / vex

    return SVector(
        oe_dot[1],                      # dp/dt
        oe_dot[2],                      # df/dt
        oe_dot[3],                      # dg/dt
        oe_dot[4],                      # dh/dt
        oe_dot[5],                      # dk/dt
        oe_dot[6] + dL_keplerian,       # dL/dt (perturbation + Keplerian)
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
    get_sun_position(ps::ComponentVector, t::Number, ::Nothing; Θrot=0.0)

Fallback simple circular orbit approximation for Sun (Earth-centered).
Computes ecliptic longitude, then rotates to J2000 equatorial frame
using the mean obliquity of the ecliptic (ε ≈ 23.4393°).
"""
function get_sun_position(ps::ComponentVector, t::Number, ::Nothing; Θrot::Number=0.0)
    JD = ps.JD + t / 86400.0
    AU = 1.495978707e8  # km
    d = JD - 2451545.0
    M = 357.529 + 0.98560028 * d
    M_rad = deg2rad(M)
    λ = M_rad + π

    x_ecl = AU * cos(λ)
    y_ecl = AU * sin(λ)

    ε = deg2rad(23.4393)
    cε = cos(ε)
    sε = sin(ε)

    sun_pos = SVector{3}(x_ecl, y_ecl * cε, y_ecl * sε)
    return apply_frame_rotation(sun_pos, Θrot)
end

"""
    get_sun_position(ps::ComponentVector, t::Number, sun_model::ThirdBodyModel; Θrot=0.0)

Get Sun position using AstroForceModels ThirdBodyModel ephemeris.
Applies Varga Θrot frame rotation if non-zero.
"""
function get_sun_position(
    ps::ComponentVector, t::Number, sun_model::ThirdBodyModel; Θrot::Number=0.0
)
    JD = ps.JD + t / 86400.0
    sun_pos_m = sun_model(JD, Position())
    sun_pos = SVector{3}(
        sun_pos_m[1] / 1000.0, sun_pos_m[2] / 1000.0, sun_pos_m[3] / 1000.0
    )
    return apply_frame_rotation(sun_pos, Θrot)
end

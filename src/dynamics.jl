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
function compute_sunlight_fraction(oe::ModEq{T}, μ::Number, sun_pos::AbstractVector,
                                    shadow_model_type::ShadowModelType;
                                    R_Sun::Number=696000.0, R_Earth::Number=6378.137) where T
    # Convert to Cartesian to get spacecraft position
    cart = Cartesian(oe, μ)
    sat_pos = SVector{3,T}(cart[1], cart[2], cart[3])
    
    # Use AstroForceModels shadow_model
    γ = shadow_model(sat_pos, sun_pos, shadow_model_type; 
                     R_Sun=T(R_Sun), R_Occulting=T(R_Earth))
    
    return T(γ)
end

"""
    qlaw_thrust_acceleration(oe::ModEq, oeT::ModEq, m::Number,
                             spacecraft::AbstractQLawSpacecraft,
                             weights::QLawWeights, params::QLawParameters,
                             μ::Number, γ::Number=1.0)

Compute the thrust acceleration vector in RTN frame using Q-Law guidance.

# Returns
- `a_thrust_rtn`: Thrust acceleration in RTN frame [km/s²]
- `throttle`: Throttle value (0-1)
- `α`: In-plane thrust angle [rad]
- `β`: Out-of-plane thrust angle [rad]
"""
function qlaw_thrust_acceleration(oe::ModEq{T}, oeT::ModEq, m::Number,
                                   spacecraft::AbstractQLawSpacecraft,
                                   weights::QLawWeights, params::QLawParameters,
                                   μ::Number, γ::Number=one(T)) where T
    
    # Get orbital radius for SEP thrust scaling
    r = compute_radius(oe)
    
    # Maximum thrust acceleration
    F_max = max_thrust_acceleration(spacecraft, m, r)
    
    # Compute optimal thrust direction
    α, β, _ = compute_thrust_direction(oe, oeT, weights, μ, F_max, params)
    
    # Compute effectivity
    η, _, _, _ = compute_effectivity(oe, oeT, weights, μ, F_max, params)
    
    # Activation (smooth throttle based on effectivity)
    activation = effectivity_activation(η, params.η_threshold, params.η_smoothness)
    
    # Throttle combines effectivity activation and sunlight fraction
    throttle = activation * T(γ)
    
    # Thrust direction in RTN
    u_rtn = thrust_direction_to_rtn(α, β)
    
    # Thrust acceleration
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
must match what Q-Law expects. We use the a-based GVE from Eq. (3) of the paper,
then convert the da/dt to dp/dt for propagation consistency.

Parameters p should be a ComponentArray with:
- `μ`: Gravitational parameter
- `JD`: Julian date
- Plus any additional parameters needed by force models
"""
function qlaw_eom(u::AbstractVector, ps::ComponentVector, t::Number,
                  problem::QLawProblem)
    
    # Extract state - let element type be inferred for AD compatibility
    oe = ModEq(u[1], u[2], u[3], u[4], u[5], u[6])
    m = u[7]
    
    μ = ps.μ
    
    # Convert to Cartesian for force model evaluation
    cart = Cartesian(oe, μ)
    cart_vec = SVector(cart[1], cart[2], cart[3], cart[4], cart[5], cart[6])
    
    # Get perturbation accelerations in inertial frame from AstroForceModels
    # (excluding central body gravity, which is handled by GVE)
    acc_inertial = build_dynamics_model(cart_vec, ps, t, problem.dynamics_model) -
                   acceleration(cart_vec, ps, t, KeplerianGravityAstroModel(; μ=μ))
    
    # Transform to RTN frame
    R_rtn = RTN_frame(cart_vec)
    acc_rtn = R_rtn * acc_inertial
    
    # Get sun position for shadow calculation (with Varga Θrot frame rotation)
    sun_pos = get_sun_position(ps, t, problem.sun_model; Θrot=problem.params.Θrot)
    
    # Compute sunlight fraction
    γ = compute_sunlight_fraction(oe, μ, sun_pos, problem.shadow_model)
    
    # Compute Q-Law thrust acceleration
    a_thrust_rtn, throttle, _, _ = qlaw_thrust_acceleration(
        oe, problem.oeT, m, problem.spacecraft, problem.weights, problem.params, μ, γ
    )
    
    # Total acceleration in RTN
    acc_total_rtn = acc_rtn + a_thrust_rtn
    
    # =========================================================================
    # GVE using SEMI-MAJOR AXIS formulation (Paper Eq. 3)
    # This is CRITICAL - Q-Law computes optimal thrust using a-based GVE,
    # so dynamics must use the same formulation for consistency!
    # =========================================================================
    
    p = oe.p
    f, g, h, k, L = oe.f, oe.g, oe.h, oe.k, oe.L
    
    # Convert to semi-major axis
    e_sq = f^2 + g^2
    a = p / (1 - e_sq)
    
    # Auxiliary quantities (matching equinoctial_gve_partials in qlaw_core.jl)
    q = compute_q(f, g, L)
    sL, cL = sincos(L)
    
    # Common factor for a-based GVE (Paper Eq. 3)
    sqrt_factor = sqrt(a * (1 - e_sq) / μ)  # = sqrt(p/μ)
    common = sqrt_factor / q
    
    # Build a-based GVE matrix (Paper Eq. 3)
    # Row 1: da/dF
    da_dFr = 2 * a * q * common * (f * sL - g * cL) / (1 - e_sq)
    da_dFt = 2 * a * q^2 * common / (1 - e_sq)
    da_dFh = 0.0
    
    # Row 2: df/dF
    df_dFr = q * common * sL
    df_dFt = common * ((q + 1) * cL + f)
    df_dFh = -common * g * (h * sL - k * cL)
    
    # Row 3: dg/dF
    dg_dFr = -q * common * cL
    dg_dFt = common * ((q + 1) * sL + g)
    dg_dFh = common * f * (h * sL - k * cL)
    
    # Row 4: dh/dF
    s_sq = 1 + h^2 + k^2
    dh_dFr = 0.0
    dh_dFt = 0.0
    dh_dFh = common * s_sq * cL / 2
    
    # Row 5: dk/dF
    dk_dFr = 0.0
    dk_dFt = 0.0
    dk_dFh = common * s_sq * sL / 2
    
    # Row 6: dL/dF (perturbation only)
    dL_dFr = 0.0
    dL_dFt = 0.0
    dL_dFh = common * (h * sL - k * cL)
    
    # Compute element rates
    Fr, Ft, Fh = acc_total_rtn[1], acc_total_rtn[2], acc_total_rtn[3]
    
    da_dt = da_dFr * Fr + da_dFt * Ft + da_dFh * Fh
    df_dt = df_dFr * Fr + df_dFt * Ft + df_dFh * Fh
    dg_dt = dg_dFr * Fr + dg_dFt * Ft + dg_dFh * Fh
    dh_dt = dh_dFr * Fr + dh_dFt * Ft + dh_dFh * Fh
    dk_dt = dk_dFr * Fr + dk_dFt * Ft + dk_dFh * Fh
    dL_dt_pert = dL_dFr * Fr + dL_dFt * Ft + dL_dFh * Fh
    
    # Convert da/dt to dp/dt for state propagation (since state uses p, not a)
    # p = a * (1 - f² - g²)
    # dp/dt = da/dt * (1 - f² - g²) + a * (-2f*df/dt - 2g*dg/dt)
    #       = da/dt * (1 - e²) - 2a * (f*df/dt + g*dg/dt)
    dp_dt = da_dt * (1 - e_sq) - 2 * a * (f * df_dt + g * dg_dt)
    
    # Add Keplerian motion for true longitude (Paper Eq. 5)
    # dL/dt = q² * sqrt(aμ(1-f²-g²)) / (a²(1-f²-g²)) = q² * sqrt(μp) / (a*p)
    dL_keplerian = q^2 * sqrt(μ * p) / (a * p)

    # Orbital radius for mass rate calculation
    r = p / q

    # Mass rate (from thrust)
    if throttle > eps(typeof(throttle))
        T_N = max_thrust(problem.spacecraft, r) * throttle
        vex = exhaust_velocity(problem.spacecraft) * 1000.0  # Convert to m/s
        dm_dt = -T_N / vex  # kg/s
    else
        dm_dt = zero(throttle)
    end
    
    return SVector(
        dp_dt,                      # dp/dt (converted from da/dt)
        df_dt,                      # df/dt
        dg_dt,                      # dg/dt
        dh_dt,                      # dh/dt
        dk_dt,                      # dk/dt
        dL_dt_pert + dL_keplerian,  # dL/dt (perturbation + Keplerian)
        dm_dt
    )
end

function qlaw_eom!(du::AbstractVector, u::AbstractVector, ps::ComponentVector, t::Number,
                   problem::QLawProblem)
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
function apply_frame_rotation(pos::SVector{3,T}, Θrot::Number) where T
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

# Arguments
- `ps`: Parameters (should contain JD0)
- `t`: Time since epoch [s]
- `sun_model`: Optional ThirdBodyModel for accurate ephemeris
- `Θrot`: Frame rotation angle [rad] (Varga optimization variable)
"""
function get_sun_position(ps::ComponentVector, t::Number, sun_model::Union{ThirdBodyModel,Nothing}=nothing; Θrot::Number=0.0)
    # Determine raw sun position
    if haskey(ps, :sun_pos)
        sun_pos = SVector{3}(ps.sun_pos...)
    elseif sun_model !== nothing
        # Compute Julian date
        JD = haskey(ps, :JD) ? ps.JD + t / 86400.0 : 2451545.0 + t / 86400.0
        # ThirdBodyModel returns position in METERS, we need KILOMETERS
        sun_pos_m = sun_model(JD, Position())
        sun_pos = SVector{3}(sun_pos_m[1] / 1000.0, sun_pos_m[2] / 1000.0, sun_pos_m[3] / 1000.0)
    else
        # Fallback: simple circular orbit approximation for Sun (Earth-centered)
        JD = haskey(ps, :JD) ? ps.JD + t / 86400.0 : 2451545.0 + t / 86400.0
        AU = 1.495978707e8  # km
        d = JD - 2451545.0
        M = 357.529 + 0.98560028 * d
        M_rad = deg2rad(M)
        λ = M_rad + π
        sun_pos = SVector{3}(AU * cos(λ), AU * sin(λ), 0.0)
    end
    
    # Apply frame rotation (Varga Θrot parameter)
    return apply_frame_rotation(sun_pos, Θrot)
end


# Note: check_convergence has been consolidated into qlaw_core.jl with
# multiple dispatch on convergence criterion type.

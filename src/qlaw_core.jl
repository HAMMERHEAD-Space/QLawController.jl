# =============================================================================
# Q-Law Core Functions
# 
# These implement the Q-Law specific algorithms from the paper:
# "Q-Law for Rapid Assessment of Low Thrust Cislunar Trajectories via AD"
# =============================================================================

export p_to_a, a_to_p, get_sma, compute_q, compute_radius
export equinoctial_gve_partials
export compute_max_rates
export compute_scaling
export compute_penalty
export compute_Q, compute_Q_from_vec
export compute_dQ_doe_analytical
export compute_Qdot_coefficients, compute_thrust_direction, thrust_direction_to_rtn
export compute_effectivity
export compute_thrust_and_effectivity
export effectivity_activation
export check_convergence

# =============================================================================
# Equinoctial Element Utilities
# =============================================================================

"""
Convert semi-latus rectum p to semi-major axis a.
ModEq uses p, but Q-Law paper uses a for better performance.
"""
@inline function p_to_a(p::T, f::T, g::T) where {T}
    return p / (one(T) - f^2 - g^2)
end

"""Convert semi-major axis a to semi-latus rectum p."""
@inline function a_to_p(a::T, f::T, g::T) where {T}
    return a * (one(T) - f^2 - g^2)
end

"""
Get semi-major axis from ModEq state.
"""
@inline function get_sma(oe::ModEq{T}) where {T}
    return p_to_a(oe.p, oe.f, oe.g)
end

"""
Compute auxiliary parameter q = 1 + f*cos(L) + g*sin(L)
"""
@inline function compute_q(f::T, g::T, L::T) where {T}
    return one(T) + f * cos(L) + g * sin(L)
end

"""
Compute orbital radius from equinoctial elements.
"""
@inline function compute_radius(oe::ModEq{T}) where {T}
    q = compute_q(oe.f, oe.g, oe.L)
    return oe.p / q
end

# =============================================================================
# Gauss Variational Equations A-Matrix for Equinoctial Elements
# 
# This gives ∂oe/∂F where F = [Fr, Fθ, Fh] is the acceleration in RTN frame.
# From Eq. (3) in the paper.
# =============================================================================

"""
    equinoctial_gve_partials(oe::ModEq, μ::Number)

Compute the A matrix for Gauss Variational Equations in equinoctial elements.
Returns a 6×3 matrix where A[i,j] = ∂(oe_i)/∂(F_j) with F = [Fr, Fθ, Fh].

Note: This uses semi-major axis internally (as per Varga & Perez) but
accepts ModEq which uses semi-latus rectum p.

# Returns
Matrix A where:
- Column 1: ∂oe/∂Fr (radial)
- Column 2: ∂oe/∂Fθ (tangential)  
- Column 3: ∂oe/∂Fh (normal)
"""
function equinoctial_gve_partials(oe::ModEq{T}, μ::Number) where {T}
    p, f, g, h, k, L = oe.p, oe.f, oe.g, oe.h, oe.k, oe.L

    # Convert to semi-major axis
    e_sq = f^2 + g^2
    a = p / (one(T) - e_sq)

    # Auxiliary quantities
    q = compute_q(f, g, L)
    sL, cL = sincos(L)

    # Common factor
    sqrt_factor = sqrt(a * (one(T) - e_sq) / μ)
    common = sqrt_factor / q

    # Build A matrix (6×3)
    # Row 1: da/dF
    A11 = 2 * a * q * common * (f * sL - g * cL) / (one(T) - e_sq)
    A12 = 2 * a * q^2 * common / (one(T) - e_sq)
    A13 = zero(T)

    # Row 2: df/dF
    A21 = q * common * sL
    A22 = common * ((q + one(T)) * cL + f)
    A23 = -common * g * (h * sL - k * cL)

    # Row 3: dg/dF
    A31 = -q * common * cL
    A32 = common * ((q + one(T)) * sL + g)
    A33 = common * f * (h * sL - k * cL)

    # Row 4: dh/dF
    s_sq = one(T) + h^2 + k^2
    A41 = zero(T)
    A42 = zero(T)
    A43 = common * s_sq * cL / 2

    # Row 5: dk/dF
    A51 = zero(T)
    A52 = zero(T)
    A53 = common * s_sq * sL / 2

    # Row 6: dL/dF
    A61 = zero(T)
    A62 = zero(T)
    A63 = common * (h * sL - k * cL)

    return SMatrix{6,3,T}(
        A11,
        A21,
        A31,
        A41,
        A51,
        A61,
        A12,
        A22,
        A32,
        A42,
        A52,
        A62,
        A13,
        A23,
        A33,
        A43,
        A53,
        A63,
    )
end

# =============================================================================
# Maximum Rates of Change (ȯₑₓₓ)
# 
# Analytical formulas from Varga & Perez (Eqs. 14-18) for maximum rate of 
# change of each orbital element over all thrust directions and true anomaly.
# These AD-friendly formulas replace numerical search over L.
# =============================================================================

"""
    compute_max_rates_analytical(a, f, g, h, k, μ, F_max)

Compute the maximum rate of change of each orbital element using the 
ANALYTICAL formulas from Varga & Perez (Eqs. 14-18).

These formulas give the maximum rates over all thrust directions AND all
true longitudes on the osculating orbit, without requiring numerical search.

This is AD-friendly because it's just algebraic operations.

Returns vector [ȧₓₓ, ḟₓₓ, ġₓₓ, ḣₓₓ, k̇ₓₓ] (5 elements, excluding L).
"""
function compute_max_rates_analytical(
    a::T,
    f::T,
    g::T,
    h::T,
    k::T,
    μ::Number,
    F_max::Number,
) where {T}
    # Eccentricity
    e = sqrt(f^2 + g^2)

    # Semi-latus rectum
    p = a * (one(T) - e^2)

    # s² = 1 + h² + k²
    s_sq = one(T) + h^2 + k^2

    # Common factor sqrt(p/μ)
    sqrt_p_mu = sqrt(p / μ)

    # Eq. 14: ȧxx = 2*F*a*sqrt(a/μ)*sqrt((1+e)/(1-e))
    # For physical orbits e < 1, so (1-e) > 0. When e = 0, sqrt(1/1) = 1.
    a_dot_xx = T(2) * F_max * a * sqrt(a / μ) * sqrt((one(T) + e) / (one(T) - e))

    # Eq. 15-16: ḟxx ≈ ġxx ≈ 2*F*sqrt(p/μ)
    f_dot_xx = T(2) * F_max * sqrt_p_mu
    g_dot_xx = T(2) * F_max * sqrt_p_mu

    # Eq. 17: ḣxx = (1/2)*F*sqrt(p/μ)*s²/(sqrt(1-g²)+f)
    # For physical orbits f²+g² < 1, so 1-g² > f² → sqrt(1-g²) > |f| → denominator > 0.
    h_dot_xx = T(0.5) * F_max * sqrt_p_mu * s_sq / (sqrt(one(T) - g^2) + f)

    # Eq. 18: k̇xx = (1/2)*F*sqrt(p/μ)*s²/(sqrt(1-f²)+g)
    # Same reasoning: 1-f² > g² → sqrt(1-f²) > |g| → denominator > 0.
    k_dot_xx = T(0.5) * F_max * sqrt_p_mu * s_sq / (sqrt(one(T) - f^2) + g)

    return SVector{5,T}(a_dot_xx, f_dot_xx, g_dot_xx, h_dot_xx, k_dot_xx)
end

"""
    compute_max_rates_at_L(a, f, g, h, k, L, μ, F_max)

Compute the maximum rate of change of each orbital element over all 
thrust directions at a specific true longitude. Works directly with element values.

Returns vector [ȧₓₓ, ḟₓₓ, ġₓₓ, ḣₓₓ, k̇ₓₓ] (5 elements, excluding L).
"""
function compute_max_rates_at_L(
    a::T,
    f::T,
    g::T,
    h::T,
    k::T,
    L::T,
    μ::Number,
    F_max::Number,
) where {T}
    # Build ModEq for GVE partials (needs p, not a)
    p = a * (one(T) - f^2 - g^2)
    oe = ModEq{T}(p, f, g, h, k, L)
    A = equinoctial_gve_partials(oe, μ)

    # Max rate for each element is F_max * ||A[i,:]||
    # Only first 5 elements (a, f, g, h, k), not L
    max_rates = SVector{5,T}(
        F_max * norm(SVector{3,T}(A[1, 1], A[1, 2], A[1, 3])),
        F_max * norm(SVector{3,T}(A[2, 1], A[2, 2], A[2, 3])),
        F_max * norm(SVector{3,T}(A[3, 1], A[3, 2], A[3, 3])),
        F_max * norm(SVector{3,T}(A[4, 1], A[4, 2], A[4, 3])),
        F_max * norm(SVector{3,T}(A[5, 1], A[5, 2], A[5, 3])),
    )

    return max_rates
end

"""
    compute_max_rates(a, f, g, h, k, L, μ, F_max; n_points=20)

Compute the maximum rate of change of each orbital element using a hybrid
analytical/numerical approach:
- a: Exact analytical formula (Varga Eq. 14) — max occurs at periapsis
- f, g, h, k: Grid search over true longitude

The Varga analytical formulas for f, g (Eqs. 15-16) and h, k (Eqs. 17-18)
are approximations that can underestimate the true orbit-wide maximum for
eccentric orbits. For f and g, the approximation ḟ_xx ≈ 2F√(p/μ) is exact
only for circular orbits. For h and k, the formulas assume the maximum of
|cos(L)|/q or |sin(L)|/q occurs at the same L as the analytical extremum,
which is incorrect when eccentricity shifts the q-denominator.

The grid search evaluates the GVE partials at `n_points` uniformly spaced
true longitudes to find the actual maximum.

Returns vector [ȧₓₓ, ḟₓₓ, ġₓₓ, ḣₓₓ, k̇ₓₓ] (5 elements, excluding L).
"""
function compute_max_rates(
    a::T,
    f::T,
    g::T,
    h::T,
    k::T,
    L::T,
    μ::Number,
    F_max::Number;
    n_points::Int = 20,
) where {T}
    # Exact analytical rate for a (Varga Eq. 14) — truly exact since max energy
    # change rate occurs at periapsis with pure tangential thrust
    analytical = compute_max_rates_analytical(a, f, g, h, k, μ, F_max)

    # Grid search for f, g, h, k max rates
    # Initialize with analytical values as lower bounds
    f_dot_max = analytical[2]
    g_dot_max = analytical[3]
    h_dot_max = analytical[4]
    k_dot_max = analytical[5]

    for i = 0:(n_points-1)
        L_test = T(2π) * T(i) / T(n_points)
        rates_at_L = compute_max_rates_at_L(a, f, g, h, k, L_test, μ, F_max)
        f_dot_max = max(f_dot_max, rates_at_L[2])
        g_dot_max = max(g_dot_max, rates_at_L[3])
        h_dot_max = max(h_dot_max, rates_at_L[4])
        k_dot_max = max(k_dot_max, rates_at_L[5])
    end

    return SVector{5,T}(analytical[1], f_dot_max, g_dot_max, h_dot_max, k_dot_max)
end

# Convenience method that takes ModEq
function compute_max_rates(oe::ModEq{T}, μ::Number, F_max::Number) where {T}
    a = get_sma(oe)
    return compute_max_rates(a, oe.f, oe.g, oe.h, oe.k, oe.L, μ, F_max)
end

# =============================================================================
# Scaling Function S_oe
# 
# Prevents divergence in semi-major axis during targeting.
# =============================================================================

"""
    compute_scaling(a, aT, m_scaling=1.0, n_scaling=4.0, r_scaling=2.0)

Compute the scaling function S for each orbital element (Varga Eq. 8).
Sa = [1 + (|a - aT| / (m * aT))^n]^(1/r)

Returns vector [Sa, Sf, Sg, Sh, Sk].
"""
function compute_scaling(
    a::T,
    aT::T,
    m_scaling::Number = 1.0,
    n_scaling::Number = 4.0,
    r_scaling::Number = 2.0,
) where {T}
    # Varga Eq. 8: Sa = [1 + (|a - at|/(m*at))^n]^(1/r)
    Sa = (one(T) + (abs(a - aT) / (T(m_scaling) * aT))^T(n_scaling))^(one(T) / T(r_scaling))

    # Other elements don't need scaling (set to 1)
    return SVector{5,T}(Sa, one(T), one(T), one(T), one(T))
end

# Convenience method that takes ModEq
function compute_scaling(
    oe::ModEq{T},
    oeT::ModEq{T};
    m_scaling::Number = 1.0,
    n_scaling::Number = 4.0,
    r_scaling::Number = 2.0,
) where {T}
    return compute_scaling(get_sma(oe), get_sma(oeT), m_scaling, n_scaling, r_scaling)
end

# =============================================================================
# Penalty Function P
# 
# Enforces minimum periapsis constraint.
# =============================================================================

"""
    compute_penalty(a, f, g, rp_min, k_pen=100.0)

Compute the periapsis penalty function P (Varga Eq. 9).
P = exp(k * (1 - rp/rp_min))

This is smooth/continuous everywhere (no branching), which is required for
AD compatibility. When rp >> rp_min the exponent is very negative and P ≈ 0.

# Arguments
- `k_pen`: Steepness of the exponential barrier (default: 100.0)
"""
function compute_penalty(a::T, f::T, g::T, rp_min::Number, k_pen::Number = 100.0) where {T}
    # Regularized eccentricity: eps(T)^2 prevents 0/0 in ForwardDiff
    # derivative of sqrt at f=g=0 (circular orbits)
    e = sqrt(f^2 + g^2 + eps(T)^2)
    rp = a * (one(T) - e)  # Periapsis radius
    return exp(T(k_pen) * (one(T) - rp / rp_min))
end

# Convenience method that takes ModEq
function compute_penalty(oe::ModEq{T}, rp_min::Number, k_pen::Number = 100.0) where {T}
    return compute_penalty(get_sma(oe), oe.f, oe.g, rp_min, k_pen)
end

# =============================================================================
# Q-Law Lyapunov Function Q
# 
# From Eq. (21): Q = (1 + Wp*P) * Σ Woe * Soe * ((oe - oeT) / ȯexx)²
# =============================================================================

"""
    compute_Q_from_vec_with_rates(oe_vec, oeT_vec, W_vec, max_rates, Wp, rp_min)

Core Q computation with precomputed max_rates (AD-friendly).
max_rates is treated as a constant, not differentiated through.

# Arguments
- `oe_vec`: Current elements [a, f, g, h, k]
- `oeT_vec`: Target elements [aT, fT, gT, hT, kT]  
- `W_vec`: Weights [Wa, Wf, Wg, Wh, Wk]
- `max_rates`: Precomputed maximum rates (constant for AD)
- `Wp`: Penalty weight
- `rp_min`: Minimum periapsis radius [km]
"""
function compute_Q_from_vec_with_rates(
    oe_vec::SVector{5,T},
    oeT_vec::SVector{5,Tt},
    W_vec::SVector{5,Tw},
    max_rates::SVector{5,Tr},
    Wp::Number,
    rp_min::Number,
    m_scaling::Number = 1.0,
    n_scaling::Number = 4.0,
    r_scaling::Number = 2.0,
    k_pen::Number = 100.0,
) where {T,Tt,Tw,Tr}

    a, f, g, h, k = oe_vec[1], oe_vec[2], oe_vec[3], oe_vec[4], oe_vec[5]
    aT = oeT_vec[1]

    # Compute scaling (Varga Eq. 8)
    S_vec = compute_scaling(a, T(aT), m_scaling, n_scaling, r_scaling)

    # Compute penalty
    P = compute_penalty(a, f, g, rp_min, k_pen)

    # Compute Q - max_rates is constant, not differentiated
    Q = zero(T)
    for i = 1:5
        if max_rates[i] > eps(Tr)
            term = W_vec[i] * S_vec[i] * ((oe_vec[i] - oeT_vec[i]) / max_rates[i])^2
            Q += term
        end
    end

    Q *= (one(T) + Wp * P)

    return Q
end

"""
    compute_dQ_doe_analytical(oe_vec, oeT_vec, W_vec, max_rates, Wp, rp_min, ...)

Analytical gradient ∂Q/∂oe where oe = [a, f, g, h, k].

Computes the exact same result as `ForwardDiff.gradient` of
`compute_Q_from_vec_with_rates`, but without AD overhead.

`max_rates` is treated as a constant (not differentiated through),
consistent with standard Q-Law practice (Varga Eq. 21).

# Derivation
Q = (1 + Wp·P) · Σᵢ Wᵢ·Sᵢ·((oeᵢ - oeTᵢ)/rateᵢ)²

By the product rule:
  ∂Q/∂oeⱼ = Wp·(∂P/∂oeⱼ)·Qsum + (1 + Wp·P)·(∂Qsum/∂oeⱼ)

where Qsum = Σᵢ Wᵢ·Sᵢ·((oeᵢ - oeTᵢ)/rateᵢ)².

- P = exp(k·(1 - rp/rp_min)) with rp = a·(1-e), e = √(f²+g²+ε²)
- Sa = (1 + (|a-aT|/(m·aT))ⁿ)^(1/r); Sf = Sg = Sh = Sk = 1

Returns `SVector{5}` gradient [∂Q/∂a, ∂Q/∂f, ∂Q/∂g, ∂Q/∂h, ∂Q/∂k].
"""
function compute_dQ_doe_analytical(
    oe_vec::SVector{5,T},
    oeT_vec::SVector{5,Tt},
    W_vec::SVector{5,Tw},
    max_rates::SVector{5,Tr},
    Wp::Number,
    rp_min::Number,
    m_scaling::Number = 1.0,
    n_scaling::Number = 4.0,
    r_scaling::Number = 2.0,
    k_pen::Number = 100.0,
) where {T,Tt,Tw,Tr}

    a, f, g, h, k = oe_vec[1], oe_vec[2], oe_vec[3], oe_vec[4], oe_vec[5]
    aT = T(oeT_vec[1])

    # Type-converted parameters
    _k_pen = T(k_pen)
    _rp_min = T(rp_min)
    _m_s = T(m_scaling)
    _n_s = T(n_scaling)
    _r_s = T(r_scaling)
    _Wp = T(Wp)

    # ---- Regularized eccentricity (matches compute_penalty) ----
    e_reg = sqrt(f^2 + g^2 + eps(T)^2)

    # ---- Penalty function P (Varga Eq. 9) ----
    # P = exp(k_pen * (1 - rp / rp_min)),  rp = a * (1 - e)
    rp = a * (one(T) - e_reg)
    P = exp(_k_pen * (one(T) - rp / _rp_min))

    # ∂P/∂a:  ∂rp/∂a = (1-e), so ∂P/∂a = P · k_pen · (-(1-e)/rp_min)
    dP_da = -P * _k_pen * (one(T) - e_reg) / _rp_min
    # ∂P/∂f:  ∂rp/∂f = -a·∂e/∂f = -a·f/e, so ∂P/∂f = P · k_pen · a·f/(rp_min·e)
    dP_df = P * _k_pen * a * f / (_rp_min * e_reg)
    # ∂P/∂g:  same structure as f
    dP_dg = P * _k_pen * a * g / (_rp_min * e_reg)
    # P does not depend on h or k

    # ---- Scaling function Sa (Varga Eq. 8) ----
    # Sa = (1 + u^n)^(1/r),  u = |a - aT| / (m · aT)
    Δa = a - aT
    u = abs(Δa) / (_m_s * aT)
    v = u^_n_s                              # u^n
    Sa = (one(T) + v)^(one(T) / _r_s)      # (1 + u^n)^(1/r)

    # ∂Sa/∂a = (n/r) · sign(Δa)/(m·aT) · u^(n-1) · (1+v)^(1/r - 1)
    # When Δa = 0: sign(0) = 0 → dSa_da = 0.  When n > 1 and u = 0: u^(n-1) = 0.
    dSa_da =
        (_n_s / _r_s) * sign(Δa) / (_m_s * aT) *
        u^(_n_s - one(T)) *
        (one(T) + v)^(one(T) / _r_s - one(T))

    # Sf = Sg = Sh = Sk = 1 (no derivatives)

    # ---- Per-element Q contributions (Qsum) and their derivatives ----
    Qsum = zero(T)
    dQsum_da = zero(T)
    dQsum_df = zero(T)
    dQsum_dg = zero(T)
    dQsum_dh = zero(T)
    dQsum_dk = zero(T)

    # Element 1 (a):  Q₁ = Wa · Sa · (δa / ra)²
    if max_rates[1] > eps(Tr)
        ra = max_rates[1]
        δa = Δa
        Qsum += W_vec[1] * Sa * (δa / ra)^2
        # ∂Q₁/∂a = Wa · (∂Sa/∂a · δa² + Sa · 2·δa) / ra²
        dQsum_da = W_vec[1] * (dSa_da * δa^2 + Sa * T(2) * δa) / ra^2
    end

    # Elements 2-5 (f, g, h, k):  Qᵢ = Wᵢ · (δᵢ / rᵢ)²   (Sᵢ = 1)
    if max_rates[2] > eps(Tr)
        δf = f - T(oeT_vec[2])
        Qsum += W_vec[2] * (δf / max_rates[2])^2
        dQsum_df = T(2) * W_vec[2] * δf / max_rates[2]^2
    end

    if max_rates[3] > eps(Tr)
        δg = g - T(oeT_vec[3])
        Qsum += W_vec[3] * (δg / max_rates[3])^2
        dQsum_dg = T(2) * W_vec[3] * δg / max_rates[3]^2
    end

    if max_rates[4] > eps(Tr)
        δh = h - T(oeT_vec[4])
        Qsum += W_vec[4] * (δh / max_rates[4])^2
        dQsum_dh = T(2) * W_vec[4] * δh / max_rates[4]^2
    end

    if max_rates[5] > eps(Tr)
        δk = k - T(oeT_vec[5])
        Qsum += W_vec[5] * (δk / max_rates[5])^2
        dQsum_dk = T(2) * W_vec[5] * δk / max_rates[5]^2
    end

    # ---- Full gradient via product rule ----
    # Q = (1 + Wp·P) · Qsum
    # ∂Q/∂oeⱼ = Wp·(∂P/∂oeⱼ)·Qsum + (1 + Wp·P)·(∂Qsum/∂oeⱼ)
    pf = one(T) + _Wp * P   # penalty factor

    dQ_da = _Wp * dP_da * Qsum + pf * dQsum_da
    dQ_df = _Wp * dP_df * Qsum + pf * dQsum_df
    dQ_dg = _Wp * dP_dg * Qsum + pf * dQsum_dg
    dQ_dh = pf * dQsum_dh    # P independent of h
    dQ_dk = pf * dQsum_dk    # P independent of k

    return SVector{5,T}(dQ_da, dQ_df, dQ_dg, dQ_dh, dQ_dk)
end

"""
    compute_Q_from_vec(oe_vec, L, oeT_vec, W_vec, μ, F_max, Wp, rp_min)

Core Q computation working directly with element vectors (a, f, g, h, k).
Computes max_rates internally - use compute_Q_from_vec_with_rates for AD.

# Arguments
- `oe_vec`: Current elements [a, f, g, h, k]
- `L`: True longitude (not used, kept for API compatibility)
- `oeT_vec`: Target elements [aT, fT, gT, hT, kT]
- `W_vec`: Weights [Wa, Wf, Wg, Wh, Wk]
- `μ`: Gravitational parameter [km³/s²]
- `F_max`: Maximum thrust acceleration [km/s²]
- `Wp`: Penalty weight
- `rp_min`: Minimum periapsis radius [km]
"""
function compute_Q_from_vec(
    oe_vec::SVector{5,T},
    L::Number,
    oeT_vec::SVector{5,Tt},
    W_vec::SVector{5,Tw},
    μ::Number,
    F_max::Number,
    Wp::Number,
    rp_min::Number,
    m_scaling::Number = 1.0,
    n_scaling::Number = 4.0,
    r_scaling::Number = 2.0,
    k_pen::Number = 100.0,
) where {T,Tt,Tw}

    a, f, g, h, k = oe_vec[1], oe_vec[2], oe_vec[3], oe_vec[4], oe_vec[5]

    # Compute max rates using analytical formulas (Varga Eqs. 14-18).
    # Consistent with compute_Qdot_coefficients, which also uses analytical rates
    # so that the Q function being minimized by the controller is the same one
    # used for convergence diagnostics.
    max_rates = compute_max_rates_analytical(a, f, g, h, k, μ, F_max)

    return compute_Q_from_vec_with_rates(
        oe_vec,
        oeT_vec,
        W_vec,
        max_rates,
        Wp,
        rp_min,
        m_scaling,
        n_scaling,
        r_scaling,
        k_pen,
    )
end

"""
    compute_Q(oe::ModEq, oeT::ModEq, weights::QLawWeights, 
              μ::Number, F_max::Number, Wp::Number, rp_min::Number)

Compute the Q-Law Lyapunov function value.

# Arguments
- `oe`: Current orbital elements
- `oeT`: Target orbital elements
- `weights`: Q-Law weights for each element
- `μ`: Gravitational parameter [km³/s²]
- `F_max`: Maximum thrust acceleration [km/s²]
- `Wp`: Penalty weight
- `rp_min`: Minimum periapsis radius [km]

# Returns
Q value (scalar)
"""
function compute_Q(
    oe::ModEq{T},
    oeT::ModEq{T},
    weights::QLawWeights{T},
    μ::Number,
    F_max::Number,
    Wp::Number,
    rp_min::Number,
    m_scaling::Number = 1.0,
    n_scaling::Number = 4.0,
    r_scaling::Number = 2.0,
    k_pen::Number = 100.0,
) where {T}

    # Get element vectors (a, f, g, h, k)
    a = get_sma(oe)
    aT = get_sma(oeT)

    oe_vec = SVector{5,T}(a, oe.f, oe.g, oe.h, oe.k)
    oeT_vec = SVector{5,T}(aT, oeT.f, oeT.g, oeT.h, oeT.k)
    W_vec = SVector{5,T}(weights.Wa, weights.Wf, weights.Wg, weights.Wh, weights.Wk)

    return compute_Q_from_vec(
        oe_vec,
        oe.L,
        oeT_vec,
        W_vec,
        μ,
        F_max,
        Wp,
        rp_min,
        m_scaling,
        n_scaling,
        r_scaling,
        k_pen,
    )
end

# Convenience: dispatch using QLawParameters
function compute_Q(
    oe::ModEq{T},
    oeT::ModEq{T},
    weights::QLawWeights{T},
    μ::Number,
    F_max::Number,
    params::QLawParameters,
) where {T}
    return compute_Q(
        oe,
        oeT,
        weights,
        μ,
        F_max,
        params.Wp,
        params.rp_min,
        params.m_scaling,
        params.n_scaling,
        params.r_scaling,
        params.k_penalty,
    )
end

# =============================================================================
# Optimal Thrust Direction
# 
# From Eq. (23)-(28): Minimize Q̇ to find optimal α*, β*
# =============================================================================

"""
    compute_Qdot_coefficients(oe::ModEq, oeT::ModEq, weights::QLawWeights,
                               μ::Number, F_max::Number, Wp::Number, rp_min::Number)

Compute the D1, D2, D3 coefficients for Q̇.
Q̇ = D1*cos(β)*cos(α) + D2*cos(β)*sin(α) + D3*sin(β)

From Eq. (24)-(26):
D1 = Σ (∂Q/∂oe) * (∂oe/∂Fθ)  [tangential]
D2 = Σ (∂Q/∂oe) * (∂oe/∂Fr)  [radial]
D3 = Σ (∂Q/∂oe) * (∂oe/∂Fh)  [normal]

Note: Per standard Q-Law (Varga Eq. 21), max_rates (ȯₑₓₓ) are treated as 
CONSTANTS when computing ∂Q/∂oe. They serve as normalization factors.
"""
function compute_Qdot_coefficients(
    oe::ModEq{T1},
    oeT::ModEq{T2},
    weights::QLawWeights{T3},
    μ::Number,
    F_max::Number,
    Wp::Number,
    rp_min::Number,
    m_scaling::Number = 1.0,
    n_scaling::Number = 4.0,
    r_scaling::Number = 2.0,
    k_pen::Number = 100.0,
) where {T1<:Number,T2<:Number,T3<:Number}
    T = promote_type(T1, T2, T3)
    # Get current orbital elements
    a = get_sma(oe)
    aT = get_sma(oeT)

    # Precompute max_rates for this orbit shape - treated as CONSTANT for ∂Q/∂oe
    # Per Varga Eq. (21): ∂Q/∂oe = 2*Soe*Woe*(oe-oeT)/ȯexx²
    # The ȯexx terms are normalization constants, not differentiated through
    #
    # IMPORTANT: Use analytical formulas here (not hybrid grid search) because
    # the grid search max() introduces non-smooth dependence on orbital state.
    # As the orbit evolves, the grid point that "wins" the max can switch
    # discretely, creating kinks in Q that propagate as thrust direction spikes.
    # The analytical formulas are smooth functions of the orbital elements,
    # ensuring the ODE integrator sees a smooth RHS.
    max_rates = compute_max_rates_analytical(a, oe.f, oe.g, oe.h, oe.k, μ, F_max)

    # Precompute target and weight vectors
    oeT_vec = SVector{5,T}(aT, oeT.f, oeT.g, oeT.h, oeT.k)
    W_vec = SVector{5,T}(weights.Wa, weights.Wf, weights.Wg, weights.Wh, weights.Wk)

    # Compute ∂Q/∂oe analytically (max_rates treated as CONSTANT)
    oe_vec = SVector{5,T}(a, oe.f, oe.g, oe.h, oe.k)
    dQ_doe = compute_dQ_doe_analytical(
        oe_vec,
        oeT_vec,
        W_vec,
        max_rates,
        Wp,
        rp_min,
        m_scaling,
        n_scaling,
        r_scaling,
        k_pen,
    )

    # Get A matrix (∂oe/∂F) at current position
    A = equinoctial_gve_partials(oe, μ)

    # D coefficients: Q̇ = D1·cos(β)cos(α) + D2·cos(β)sin(α) + D3·sin(β)
    # Per Varga Eq. 20-23: D_i = F_max · Σ (∂Q/∂oe_j)(∂oe_j/∂F_i)
    # A columns: [Fr, Fθ, Fh] = [radial, tangential, normal]
    D1 = zero(T)  # Tangential: ∂oe/∂Fθ is column 2
    D2 = zero(T)  # Radial: ∂oe/∂Fr is column 1
    D3 = zero(T)  # Normal: ∂oe/∂Fh is column 3

    for i = 1:5
        D1 += dQ_doe[i] * A[i, 2]  # Tangential
        D2 += dQ_doe[i] * A[i, 1]  # Radial
        D3 += dQ_doe[i] * A[i, 3]  # Normal
    end

    # Scale by F_max so that returned Q̇ values have the correct physical magnitude.
    # This does not affect optimal thrust angles (F_max cancels in arctan) or
    # effectivity ratios (F_max cancels in Qdot_n / Qdot_nn).
    D1 *= F_max
    D2 *= F_max
    D3 *= F_max

    return (D1, D2, D3)
end

"""
    compute_thrust_direction(oe::ModEq, oeT::ModEq, weights::QLawWeights,
                              μ::Number, F_max::Number, Wp::Number, rp_min::Number)

Compute the optimal thrust direction angles (α*, β*) in RTN frame.

# Returns
- `α`: In-plane angle (from tangential toward radial) [rad]
- `β`: Out-of-plane angle [rad]
- `Qdot_n`: Minimized Q̇ value
"""
function compute_thrust_direction(
    oe::ModEq{T1},
    oeT::ModEq{T2},
    weights::QLawWeights{T3},
    μ::Number,
    F_max::Number,
    Wp::Number,
    rp_min::Number,
    m_scaling::Number = 1.0,
    n_scaling::Number = 4.0,
    r_scaling::Number = 2.0,
    k_pen::Number = 100.0,
) where {T1<:Number,T2<:Number,T3<:Number}

    D1, D2, D3 = compute_Qdot_coefficients(
        oe,
        oeT,
        weights,
        μ,
        F_max,
        Wp,
        rp_min,
        m_scaling,
        n_scaling,
        r_scaling,
        k_pen,
    )

    # From Eq. (27)-(28)
    α_opt = atan(-D2, -D1)
    D12 = sqrt(D1^2 + D2^2)
    β_opt = atan(-D3, D12)

    # Compute Q̇ at optimal direction
    Qdot_n = D1 * cos(β_opt) * cos(α_opt) + D2 * cos(β_opt) * sin(α_opt) + D3 * sin(β_opt)

    return (α_opt, β_opt, Qdot_n)
end

# Convenience: dispatch using QLawParameters
function compute_thrust_direction(
    oe::ModEq{T1},
    oeT::ModEq{T2},
    weights::QLawWeights{T3},
    μ::Number,
    F_max::Number,
    params::QLawParameters,
) where {T1<:Number,T2<:Number,T3<:Number}
    return compute_thrust_direction(
        oe,
        oeT,
        weights,
        μ,
        F_max,
        params.Wp,
        params.rp_min,
        params.m_scaling,
        params.n_scaling,
        params.r_scaling,
        params.k_penalty,
    )
end

"""
    thrust_direction_to_rtn(α::Number, β::Number)

Convert thrust angles to unit vector in RTN frame.
"""
function thrust_direction_to_rtn(α::AT, β::BT) where {AT<:Number,BT<:Number}
    # RTN components: [radial, tangential, normal]
    T = promote_type(AT, BT)
    Fr = cos(β) * sin(α)
    Fθ = cos(β) * cos(α)
    Fh = sin(β)
    return SVector{3,T}(Fr, Fθ, Fh)
end

# =============================================================================
# Effectivity
# 
# From Eq. (29)-(32): Determines when to coast
# =============================================================================

"""
    compute_effectivity(oe, oeT, weights, μ, F_max, Wp, rp_min, n_points, eff_type,
                        search_method; Qdot_n_precomputed=nothing, kwargs...)

Compute the effectivity η.

# Arguments
- `eff_type`: `AbsoluteEffectivity()` or `RelativeEffectivity()`
- `search_method`: `GridSearch()` or `RefinedSearch()` (Brent refinement via Optim.jl)
- `n_points`: Number of points for grid search over true longitude
- `Qdot_n_precomputed`: If provided, skip recomputing Q̇ at current position (avoids redundant work)

# Returns
- `η`: Effectivity value
- `Qdot_n`: Q̇ at current position
- `Qdot_nn`: Minimum Q̇ over all true longitudes
- `Qdot_nx`: Maximum Q̇ over all true longitudes
"""
function compute_effectivity(
    oe::ModEq{T1},
    oeT::ModEq{T2},
    weights::QLawWeights{T3},
    μ::Number,
    F_max::Number,
    Wp::Number,
    rp_min::Number,
    n_points::Int,
    eff_type::AbstractEffectivityType,
    search_method::AbstractEffectivitySearch = RefinedSearch();
    m_scaling::Number = 1.0,
    n_scaling::Number = 4.0,
    r_scaling::Number = 2.0,
    k_pen::Number = 100.0,
    Qdot_n_precomputed::Union{Nothing,Number} = nothing,
) where {T1<:Number,T2<:Number,T3<:Number}

    T = promote_type(T1, T2, T3)

    # Get Q̇ at current position (reuse pre-computed value if available)
    if Qdot_n_precomputed !== nothing
        Qdot_n = T(Qdot_n_precomputed)
    else
        _, _, Qdot_n = compute_thrust_direction(
            oe,
            oeT,
            weights,
            μ,
            F_max,
            Wp,
            rp_min,
            m_scaling,
            n_scaling,
            r_scaling,
            k_pen,
        )
    end

    # Helper: compute Q̇ at a given true longitude
    function _qdot_at_L(L_val)
        oe_test = ModEq{T}(oe.p, oe.f, oe.g, oe.h, oe.k, T(L_val))
        _, _, Qdot_test = compute_thrust_direction(
            oe_test,
            oeT,
            weights,
            μ,
            F_max,
            Wp,
            rp_min,
            m_scaling,
            n_scaling,
            r_scaling,
            k_pen,
        )
        return Qdot_test
    end

    # Grid search to find approximate extrema
    Qdot_nn = Qdot_n
    Qdot_nx = Qdot_n
    L_nn = oe.L  # L at approximate minimum
    L_nx = oe.L  # L at approximate maximum

    L_range = range(zero(T), T(2π), length = n_points)

    for L_test in L_range
        Qdot_test = _qdot_at_L(L_test)

        if Qdot_test < Qdot_nn
            Qdot_nn = Qdot_test
            L_nn = L_test
        end
        if Qdot_test > Qdot_nx
            Qdot_nx = Qdot_test
            L_nx = L_test
        end
    end

    # Refine extrema using Brent's method (Optim.jl) if requested
    if search_method isa RefinedSearch
        ΔL = T(2π) / n_points

        # Refine minimum (Qdot_nn) using Brent's method
        L_lo_min = Float64(L_nn - ΔL)
        L_hi_min = Float64(L_nn + ΔL)
        result_min = optimize(
            L_val -> Float64(_qdot_at_L(L_val)),
            L_lo_min,
            L_hi_min,
            Brent();
            abs_tol = 1e-8,
        )
        Qdot_nn_refined = T(Optim.minimum(result_min))
        Qdot_nn = min(Qdot_nn, Qdot_nn_refined)

        # Refine maximum (Qdot_nx) by minimizing the negative
        L_lo_max = Float64(L_nx - ΔL)
        L_hi_max = Float64(L_nx + ΔL)
        result_max = optimize(
            L_val -> -Float64(_qdot_at_L(L_val)),
            L_lo_max,
            L_hi_max,
            Brent();
            abs_tol = 1e-8,
        )
        Qdot_nx_refined = T(-Optim.minimum(result_max))
        Qdot_nx = max(Qdot_nx, Qdot_nx_refined)
    end

    η = _compute_eta(eff_type, Qdot_n, Qdot_nn, Qdot_nx, T)

    return (η, Qdot_n, Qdot_nn, Qdot_nx)
end

# Absolute effectivity: ηa = Q̇n / Q̇nn (Eq. 29)
function _compute_eta(::AbsoluteEffectivity, Qdot_n, Qdot_nn, Qdot_nx, ::Type{T}) where {T}
    return Qdot_nn ≈ zero(T) ? one(T) : Qdot_n / Qdot_nn
end

# Relative effectivity: ηr = (Q̇n - Q̇nx) / (Q̇nn - Q̇nx) (Eq. 30)
function _compute_eta(::RelativeEffectivity, Qdot_n, Qdot_nn, Qdot_nx, ::Type{T}) where {T}
    denom = Qdot_nn - Qdot_nx
    return abs(denom) < eps(T) ? one(T) : (Qdot_n - Qdot_nx) / denom
end

# Convenience: dispatch using QLawParameters
function compute_effectivity(
    oe::ModEq{T1},
    oeT::ModEq{T2},
    weights::QLawWeights{T3},
    μ::Number,
    F_max::Number,
    params::QLawParameters;
    Qdot_n_precomputed::Union{Nothing,Number} = nothing,
) where {T1<:Number,T2<:Number,T3<:Number}
    return compute_effectivity(
        oe,
        oeT,
        weights,
        μ,
        F_max,
        params.Wp,
        params.rp_min,
        params.n_search_points,
        params.effectivity_type,
        params.effectivity_search;
        m_scaling = params.m_scaling,
        n_scaling = params.n_scaling,
        r_scaling = params.r_scaling,
        k_pen = params.k_penalty,
        Qdot_n_precomputed = Qdot_n_precomputed,
    )
end

# =============================================================================
# Combined Thrust Direction + Effectivity (avoids redundant computation)
# =============================================================================

"""
    compute_thrust_and_effectivity(oe, oeT, weights, μ, F_max, params)

Compute both the optimal thrust direction and effectivity in a single call,
avoiding the redundant Q̇ computation at the current position.

# Returns
- `α`: In-plane thrust angle [rad]
- `β`: Out-of-plane thrust angle [rad]
- `Qdot_n`: Minimized Q̇ value at current position
- `η`: Effectivity value
"""
function compute_thrust_and_effectivity(
    oe::ModEq{T1},
    oeT::ModEq{T2},
    weights::QLawWeights{T3},
    μ::Number,
    F_max::Number,
    params::QLawParameters,
) where {T1<:Number,T2<:Number,T3<:Number}

    # Compute thrust direction (once)
    α, β, Qdot_n = compute_thrust_direction(oe, oeT, weights, μ, F_max, params)

    # Compute effectivity, reusing the already-computed Qdot_n
    η, _, _, _ =
        compute_effectivity(oe, oeT, weights, μ, F_max, params; Qdot_n_precomputed = Qdot_n)

    return (α, β, Qdot_n, η)
end

# =============================================================================
# Activation Function (smooth effectivity threshold)
# 
# From Eq. (33): AD-compatible alternative to if-then statements
# =============================================================================

"""
    effectivity_activation(η::Number, η_threshold::Number, μ_smooth::Number=1e-4)

Compute the activation value for effectivity-based coasting.
Returns value in (0, 1) that smoothly transitions at threshold.

From Eq. (33): activation = 0.5 * (1 + tanh((η - η_tr) / μ))
"""
function effectivity_activation(
    η::NT,
    η_threshold::NT2,
    μ_smooth::MT3 = NT(1e-4),
) where {NT<:Number,NT2<:Number,MT3<:Number}
    T = promote_type(NT, NT2, MT3)
    return T(0.5) * (one(T) + tanh((η - η_threshold) / μ_smooth))
end

# =============================================================================
# Convergence Checks
#
# Multiple dispatch on convergence criterion type.
# =============================================================================

"""
    _element_errors(oe, oeT)

Compute per-element relative errors between current and target orbital elements.

Normalization scales:
- Semi-major axis: aT  (always positive for physical orbits)
- f, g: max(√(fT² + gT²), ε) — eccentricity magnitude, floored for near-circular targets
- h, k: max(√(hT² + kT²), ε) — inclination factor, floored for near-equatorial targets

The floor ε = 0.01 prevents division by zero for circular/equatorial targets
and ensures absolute errors below ε × tol are always considered converged.
"""
function _element_errors(oe::ModEq, oeT::ModEq)
    a = get_sma(oe)
    aT = get_sma(oeT)

    ε = 0.01
    e_ref = max(sqrt(oeT.f^2 + oeT.g^2), ε)
    hk_ref = max(sqrt(oeT.h^2 + oeT.k^2), ε)

    err_a = abs(a - aT) / aT
    err_f = abs(oe.f - oeT.f) / e_ref
    err_g = abs(oe.g - oeT.g) / e_ref
    err_h = abs(oe.h - oeT.h) / hk_ref
    err_k = abs(oe.k - oeT.k) / hk_ref

    return (err_a, err_f, err_g, err_h, err_k)
end

"""
    check_convergence(oe, oeT, criterion::SummedErrorConvergence)

Check convergence using summed per-element relative errors.
Legacy method without weights — checks all elements.
"""
function check_convergence(
    oe::ModEq{T},
    oeT::ModEq{T2},
    criterion::SummedErrorConvergence,
) where {T<:Number,T2<:Number}
    err_a, err_f, err_g, err_h, err_k = _element_errors(oe, oeT)
    return (err_a + err_f + err_g + err_h + err_k) < criterion.tol
end

"""
    check_convergence(oe, oeT, weights, μ, F_max, params, criterion::VargaConvergence)

Check convergence using Q-function value (Varga Eq. 35).

Q is normalized by the target orbit's characteristic timescale squared
(aT³/μ) so that the criterion is independent of physical units:
    Q * (μ / aT³) < Rc * √(Σ Woe)
With Rc=1 (Varga's nominal value), this converges when the orbit is
within ~0.5-1% of the target elements.
"""
function check_convergence(
    oe::ModEq{T1},
    oeT::ModEq{T2},
    weights::QLawWeights{T3},
    μ::Number,
    F_max::Number,
    params::QLawParameters,
    criterion::VargaConvergence,
) where {T1<:Number,T2<:Number,T3<:Number}
    Q_val = compute_Q(oe, oeT, weights, μ, F_max, params)
    W_sum = weights.Wa + weights.Wf + weights.Wg + weights.Wh + weights.Wk

    # Normalize Q by target orbit characteristic timescale squared (aT³/μ)
    # This makes Q dimensionless, matching Varga's assumed canonical units
    aT = get_sma(oeT)
    Q_normalized = Q_val * μ / aT^3

    return Q_normalized < criterion.Rc * sqrt(W_sum)
end

"""
    check_convergence(oe, oeT, weights, μ, F_max, params, criterion::SummedErrorConvergence)

Weight-aware summed relative error convergence. Only checks elements with non-zero weight,
so elements marked "free" (W=0) are not required to match the target.
"""
function check_convergence(
    oe::ModEq{T1},
    oeT::ModEq{T2},
    weights::QLawWeights{T3},
    μ::Number,
    F_max::Number,
    params::QLawParameters,
    criterion::SummedErrorConvergence,
) where {T1<:Number,T2<:Number,T3<:Number}
    err_a, err_f, err_g, err_h, err_k = _element_errors(oe, oeT)

    total_err = zero(promote_type(T1, T2))

    # Only accumulate errors for targeted elements (W > 0)
    (weights.Wa > 0) && (total_err += err_a)
    (weights.Wf > 0) && (total_err += err_f)
    (weights.Wg > 0) && (total_err += err_g)
    (weights.Wh > 0) && (total_err += err_h)
    (weights.Wk > 0) && (total_err += err_k)

    return total_err < criterion.tol
end

"""
    check_convergence(oe, oeT, criterion::MaxElementConvergence)

Check convergence using the maximum per-element relative error.
Legacy method without weights — checks all elements.
"""
function check_convergence(
    oe::ModEq{T},
    oeT::ModEq{T2},
    criterion::MaxElementConvergence,
) where {T<:Number,T2<:Number}
    err_a, err_f, err_g, err_h, err_k = _element_errors(oe, oeT)
    return max(err_a, err_f, err_g, err_h, err_k) < criterion.tol
end

"""
    check_convergence(oe, oeT, weights, μ, F_max, params, criterion::MaxElementConvergence)

Weight-aware max-element convergence. Only checks elements with non-zero weight,
so elements marked "free" (W=0) are not required to match the target.
"""
function check_convergence(
    oe::ModEq{T1},
    oeT::ModEq{T2},
    weights::QLawWeights{T3},
    μ::Number,
    F_max::Number,
    params::QLawParameters,
    criterion::MaxElementConvergence,
) where {T1<:Number,T2<:Number,T3<:Number}
    err_a, err_f, err_g, err_h, err_k = _element_errors(oe, oeT)

    max_err = zero(promote_type(T1, T2))

    # Only check errors for targeted elements (W > 0)
    (weights.Wa > 0) && (max_err = max(max_err, err_a))
    (weights.Wf > 0) && (max_err = max(max_err, err_f))
    (weights.Wg > 0) && (max_err = max(max_err, err_g))
    (weights.Wh > 0) && (max_err = max(max_err, err_h))
    (weights.Wk > 0) && (max_err = max(max_err, err_k))

    return max_err < criterion.tol
end

"""
    check_convergence(oe, oeT, weights, μ, F_max, params)

Dispatch convergence check based on criterion stored in `params.convergence_criterion`.
"""
function check_convergence(
    oe::ModEq,
    oeT::ModEq,
    weights::QLawWeights,
    μ::Number,
    F_max::Number,
    params::QLawParameters,
)
    return check_convergence(
        oe,
        oeT,
        weights,
        μ,
        F_max,
        params,
        params.convergence_criterion,
    )
end

# Legacy: keyword tol version (backward compatibility with tests)
function check_convergence(oe::ModEq, oeT::ModEq; tol::Number = 0.05)
    return check_convergence(oe, oeT, SummedErrorConvergence(tol))
end

# Legacy: positional tol version (backward compatibility with old api.jl calls)
function check_convergence(oe::ModEq, oeT::ModEq, tol::Number)
    return check_convergence(oe, oeT, SummedErrorConvergence(tol))
end

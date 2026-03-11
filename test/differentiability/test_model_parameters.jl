const _μ = 398600.4418
const _JD = 2451545.0
const _t = 0.0

const _kep0 = Keplerian(6778.0, 0.001, deg2rad(28.5), 0.0, 0.0, 0.0)
const _kepT = Keplerian(42164.0, 0.001, 0.0, 0.0, 0.0, 0.0)
const _oe0 = ModEq(_kep0, _μ)
const _oeT = ModEq(_kepT, _μ)

const _spacecraft = QLawSpacecraft(500.0, 1000.0, 1.0, 3000.0)
const _weights = QLawWeights(1.0)
const _params = QLawParameters(; effectivity_search=GridSearch(), n_search_points=20)

const _F_max = max_thrust_acceleration(_spacecraft, 1000.0, 6778.0)
const _a0 = get_sma(_oe0)
const _aT = get_sma(_oeT)

const _oe_vec = SVector{5,Float64}(_a0, _oe0.f, _oe0.g, _oe0.h, _oe0.k)
const _oeT_vec = SVector{5,Float64}(_aT, _oeT.f, _oeT.g, _oeT.h, _oeT.k)
const _W_vec = SVector{5,Float64}(1.0, 1.0, 1.0, 1.0, 1.0)
const _max_rates = compute_max_rates_analytical(
    _a0, _oe0.f, _oe0.g, _oe0.h, _oe0.k, _μ, _F_max
)

const _dynamics_model = CentralBodyDynamicsModel(KeplerianGravityAstroModel(; μ=_μ), ())

const _ps = ComponentVector(; μ=_μ, JD=_JD)

const _prob = qlaw_problem(
    _oe0,
    _oeT,
    (0.0, 365.25 * 86400.0),
    _μ,
    _spacecraft;
    weights=_weights,
    qlaw_params=_params,
    dynamics_model=_dynamics_model,
)

const _u0 = SVector{7,Float64}(_oe0.p, _oe0.f, _oe0.g, _oe0.h, _oe0.k, _oe0.L, 1000.0)

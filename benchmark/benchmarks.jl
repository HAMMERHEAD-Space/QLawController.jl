using QLawController
using AstroCoords
using AstroForceModels
using BenchmarkTools
using ComponentArrays
using SatelliteToolboxGravityModels
using SatelliteToolboxTransformations
using SpaceIndices
using StaticArrays

const SUITE = BenchmarkGroup()

SUITE["core"] = BenchmarkGroup(["q-law math"])
SUITE["dynamics"] = BenchmarkGroup(["equations of motion"])
SUITE["activation"] = BenchmarkGroup(["utilities"])

# ---------------------
# Common state and parameters
# ---------------------
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

# Sun position for shadow benchmarks
const _sun_pos = SVector{3,Float64}(1.495978707e8, 0.0, 0.0)

# ---------------------
# Core Q-Law math benchmarks
# ---------------------
SUITE["core"]["compute_max_rates_analytical"] = @benchmarkable compute_max_rates_analytical(
    $_a0, $(_oe0.f), $(_oe0.g), $(_oe0.h), $(_oe0.k), $_μ, $_F_max
)

SUITE["core"]["equinoctial_gve_partials"] = @benchmarkable equinoctial_gve_partials(
    $_oe0, $_μ
)

SUITE["core"]["compute_Q_from_vec_with_rates"] = @benchmarkable compute_Q_from_vec_with_rates(
    $_oe_vec, $_oeT_vec, $_W_vec, $_max_rates, 1.0, 6578.0
)

SUITE["core"]["compute_dQ_doe_analytical"] = @benchmarkable compute_dQ_doe_analytical(
    $_oe_vec, $_oeT_vec, $_W_vec, $_max_rates, 1.0, 6578.0
)

SUITE["core"]["compute_Qdot_coefficients"] = @benchmarkable compute_Qdot_coefficients(
    $_oe0, $_oeT, $_weights, $_μ, $_F_max, $_params
)

SUITE["core"]["compute_thrust_direction"] = @benchmarkable compute_thrust_direction(
    $_oe0, $_oeT, $_weights, $_μ, $_F_max, $_params
)

SUITE["core"]["compute_thrust_and_effectivity"] = @benchmarkable compute_thrust_and_effectivity(
    $_oe0, $_oeT, $_weights, $_μ, $_F_max, $_params
)

SUITE["core"]["compute_Q"] = @benchmarkable compute_Q(
    $_oe0, $_oeT, $_weights, $_μ, $_F_max, $_params
)

# ---------------------
# Dynamics benchmarks (Keplerian)
# ---------------------
SUITE["dynamics"]["qlaw_eom_keplerian"] = @benchmarkable qlaw_eom($_u0, $_ps, $_t, $_prob)

# ---------------------
# Dynamics benchmarks (High-Fidelity)
# ---------------------
SpaceIndices.init()
const _eop_data = fetch_iers_eop()
const _grav_coeffs = GravityModels.load(IcgemFile, fetch_icgem_file(:EGM96))

const _hifi_grav = GravityHarmonicsAstroModel(;
    gravity_model=_grav_coeffs,
    eop_data=_eop_data,
    degree=36,
    order=36,
    P=MMatrix{37,37,Float64}(zeros(37, 37)),
    dP=MMatrix{37,37,Float64}(zeros(37, 37)),
)
const _hifi_moon = ThirdBodyModel(; body=MoonBody(), eop_data=_eop_data)
const _hifi_sun = ThirdBodyModel(; body=SunBody(), eop_data=_eop_data)
const _hifi_srp = SRPAstroModel(;
    satellite_srp_model=CannonballFixedSRP(1.784, 1000.0, 1.3),
    sun_data=_hifi_sun,
    eop_data=_eop_data,
    shadow_model=Conical(),
)
const _hifi_drag = DragAstroModel(;
    satellite_drag_model=CannonballFixedDrag(1.784, 1000.0, 2.2),
    atmosphere_model=ExpAtmo(),
    eop_data=_eop_data,
)
const _hifi_dynamics = CentralBodyDynamicsModel(
    _hifi_grav, (_hifi_moon, _hifi_sun, _hifi_srp, _hifi_drag)
)
const _hifi_prob = qlaw_problem(
    _oe0,
    _oeT,
    (0.0, 365.25 * 86400.0),
    _μ,
    _spacecraft;
    weights=_weights,
    qlaw_params=_params,
    dynamics_model=_hifi_dynamics,
    sun_model=_hifi_sun,
    JD0=Float64(_JD),
)
const _hifi_ps = ComponentVector(; μ=_μ, JD=_JD)

SUITE["dynamics"]["qlaw_eom_hifi"] = @benchmarkable qlaw_eom(
    $_u0, $_hifi_ps, $_t, $_hifi_prob
)

# ---------------------
# Utility benchmarks
# ---------------------
SUITE["activation"]["effectivity_activation"] = @benchmarkable effectivity_activation(
    0.5, -0.01, 1e-4
)
SUITE["activation"]["thrust_direction_to_rtn"] = @benchmarkable thrust_direction_to_rtn(
    0.5, 0.1
)
SUITE["activation"]["compute_sunlight_fraction"] = @benchmarkable compute_sunlight_fraction(
    $_oe0, $_μ, $_sun_pos, Conical()
)
SUITE["activation"]["apply_frame_rotation"] = @benchmarkable apply_frame_rotation(
    SVector{3,Float64}(6778.0, 0.0, 0.0), 0.1
)

# ---------------------
# Tune and cache
# ---------------------
paramspath = joinpath(dirname(@__FILE__), "params.json")

if isfile(paramspath)
    loadparams!(SUITE, BenchmarkTools.load(paramspath)[1], :evals)
else
    tune!(SUITE)
    BenchmarkTools.save(paramspath, BenchmarkTools.params(SUITE))
end

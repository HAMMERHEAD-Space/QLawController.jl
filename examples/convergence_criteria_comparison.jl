# =============================================================================
# Convergence Criteria Comparison Example
#
# Compares two convergence criteria for Q-Law weight optimization:
#   1. SummedErrorConvergence (default) - summed normalized orbital element errors
#   2. VargaConvergence - Q-function based criterion (Varga Eq. 35)
#
# For each criterion, a BBO (Differential Evolution) global optimization is run
# to find optimal weights and effectivity threshold that minimize transfer time.
#
# To run this example, first activate the examples environment:
#   using Pkg
#   Pkg.activate("examples")
#   Pkg.instantiate()
# =============================================================================

using QLaw

using AstroCoords
using AstroForceModels
using SatelliteToolboxGravityModels
using SatelliteToolboxTransformations
using Optimization
using OptimizationBBO

# =============================================================================
# Problem Setup (LEO to GEO from Table 1)
# =============================================================================

μ = 398600.4418  # Earth gravitational parameter [km³/s²]

# Initial orbit (LEO - ~500 km altitude, Cape Canaveral inclination)
a0 = 6878.0              # Semi-major axis [km]
e0 = 0.0                 # Eccentricity
i0 = deg2rad(28.5)       # Inclination [rad]

# Target orbit (GEO)
aT = 42164.0             # Semi-major axis [km]
eT = 0.0                 # Eccentricity
iT = 0.0                 # Inclination [rad]

kep0 = Keplerian(a0, e0, i0, 0.0, 0.0, 0.0)
kepT = Keplerian(aT, eT, iT, 0.0, 0.0, 0.0)

oe0 = ModEq(kep0, μ)
oeT = ModEq(kepT, μ)

# Spacecraft (from Table 1)
Tmax = 1.445             # Maximum thrust [N]
Isp = 1850.0             # Specific impulse [s]
m0 = 1000.0              # Initial mass [kg]
m_dry = 500.0            # Dry mass [kg]
spacecraft = QLawSpacecraft(m_dry, m0, Tmax, Isp)

# Time span
tspan = (0.0, 100.0 * 86400.0)  # 54 days

# =============================================================================
# Dynamics Model (J2 + Moon + Sun)
# =============================================================================

JD0 = date_to_jd(2024, 1, 5, 0, 0, 0)

eop_data = fetch_iers_eop()

egm96_file = fetch_icgem_file(:EGM96)
gravity_coeffs = GravityModels.load(IcgemFile, egm96_file)

gravity_model = GravityHarmonicsAstroModel(;
    gravity_model = gravity_coeffs,
    eop_data = eop_data,
    degree = 2,
    order = 0
)

moon_model = ThirdBodyModel(; body=MoonBody(), eop_data=eop_data)
sun_model = ThirdBodyModel(; body=SunBody(), eop_data=eop_data)

dynamics_model = CentralBodyDynamicsModel(gravity_model, (moon_model, sun_model))

# =============================================================================
# Fixed Q-Law Parameters (not optimized)
# =============================================================================

Wp = 1.0
rp_min = 6378.0 + 200.0
η_smoothness = 1e-4

# Bounds for optimization (from paper):
# - Weights bounded [0.01, 1.0]
# - ηth bounded [-0.01, 0.3]
lb = [0.01, 0.01, 0.01, 0.01, 0.01, -0.01]
ub = [1.0, 1.0, 1.0, 1.0, 1.0, 0.3]

# Problem parameters tuple
p_common = (oe0, oeT, tspan, μ, spacecraft, dynamics_model, sun_model, JD0)


# =============================================================================
# Objective Functions
#
# One for each convergence criterion. Each creates a QLawParameters with the
# appropriate criterion, then solves and returns transfer time (or penalty).
# =============================================================================

function objective_summed(x, p)
    oe0, oeT, tspan, μ, spacecraft, dynamics_model, sun_model, JD0 = p

    weights = QLawWeights(x[1], x[2], x[3], x[4], x[5])

    params = QLawParameters(;
        Wp = Wp,
        rp_min = rp_min,
        η_threshold = x[6],
        η_smoothness = η_smoothness,
        effectivity_type = :absolute,
        n_search_points = 50,
        convergence_criterion = SummedErrorConvergence(0.05)
    )

    prob = qlaw_problem(oe0, oeT, tspan, μ, spacecraft;
                        weights = weights,
                        qlaw_params = params,
                        dynamics_model = dynamics_model,
                        sun_model = sun_model,
                        JD0 = JD0)

    sol = solve(prob)

    if sol.converged
        return sol.elapsed_time / tspan[2]
    else
        return 1e10
    end
end

function objective_varga(x, p)
    oe0, oeT, tspan, μ, spacecraft, dynamics_model, sun_model, JD0 = p

    weights = QLawWeights(x[1], x[2], x[3], x[4], x[5])

    params = QLawParameters(;
        Wp = Wp,
        rp_min = rp_min,
        η_threshold = x[6],
        η_smoothness = η_smoothness,
        effectivity_type = :absolute,
        n_search_points = 50,
        convergence_criterion = VargaConvergence(1.0)
    )

    prob = qlaw_problem(oe0, oeT, tspan, μ, spacecraft;
                        weights = weights,
                        qlaw_params = params,
                        dynamics_model = dynamics_model,
                        sun_model = sun_model,
                        JD0 = JD0)

    sol = solve(prob)

    if sol.converged
        return sol.elapsed_time / tspan[2]
    else
        return 1e10
    end
end

# =============================================================================
# BBO Optimization: SummedErrorConvergence
# =============================================================================

println("=" ^ 70)
println("CRITERION 1: SummedErrorConvergence (tol=0.05)")
println("  Converges when: Σ |oe_i - oeT_i| / oeT_i < tol")
println("=" ^ 70)
println("Running BBO optimization...")
println()

x0 = lb .+ (ub .- lb) .* 0.5

opt_f_summed = OptimizationFunction(objective_summed)
opt_prob_summed = OptimizationProblem(opt_f_summed, x0, p_common; lb=lb, ub=ub)

@time sol_summed = Optimization.solve(opt_prob_summed,
    BBO_adaptive_de_rand_1_bin_radiuslimited();
    maxiters=500, maxtime=600.0)

println("\nSummedError BBO Results:")
println("  Optimal weights: Wa=$(round(sol_summed.u[1], digits=4)), " *
        "Wf=$(round(sol_summed.u[2], digits=4)), Wg=$(round(sol_summed.u[3], digits=4)), " *
        "Wh=$(round(sol_summed.u[4], digits=4)), Wk=$(round(sol_summed.u[5], digits=4))")
println("  Optimal ηth: $(round(sol_summed.u[6], digits=4))")
println("  Objective value: $(round(sol_summed.objective, digits=6))")

# Verify and extract final result
weights_summed = QLawWeights(sol_summed.u[1], sol_summed.u[2], sol_summed.u[3],
                              sol_summed.u[4], sol_summed.u[5])
params_summed = QLawParameters(;
    Wp=Wp, rp_min=rp_min, η_threshold=sol_summed.u[6],
    η_smoothness=η_smoothness, effectivity_type=:absolute, n_search_points=50,
    convergence_criterion=SummedErrorConvergence(0.05))
prob_summed = qlaw_problem(oe0, oeT, tspan, μ, spacecraft;
    weights=weights_summed, qlaw_params=params_summed,
    dynamics_model=dynamics_model, sun_model=sun_model, JD0=JD0)
result_summed = solve(prob_summed)

println("  Transfer time: $(round(result_summed.elapsed_time / 86400.0, digits=2)) days")
println("  Final mass: $(round(result_summed.final_mass, digits=2)) kg")
println("  Converged: $(result_summed.converged)")

# =============================================================================
# BBO Optimization: VargaConvergence
# =============================================================================

println()
println("=" ^ 70)
println("CRITERION 2: VargaConvergence (Rc=1.0)")
println("  Converges when: Q * μ/aT³ < Rc * √(Σ W_oe)  (Varga Eq. 35, timescale-normalized)")
println("=" ^ 70)
println("Running BBO optimization...")
println()

opt_f_varga = OptimizationFunction(objective_varga)
opt_prob_varga = OptimizationProblem(opt_f_varga, x0, p_common; lb=lb, ub=ub)

@time sol_varga = Optimization.solve(opt_prob_varga,
    BBO_adaptive_de_rand_1_bin_radiuslimited();
    maxiters=500, maxtime=600.0)

println("\nVarga BBO Results:")
println("  Optimal weights: Wa=$(round(sol_varga.u[1], digits=4)), " *
        "Wf=$(round(sol_varga.u[2], digits=4)), Wg=$(round(sol_varga.u[3], digits=4)), " *
        "Wh=$(round(sol_varga.u[4], digits=4)), Wk=$(round(sol_varga.u[5], digits=4))")
println("  Optimal ηth: $(round(sol_varga.u[6], digits=4))")
println("  Objective value: $(round(sol_varga.objective, digits=6))")

# Verify and extract final result
weights_varga = QLawWeights(sol_varga.u[1], sol_varga.u[2], sol_varga.u[3],
                             sol_varga.u[4], sol_varga.u[5])
params_varga = QLawParameters(;
    Wp=Wp, rp_min=rp_min, η_threshold=sol_varga.u[6],
    η_smoothness=η_smoothness, effectivity_type=:absolute, n_search_points=50,
    convergence_criterion=VargaConvergence(1.0))
prob_varga = qlaw_problem(oe0, oeT, tspan, μ, spacecraft;
    weights=weights_varga, qlaw_params=params_varga,
    dynamics_model=dynamics_model, sun_model=sun_model, JD0=JD0)
result_varga = solve(prob_varga)

println("  Transfer time: $(round(result_varga.elapsed_time / 86400.0, digits=2)) days")
println("  Final mass: $(round(result_varga.final_mass, digits=2)) kg")
println("  Converged: $(result_varga.converged)")

# =============================================================================
# Cross-Evaluation
#
# Run each set of optimal weights under BOTH criteria to see how the choice
# of convergence criterion during optimization affects the actual transfer.
# =============================================================================

println()
println("=" ^ 70)
println("CROSS-EVALUATION")
println("=" ^ 70)

# Helper to evaluate a set of weights under both criteria
function evaluate_weights(name, weights, ηth)
    println("\n  --- $name ---")

    for (crit_name, criterion) in [
        ("SummedError(0.05)", SummedErrorConvergence(0.05)),
        ("Varga(Rc=1.0)",     VargaConvergence(1.0))
    ]
        params = QLawParameters(;
            Wp=Wp, rp_min=rp_min, η_threshold=ηth,
            η_smoothness=η_smoothness, effectivity_type=:absolute,
            n_search_points=50,
            convergence_criterion=criterion)

        prob = qlaw_problem(oe0, oeT, tspan, μ, spacecraft;
            weights=weights, qlaw_params=params,
            dynamics_model=dynamics_model, sun_model=sun_model, JD0=JD0)
        result = solve(prob)

        time_str = result.converged ?
            "$(round(result.elapsed_time / 86400.0, digits=2)) days" : "N/A (did not converge)"
        mass_str = result.converged ?
            "$(round(result.final_mass, digits=2)) kg" : "N/A"

        println("    $crit_name → Transfer: $time_str, Mass: $mass_str, Converged: $(result.converged)")
    end
end

evaluate_weights("Weights optimized with SummedError",
                  weights_summed, sol_summed.u[6])
evaluate_weights("Weights optimized with Varga",
                  weights_varga, sol_varga.u[6])

# =============================================================================
# Summary Table
# =============================================================================

println()
println("=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

a_T  = QLaw.get_sma(oeT)
e_T  = sqrt(oeT.f^2 + oeT.g^2)
i_T  = 2 * atan(sqrt(oeT.h^2 + oeT.k^2))

for (name, result, sol_opt) in [
    ("SummedError", result_summed, sol_summed),
    ("Varga",       result_varga,  sol_varga)
]
    w = sol_opt.u
    oe = result.final_oe
    a_f = QLaw.get_sma(oe)
    e_f = sqrt(oe.f^2 + oe.g^2)
    i_f = 2 * atan(sqrt(oe.h^2 + oe.k^2))
    t = result.converged ? "$(round(result.elapsed_time / 86400.0, digits=2)) days" : "N/A"
    m = result.converged ? "$(round(result.final_mass, digits=2)) kg" : "N/A"

    println()
    println("  $name")
    println("  ├─ Weights: Wa=$(round(w[1],digits=3)), Wf=$(round(w[2],digits=3)), " *
            "Wg=$(round(w[3],digits=3)), Wh=$(round(w[4],digits=3)), Wk=$(round(w[5],digits=3))")
    println("  ├─ ηth:     $(round(w[6], digits=3))")
    println("  ├─ Time:    $t")
    println("  ├─ Mass:    $m")
    println("  ├─ Final a: $(round(a_f, digits=2)) km   (target: $(round(a_T, digits=2)) km,  Δ = $(round(a_f - a_T, digits=2)) km)")
    println("  ├─ Final e: $(round(e_f, digits=6))        (target: $(round(e_T, digits=6)),  Δ = $(round(e_f - e_T, sigdigits=3)))")
    println("  └─ Final i: $(round(rad2deg(i_f), digits=4))°       (target: $(round(rad2deg(i_T), digits=4))°,  Δ = $(round(rad2deg(i_f - i_T), digits=4))°)")
end
println()
println("=" ^ 70)

# =============================================================================
# Q-Law Weight Optimization Example
#
# Based on: "Q-Law for Rapid Assessment of Low Thrust Cislunar Trajectories
#           via Automatic Differentiation" by Steffen, Falck, and Faller
#
# This example demonstrates optimizing Q-Law weights to minimize transfer time
# or maximize final mass, similar to the paper's approach.
#
# To run this example, first activate the examples environment:
#   using Pkg
#   Pkg.activate("examples")
#   Pkg.instantiate()
# =============================================================================

using QLaw

using AstroCoords
using AstroForceModels
using Distributions
using SatelliteToolboxGravityModels
using SatelliteToolboxTransformations
using Optimization
using OptimizationOptimJL
using OptimizationBBO
using ExperimentalDesign

# =============================================================================
# Problem Setup (LEO to GEO from Table 1)
# =============================================================================

μ = 398600.4418  # Earth gravitational parameter [km³/s²]

# Initial orbit (LEO - ~500 km altitude, Cape Canaveral inclination)
a0 = 6878.0              # Semi-major axis [km]
e0 = 0.0                 # Eccentricity (Table 1)
i0 = deg2rad(28.5)       # Inclination [rad] (Table 1: 28.5 deg)

# Target orbit (GEO)
aT = 42164.0             # Semi-major axis [km] (Table 1)
eT = 0.0                 # Eccentricity (Table 1)
iT = 0.0                 # Inclination [rad] (Table 1: 0 deg)

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

# Time span - paper uses 54 day constraint
tspan = (0.0, 54.0 * 86400.0)

# =============================================================================
# Dynamics Model (Matching Paper: Eq. 7)
#
# Δ = ΔJ2 + Δmoon + Δsun + ΔT
# =============================================================================

# Epoch - use January 5, 2024 00:00:00 UTC as reference (Table 1)
JD0 = date_to_jd(2024, 1, 5, 0, 0, 0)

# Fetch EOP (Earth Orientation Parameters) data
eop_data = fetch_iers_eop()

# Fetch gravity model (EGM96 - standard model with J2 and higher harmonics)
egm96_file = fetch_icgem_file(:EGM96)
gravity_coeffs = GravityModels.load(IcgemFile, egm96_file)

# Create gravity harmonics model (degree 2, order 0 = J2 only as in paper)
gravity_model = GravityHarmonicsAstroModel(;
    gravity_model = gravity_coeffs,
    eop_data = eop_data,
    degree = 2,   # J2 (degree 2)
    order = 0,     # Zonal only (order 0)
)

# Third body models (Moon and Sun)
moon_model = ThirdBodyModel(; body = MoonBody(), eop_data = eop_data)
sun_model = ThirdBodyModel(; body = SunBody(), eop_data = eop_data)

# Combined dynamics model: central body gravity (J2) + Moon + Sun third body
dynamics_model = CentralBodyDynamicsModel(gravity_model, (moon_model, sun_model))

# =============================================================================
# Fixed Q-Law Parameters (not optimized)
# =============================================================================

# These are fixed - only weights and η_threshold are optimized
Wp = 1.0
rp_min = 6378.0 + 200.0
η_smoothness = 1e-4

# =============================================================================
# Objective Function
#
# Optimize: 5 weights (Wa, Wf, Wg, Wh, Wk) + effectivity threshold (ηth)
# Minimize: transfer time + penalty for non-convergence
# =============================================================================

function objective(x, p)
    oe0, oeT, tspan, μ, spacecraft, dynamics_model, sun_model, JD0 = p

    # x = [Wa, Wf, Wg, Wh, Wk, ηth]
    weights = QLawWeights(x[1], x[2], x[3], x[4], x[5])

    params = QLawParameters(;
        Wp = Wp,
        rp_min = rp_min,
        η_threshold = x[6],
        η_smoothness = η_smoothness,
        effectivity_type = AbsoluteEffectivity(),
        n_search_points = 50,
    )

    prob = qlaw_problem(
        oe0,
        oeT,
        tspan,
        μ,
        spacecraft;
        weights = weights,
        qlaw_params = params,
        dynamics_model = dynamics_model,
        sun_model = sun_model,
        JD0 = JD0,
    )

    sol = solve(prob)  # Uses convergence_criterion from params (default: SummedErrorConvergence(0.05))

    if sol.converged
        return sol.elapsed_time / tspan[2]
    else
        return 1e10
    end
end

# Alternative objective: maximize final mass
function objective_mass(x, p)
    oe0, oeT, tspan, μ, spacecraft, dynamics_model, sun_model, JD0 = p

    # x = [Wa, Wf, Wg, Wh, Wk, ηth]
    weights = QLawWeights(x[1], x[2], x[3], x[4], x[5])

    params = QLawParameters(;
        Wp = Wp,
        rp_min = rp_min,
        η_threshold = x[6],
        η_smoothness = η_smoothness,
        effectivity_type = AbsoluteEffectivity(),
        n_search_points = 50,
    )

    prob = qlaw_problem(
        oe0,
        oeT,
        tspan,
        μ,
        spacecraft;
        weights = weights,
        qlaw_params = params,
        dynamics_model = dynamics_model,
        sun_model = sun_model,
        JD0 = JD0,
    )

    sol = solve(prob)  # Uses convergence_criterion from params (default: SummedErrorConvergence(0.05))

    if sol.converged
        return -sol.final_mass
    else
        return 1e10
    end
end

# =============================================================================
# Setup Optimization Problem
# =============================================================================

# Problem parameters tuple
p = (oe0, oeT, tspan, μ, spacecraft, dynamics_model, sun_model, JD0)

# Bounds (from paper):
# - Weights bounded [0.01, 1.0]
# - ηth bounded [-0.01, 0.3]
lb = [0.01, 0.01, 0.01, 0.01, 0.01, -0.01]
ub = [1.0, 1.0, 1.0, 1.0, 1.0, 0.3]

# =============================================================================
# Method 1: BlackBoxOptim / Differential Evolution (Global Search)
# 
# Note: This is slower but explores the search space more thoroughly.
# BBO handles bounds natively. No AD needed (gradient-free).
# =============================================================================

println("=" ^ 70)
println("METHOD 1: BlackBoxOptim (Differential Evolution)")
println("=" ^ 70)
println("Starting optimization (this may take 10+ minutes)...")
println()

opt_f_bbo = OptimizationFunction(objective)
# Initial guess: midpoint of bounds
x0 = lb .+ (ub .- lb) .* 0.5
opt_prob_bbo = OptimizationProblem(opt_f_bbo, x0, p; lb = lb, ub = ub)

@time sol_bbo = Optimization.solve(
    opt_prob_bbo,
    BBO_adaptive_de_rand_1_bin_radiuslimited();
    maxiters = 500,
    maxtime = 600.0,
)

println("\nBlackBoxOptim Results:")
println(
    "  Optimal weights: Wa=$(round(sol_bbo.u[1], digits=4)), " *
    "Wf=$(round(sol_bbo.u[2], digits=4)), Wg=$(round(sol_bbo.u[3], digits=4)), " *
    "Wh=$(round(sol_bbo.u[4], digits=4)), Wk=$(round(sol_bbo.u[5], digits=4))",
)
println("  Optimal ηth: $(round(sol_bbo.u[6], digits=4))")
println("  Objective value: $(round(sol_bbo.objective, digits=4))")

# Verify result
weights_bbo =
    QLawWeights(sol_bbo.u[1], sol_bbo.u[2], sol_bbo.u[3], sol_bbo.u[4], sol_bbo.u[5])
params_bbo = QLawParameters(;
    Wp = Wp,
    rp_min = rp_min,
    η_threshold = sol_bbo.u[6],
    η_smoothness = η_smoothness,
    effectivity_type = AbsoluteEffectivity(),
    n_search_points = 50,
)
prob_bbo = qlaw_problem(
    oe0,
    oeT,
    tspan,
    μ,
    spacecraft;
    weights = weights_bbo,
    qlaw_params = params_bbo,
    dynamics_model = dynamics_model,
    sun_model = sun_model,
    JD0 = JD0,
)
result_bbo = solve(prob_bbo)
println("  Transfer time: $(round(result_bbo.elapsed_time / 86400.0, digits=2)) days")
println("  Final mass: $(round(result_bbo.final_mass, digits=2)) kg")
println("  Converged: $(result_bbo.converged)")

# =============================================================================
# Comparison with Paper Values (Table 3 - with effectivity)
# =============================================================================

println()
println("=" ^ 70)
println("COMPARISON WITH PAPER VALUES (Table 3 - with effectivity)")
println("=" ^ 70)

# Table 3 reference values from paper
paper_results = [
    ("DOE (Table 3)", QLawWeights(0.0120, 0.5846, 0.1189, 0.9475, 0.3578), 0.247),
    ("IPOPT (Table 3)", QLawWeights(0.9890, 0.9890, 0.9890, 0.9890, 0.9890), 0.004),
    ("GA (Table 3)", QLawWeights(0.0902, 0.8902, 0.9137, 0.9686, 0.9882), 0.266),
    ("BBO (this run)", weights_bbo, sol_bbo.u[6]),
]

println("\nResults comparison:")
println("-" ^ 85)
println("Method              | ηth   | Transfer Time (days) | Final Mass (kg) | Converged")
println("-" ^ 85)

for (name, weights, ηth) in paper_results
    # Create params with the correct effectivity threshold
    params_test = QLawParameters(;
        Wp = Wp,
        rp_min = rp_min,
        η_threshold = ηth,
        η_smoothness = η_smoothness,
        effectivity_type = AbsoluteEffectivity(),
        n_search_points = 50,
    )

    prob_test = qlaw_problem(
        oe0,
        oeT,
        tspan,
        μ,
        spacecraft;
        weights = weights,
        qlaw_params = params_test,
        dynamics_model = dynamics_model,
        sun_model = sun_model,
        JD0 = JD0,
    )
    result = solve(prob_test)

    time_str =
        result.converged ? string(round(result.elapsed_time / 86400.0, digits = 2)) : "N/A"
    mass_str = result.converged ? string(round(result.final_mass, digits = 2)) : "N/A"

    println(
        "$(rpad(name, 19)) | $(lpad(string(round(ηth, digits=3)), 5)) | $(lpad(time_str, 20)) | $(lpad(mass_str, 15)) | $(result.converged)",
    )
end
println("-" ^ 85)

# =============================================================================
# Method 2: Latin Hypercube DOE + Local Refinement (Fminbox + Nelder-Mead)
# =============================================================================

println()
println("=" ^ 70)
println("METHOD 2: Random DOE + Local Refinement")
println("=" ^ 70)
println("Sampling design space with RandomDesign...")

# Generate random DOE samples (6 dimensions: 5 weights + ηth)
n_samples = 20
samples = random_design!(
    (
        Uniform(lb[1], ub[1]),   # Wa
        Uniform(lb[2], ub[2]),   # Wf
        Uniform(lb[3], ub[3]),   # Wg
        Uniform(lb[4], ub[4]),   # Wh
        Uniform(lb[5], ub[5]),   # Wk
        Uniform(lb[6], ub[6]),
    ),  # ηth
    n_samples,
)

# Evaluate all samples
println("Evaluating $n_samples samples...")
best_obj = Inf
best_idx = 0
results_doe = []

for i = 1:n_samples
    obj_val = objective(samples[i, :], p)
    push!(results_doe, obj_val)
    if obj_val < best_obj
        best_obj = obj_val
        best_idx = i
    end
    if i % 5 == 0
        println(
            "  Evaluated $i / $n_samples samples (best so far: $(round(best_obj, digits=4)))",
        )
    end
end

println("\nBest DOE sample: $(round(best_obj, digits=4))")
println(
    "  Weights: Wa=$(round(samples[best_idx, 1], digits=4)), " *
    "Wf=$(round(samples[best_idx, 2], digits=4)), Wg=$(round(samples[best_idx, 3], digits=4)), " *
    "Wh=$(round(samples[best_idx, 4], digits=4)), Wk=$(round(samples[best_idx, 5], digits=4))",
)
println("  ηth: $(round(samples[best_idx, 6], digits=4))")

# Local refinement from best DOE sample using SAMIN (Simulated Annealing).
# SAMIN natively respects box constraints and is gradient-free.
println("\nRefining with SAMIN from best DOE point...")
x0_refined = samples[best_idx, :]
opt_f_refined = OptimizationFunction(objective)
opt_prob_refined = OptimizationProblem(opt_f_refined, x0_refined, p; lb = lb, ub = ub)
@time sol_refined = Optimization.solve(opt_prob_refined, SAMIN(); maxiters = 100)

println("\nDOE + Refinement Results:")
println(
    "  Optimal weights: Wa=$(round(sol_refined.u[1], digits=4)), " *
    "Wf=$(round(sol_refined.u[2], digits=4)), Wg=$(round(sol_refined.u[3], digits=4)), " *
    "Wh=$(round(sol_refined.u[4], digits=4)), Wk=$(round(sol_refined.u[5], digits=4))",
)
println("  Optimal ηth: $(round(sol_refined.u[6], digits=4))")
println("  Objective value: $(round(sol_refined.objective, digits=4))")

# Verify result
weights_refined = QLawWeights(
    sol_refined.u[1],
    sol_refined.u[2],
    sol_refined.u[3],
    sol_refined.u[4],
    sol_refined.u[5],
)
params_refined = QLawParameters(;
    Wp = Wp,
    rp_min = rp_min,
    η_threshold = sol_refined.u[6],
    η_smoothness = η_smoothness,
    effectivity_type = AbsoluteEffectivity(),
    n_search_points = 50,
)
prob_refined = qlaw_problem(
    oe0,
    oeT,
    tspan,
    μ,
    spacecraft;
    weights = weights_refined,
    qlaw_params = params_refined,
    dynamics_model = dynamics_model,
    sun_model = sun_model,
    JD0 = JD0,
)
result_refined = solve(prob_refined)
println("  Transfer time: $(round(result_refined.elapsed_time / 86400.0, digits=2)) days")
println("  Final mass: $(round(result_refined.final_mass, digits=2)) kg")
println("  Converged: $(result_refined.converged)")

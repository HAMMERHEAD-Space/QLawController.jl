# =============================================================================
# LEO to GEO Transfer Example
#
# Based on: "Q-Law for Rapid Assessment of Low Thrust Cislunar Trajectories
#           via Automatic Differentiation" by Steffen, Falck, and Faller
#
# Table 1: Problem Parameters
# Table 2: Optimized Weight Values
#
# To run:
#   using Pkg
#   Pkg.activate("examples")
#   Pkg.instantiate()
#   include("examples/leo_to_geo.jl")
# =============================================================================

using QLawController

using AstroCoords
using AstroForceModels
using AstroForceModels: Position, Conical  # For eclipse computation in plots
using LinearAlgebra
using SatelliteToolboxGravityModels
using SatelliteToolboxTransformations
using Plots

# =============================================================================
# Problem Parameters (Table 1)
# =============================================================================

# Earth gravitational parameter [km³/s²]
μ = 398600.4418

# Initial orbit (LEO - ~500 km altitude, Cape Canaveral inclination)
a0 = 6878.0              # Semi-major axis [km]
e0 = 0.0                 # Eccentricity (Table 1)
i0 = deg2rad(28.5)       # Inclination [rad] (Table 1: 28.5 deg)
Ω0 = 0.0                 # RAAN [rad]
ω0 = 0.0                 # Argument of periapsis [rad]
ν0 = 0.0                 # True anomaly [rad]

# Target orbit (GEO)
aT = 42164.0             # Semi-major axis [km] (Table 1)
eT = 0.0                 # Eccentricity (Table 1)
iT = 0.0                 # Inclination [rad] (Table 1: 0 deg)
ΩT = 0.0                 # RAAN [rad]
ωT = 0.0                 # Argument of periapsis [rad]
νT = 0.0                 # True anomaly [rad]

# Spacecraft parameters (Table 1)
Tmax = 1.445             # Maximum thrust [N]
Isp = 1850.0             # Specific impulse [s]
m0 = 1000.0              # Initial mass [kg]
m_dry = 500.0            # Dry mass [kg]

# Time span - paper uses 54 day constraint
tspan = (0.0, 54.0 * 86400.0)  # [s]

# Epoch - use January 5, 2024 00:00:00 UTC as reference (Table 1)
JD0 = date_to_jd(2024, 1, 5, 0, 0, 0)  # Julian date at t=0

# =============================================================================
# Create Orbital Elements
# =============================================================================

kep0 = Keplerian(a0, e0, i0, Ω0, ω0, ν0)
kepT = Keplerian(aT, eT, iT, ΩT, ωT, νT)

oe0 = ModEq(kep0, μ)
oeT = ModEq(kepT, μ)

# =============================================================================
# Create Spacecraft
# =============================================================================

spacecraft = QLawSpacecraft(m_dry, m0, Tmax, Isp)

println("Spacecraft Parameters:")
println("  Dry mass: $(m_dry) kg")
println("  Wet mass: $(m0) kg")
println("  Thrust: $(Tmax) N")
println("  Isp: $(Isp) s")
println("  Exhaust velocity: $(exhaust_velocity(spacecraft)) km/s")
println()

# =============================================================================
# Q-Law Weights from Paper (Tables 2 and 3)
# 
# Table 3 contains DOE-optimized weights for use WITH effectivity (coasting).
# Table 2 contains weights optimized WITHOUT effectivity (constant thrust).
# Since we use effectivity-based coasting, we use Table 3 weights.
# =============================================================================

# Table 3: DOE weights WITH effectivity (ηth = 0.247)
weights_doe = QLawWeights(0.0120, 0.5846, 0.1189, 0.9475, 0.3578)
ηth_doe = 0.247

# Table 3: IPOPT weights WITH effectivity (ηth = 0.004)
weights_ipopt = QLawWeights(0.9890, 0.9890, 0.9890, 0.9890, 0.9890)
ηth_ipopt = 0.004

# Table 3: GA weights WITH effectivity (ηth = 0.266)
weights_ga = QLawWeights(0.0902, 0.8902, 0.9137, 0.9686, 0.9882)
ηth_ga = 0.266

# Default weights (all equal)
weights_default = QLawWeights()

println("Q-Law Weights (Table 3 - with effectivity optimization):")
println(
    "  DOE:    Wa=$(weights_doe.Wa), Wf=$(weights_doe.Wf), Wg=$(weights_doe.Wg), Wh=$(weights_doe.Wh), Wk=$(weights_doe.Wk), ηth=$(ηth_doe)",
)
println(
    "  IPOPT:  Wa=$(weights_ipopt.Wa), Wf=$(weights_ipopt.Wf), Wg=$(weights_ipopt.Wg), Wh=$(weights_ipopt.Wh), Wk=$(weights_ipopt.Wk), ηth=$(ηth_ipopt)",
)
println(
    "  GA:     Wa=$(weights_ga.Wa), Wf=$(weights_ga.Wf), Wg=$(weights_ga.Wg), Wh=$(weights_ga.Wh), Wk=$(weights_ga.Wk), ηth=$(ηth_ga)",
)
println()

# =============================================================================
# Q-Law Parameters
# 
# Using weights and η_threshold from Table 3 (DOE case).
# Coasting occurs when effectivity η < η_threshold.
# =============================================================================

params = QLawParameters(;
    Wp = 1.0,                    # Periapsis penalty weight
    rp_min = 6378.0 + 200.0,     # Minimum periapsis (200 km altitude)
    η_threshold = ηth_doe,       # Effectivity threshold from Table 3 DOE
    η_smoothness = 1e-4,         # Activation smoothness
    effectivity_type = AbsoluteEffectivity(),
    n_search_points = 50,
)

println("Q-Law Parameters:")
println("  Periapsis penalty weight: $(params.Wp)")
println("  Minimum periapsis: $(params.rp_min) km")
println("  Effectivity threshold: $(params.η_threshold)")
println()

# =============================================================================
# Dynamics Model (Matching Paper: Eq. 7)
#
# Δ = ΔJ2 + Δmoon + Δsun + ΔT
#
# - J2 zonal harmonics (Earth oblateness)
# - Third body gravity from Moon
# - Third body gravity from Sun
# =============================================================================

println("Setting up dynamics model...")

# Fetch EOP (Earth Orientation Parameters) data
println("  Fetching IERS EOP data...")
eop_data = fetch_iers_eop()

# Fetch gravity model (EGM96 - standard model with J2 and higher harmonics)
println("  Fetching EGM96 gravity model...")
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

println("Dynamics Model (Paper Eq. 7):")
println("  Gravity: EGM96 (degree=2, order=0 for J2)")
println("  Third body: Moon")
println("  Third body: Sun")
println("  EOP data: IERS finals1980")
println()

# =============================================================================
# Create and Solve Problem
# =============================================================================

println("Creating Q-Law problem with DOE-optimized weights...")
prob = qlaw_problem(
    oe0,
    oeT,
    tspan,
    μ,
    spacecraft;
    weights = weights_doe,
    qlaw_params = params,
    dynamics_model = dynamics_model,
    sun_model = sun_model,
    JD0 = JD0,
)

println("Initial orbit:")
println("  Semi-major axis: $(QLawController.get_sma(prob.oe0)) km")
println("  Eccentricity: $(sqrt(prob.oe0.f^2 + prob.oe0.g^2))")
println("  Inclination: $(rad2deg(2*atan(sqrt(prob.oe0.h^2 + prob.oe0.k^2)))) deg")
println()

println("Target orbit:")
println("  Semi-major axis: $(QLawController.get_sma(prob.oeT)) km")
println("  Eccentricity: $(sqrt(prob.oeT.f^2 + prob.oeT.g^2))")
println("  Inclination: $(rad2deg(2*atan(sqrt(prob.oeT.h^2 + prob.oeT.k^2)))) deg")
println()

# Solve the problem
println("Solving Q-Law transfer...")
println("(This may take a few minutes for long transfers)")
println()

@time sol = solve(prob)  # Uses convergence_criterion from params (default: SummedErrorConvergence(0.05))

# =============================================================================
# Results
# =============================================================================

println()
println("=" ^ 60)
println("RESULTS")
println("=" ^ 60)

# Extract trajectory data from ODE solution
t = sol.trajectory.t ./ 86400.0  # Convert to days
states = sol.trajectory.u
a_history = [QLawController.get_sma(ModEq(u[1], u[2], u[3], u[4], u[5], u[6])) for u in states]
e_history = [sqrt(u[2]^2 + u[3]^2) for u in states]
i_history = [rad2deg(2*atan(sqrt(u[4]^2 + u[5]^2))) for u in states]
m_history = [u[7] for u in states]

println()
println("Trajectory Summary:")
println("  Transfer time: $(sol.elapsed_time / 86400.0) days")
println("  Total ΔV: $(sol.Δv_total) km/s")
println("  Propellant used: $(m0 - sol.final_mass) kg")
println("  Final mass: $(sol.final_mass) kg")
println()

println("Final Orbital Elements:")
println("  Semi-major axis: $(QLawController.get_sma(sol.final_oe)) km (target: $(aT) km)")
println("  Eccentricity: $(sqrt(sol.final_oe.f^2 + sol.final_oe.g^2)) (target: $(eT))")
println(
    "  Inclination: $(rad2deg(2*atan(sqrt(sol.final_oe.h^2 + sol.final_oe.k^2)))) deg (target: $(rad2deg(iT)) deg)",
)

# =============================================================================
# Plotting (Matching Paper Figures 9, 10, 11)
#
# Figure 9: States - a, e, i, TA vs time
# Figure 10: Controls - thrust accel, α, β, η vs time
# Figure 11: 2D orbit projections
# =============================================================================

println()
println("=" ^ 60)
println("GENERATING PLOTS (Paper Figures 9, 10, 11)")
println("=" ^ 60)

# Extract classical elements for Figure 9
# True anomaly: ν = L - (ω + Ω)
# Compute cumulative TA in revolutions (unwrapped)
ν_wrapped = [u[6] - atan(u[3], u[2]) for u in states]
ν_cumulative = zeros(length(ν_wrapped))
ν_cumulative[1] = ν_wrapped[1]
for i = 2:length(ν_wrapped)
    dν = ν_wrapped[i] - ν_wrapped[i-1]
    # Handle wraparound
    if dν < -π
        dν += 2π
    elseif dν > π
        dν -= 2π
    end
    ν_cumulative[i] = ν_cumulative[i-1] + dν
end
ν_revs = ν_cumulative ./ (2π)  # Convert to revolutions

# Target values for classical elements
aT_val = QLawController.get_sma(oeT)
eT_val = sqrt(oeT.f^2 + oeT.g^2)
iT_val = rad2deg(2*atan(sqrt(oeT.h^2 + oeT.k^2)))

# Compute control history (thrust direction at each saved state)
α_history = Float64[]
β_history = Float64[]
thrust_accel_history = Float64[]
effectivity_history = Float64[]
sunlight_history = Float64[]

# Shadow model for eclipse computation
shadow_model = Conical()

println("Computing control history for plots...")
for (i, u) in enumerate(states)
    oe_i = ModEq(u[1], u[2], u[3], u[4], u[5], u[6])
    m_i = u[7]
    r_i = QLawController.compute_radius(oe_i)
    t_i = sol.trajectory.t[i]  # Time in seconds

    # Maximum thrust acceleration at this state [km/s²]
    F_max_accel = max_thrust_acceleration(spacecraft, m_i, r_i)

    # Optimal thrust direction (uses params for Wp, rp_min, scaling)
    α_opt, β_opt, _ =
        QLawController.compute_thrust_direction(oe_i, oeT, weights_doe, μ, F_max_accel, params)

    # Compute effectivity to get actual throttle (uses params for all settings)
    η, _, _, _ = QLawController.compute_effectivity(oe_i, oeT, weights_doe, μ, F_max_accel, params)
    activation = QLawController.effectivity_activation(η, params.η_threshold, params.η_smoothness)

    # Compute sunlight fraction (eclipse)
    JD_i = JD0 + t_i / 86400.0
    sun_pos = sun_model(JD_i, Position())
    γ = QLawController.compute_sunlight_fraction(oe_i, μ, sun_pos, shadow_model)

    # Total throttle = effectivity * sunlight
    throttle = activation * γ

    push!(α_history, α_opt)
    push!(β_history, β_opt)
    push!(thrust_accel_history, throttle * F_max_accel * 1e6)  # mm/s² (actual thrust)
    push!(effectivity_history, η)
    push!(sunlight_history, γ)
end

# Convert Cartesian for orbit plots
x_history = Float64[]
y_history = Float64[]
z_history = Float64[]
for u in states
    oe_i = ModEq(u[1], u[2], u[3], u[4], u[5], u[6])
    cart = Cartesian(oe_i, μ)
    push!(x_history, cart.x)
    push!(y_history, cart.y)
    push!(z_history, cart.z)
end

# =============================================================================
# Figure 9: States of Best DOE Run (Paper Figure 9)
# 4 subplots: a, e, i, ν vs time with red dashed target lines
# =============================================================================
println("Generating Figure 9: States...")

p_a = plot(
    t,
    a_history ./ 1000,
    ylabel = "a [×10³ km]",
    label = false,
    linewidth = 1,
    color = :blue,
)
hline!([aT_val / 1000], linestyle = :dash, color = :red, linewidth = 2, label = false)

p_e = plot(t, e_history, ylabel = "e [-]", label = false, linewidth = 1, color = :blue)
hline!([eT_val], linestyle = :dash, color = :red, linewidth = 2, label = false)

p_i = plot(t, i_history, ylabel = "i [deg]", label = false, linewidth = 1, color = :blue)
hline!([iT_val], linestyle = :dash, color = :red, linewidth = 2, label = false)

p_nu = plot(
    t,
    ν_revs,
    xlabel = "Time [days]",
    ylabel = "TA [rev]",
    label = false,
    linewidth = 1,
    color = :blue,
)

fig9 = plot(
    p_a,
    p_e,
    p_i,
    p_nu,
    layout = (2, 2),
    size = (1000, 700),
    plot_title = "Figure 9: States (Classical Elements)",
)

savefig(fig9, "figure9_states.png")
println("  Saved: figure9_states.png")

# =============================================================================
# Figure 10: Controls of Best DOE Run (Paper Figure 10)
# 4 subplots: Thrust acceleration, α, β, effectivity vs time
# Shows coasts from effectivity cut-off and eclipses
# =============================================================================
println("Generating Figure 10: Controls...")

p_thrust = plot(
    t,
    thrust_accel_history,
    ylabel = "Thrust Accel\n[mm/s²]",
    label = false,
    linewidth = 0.5,
    color = :blue,
)

p_alpha = plot(
    t,
    rad2deg.(α_history),
    ylabel = "α* [deg]",
    label = false,
    linewidth = 0.5,
    color = :blue,
)

p_beta = plot(
    t,
    rad2deg.(β_history),
    ylabel = "β* [deg]",
    label = false,
    linewidth = 0.5,
    color = :blue,
)

p_eta = plot(
    t,
    effectivity_history,
    xlabel = "Time [days]",
    ylabel = "η [-]",
    label = false,
    linewidth = 0.5,
    color = :blue,
)
hline!([params.η_threshold], linestyle = :dash, color = :red, linewidth = 1, label = "ηₜₕ")

fig10 = plot(
    p_thrust,
    p_alpha,
    p_beta,
    p_eta,
    layout = (4, 1),
    size = (1000, 900),
    plot_title = "Figure 10: Controls of Best DOE Run",
)

savefig(fig10, "figure10_controls.png")
println("  Saved: figure10_controls.png")

# =============================================================================
# Figure 11: Shape of Orbit (Paper Figure 11)
# Three 2D projections: X-Y, Y-Z, X-Z planes
# =============================================================================
println("Generating Figure 11: Orbit Shape (2D projections)...")

# X-Y plane (equatorial view)
p_xy = plot(
    x_history ./ 1000,
    y_history ./ 1000,
    xlabel = "X [×10³ km]",
    ylabel = "Y [×10³ km]",
    linewidth = 0.3,
    color = :blue,
    legend = false,
    title = "X-Y Plane",
    aspect_ratio = :equal,
)
scatter!(
    [x_history[1] / 1000],
    [y_history[1] / 1000],
    markersize = 4,
    color = :green,
    label = "Start",
)
scatter!(
    [x_history[end] / 1000],
    [y_history[end] / 1000],
    markersize = 4,
    color = :red,
    label = "End",
)

# Y-Z plane
p_yz = plot(
    y_history ./ 1000,
    z_history ./ 1000,
    xlabel = "Y [×10³ km]",
    ylabel = "Z [×10³ km]",
    linewidth = 0.3,
    color = :blue,
    legend = false,
    title = "Y-Z Plane",
    aspect_ratio = :equal,
)
scatter!([y_history[1] / 1000], [z_history[1] / 1000], markersize = 4, color = :green)
scatter!([y_history[end] / 1000], [z_history[end] / 1000], markersize = 4, color = :red)

# X-Z plane
p_xz = plot(
    x_history ./ 1000,
    z_history ./ 1000,
    xlabel = "X [×10³ km]",
    ylabel = "Z [×10³ km]",
    linewidth = 0.3,
    color = :blue,
    legend = false,
    title = "X-Z Plane",
    aspect_ratio = :equal,
)
scatter!([x_history[1] / 1000], [z_history[1] / 1000], markersize = 4, color = :green)
scatter!([x_history[end] / 1000], [z_history[end] / 1000], markersize = 4, color = :red)

fig11 = plot(
    p_xy,
    p_yz,
    p_xz,
    layout = (1, 3),
    size = (1500, 500),
    plot_title = "Figure 11: Shape of Orbit",
)

savefig(fig11, "figure11_orbit_shape.png")
println("  Saved: figure11_orbit_shape.png")

# =============================================================================
# Additional plots for analysis
# =============================================================================

# Propellant consumption
p_mass = plot(
    t,
    m_history,
    xlabel = "Time [days]",
    ylabel = "Mass [kg]",
    label = "Spacecraft",
    linewidth = 1.5,
    color = :blue,
    title = "Propellant Consumption",
    size = (800, 400),
    legend = :topright,
)
hline!([m_dry], linestyle = :dash, color = :red, linewidth = 2, label = "Dry mass")
savefig(p_mass, "mass_history.png")
println("  Saved: mass_history.png")


# Display plots
display(fig9)
display(fig10)
display(fig11)

println()
println("All plots saved to current directory.")

# =============================================================================
# Optional: Compare with Other Weight Sets
# =============================================================================

println()
println("=" ^ 60)
println("COMPARISON WITH OTHER WEIGHT SETS")
println("=" ^ 60)

for (name, weights, ηth) in [
    ("Default", weights_default, 0.1),  # Default threshold
    ("IPOPT", weights_ipopt, ηth_ipopt),
    ("GA", weights_ga, ηth_ga),
]
    # Create params with the correct effectivity threshold for each weight set
    params_test = QLawParameters(;
        Wp = 1.0,
        rp_min = 6378.0 + 200.0,
        η_threshold = ηth,
        η_smoothness = 1e-4,
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

    println("\nSolving with $(name) weights (ηth=$(ηth))...")
    @time sol_test = solve(prob_test)

    println(
        "  $(name): ΔV = $(round(sol_test.Δv_total, digits=3)) km/s, " *
        "Time = $(round(sol_test.elapsed_time / 86400.0, digits=2)) days, " *
        "Converged = $(sol_test.converged)",
    )

    # Print final orbital elements
    final_a = QLawController.get_sma(sol_test.final_oe)
    final_e = sqrt(sol_test.final_oe.f^2 + sol_test.final_oe.g^2)
    final_i = rad2deg(2*atan(sqrt(sol_test.final_oe.h^2 + sol_test.final_oe.k^2)))
    println(
        "  Final OE: a=$(round(final_a, digits=1)) km (target: $(aT)), " *
        "e=$(round(final_e, digits=4)) (target: $(eT)), " *
        "i=$(round(final_i, digits=2))° (target: $(rad2deg(iT))°)",
    )
end

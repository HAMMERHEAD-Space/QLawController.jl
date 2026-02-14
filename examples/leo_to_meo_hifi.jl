# =============================================================================
# High-Fidelity LEO to Inclined MEO Transfer
#
# Demonstrates Q-Law with a full perturbation environment:
#   - 36×36 gravity harmonics (EGM96)
#   - Third-body Sun and Moon
#   - Solar Radiation Pressure (cannonball model, conical shadow)
#   - Atmospheric drag (exponential atmosphere, cannonball model)
#   - Eclipse/shadow effects on thrust and SRP
#
# Transfer: Equatorial LEO (400 km) → Inclined MEO (20,200 km, 55°)
#           Typical of GPS constellation insertion
#
# To run:
#   using Pkg
#   Pkg.activate("examples")
#   Pkg.instantiate()
#   include("examples/leo_to_meo_hifi.jl")
# =============================================================================

using QLawController

using AstroCoords
using AstroForceModels
using AstroForceModels: Position, Conical
using LinearAlgebra
using SatelliteToolboxGravityModels
using SatelliteToolboxTransformations
using Plots

# =============================================================================
# Problem Setup
# =============================================================================

μ = 398600.4418  # Earth gravitational parameter [km³/s²]

# Initial orbit: equatorial LEO at 400 km altitude
a0 = 6378.137 + 400.0    # Semi-major axis [km]
e0 = 0.001                # Near-circular
i0 = deg2rad(0.0)         # Equatorial
Ω0 = 0.0
ω0 = 0.0
ν0 = 0.0

# Target orbit: GPS-like MEO
aT = 26559.7              # Semi-major axis [km] (~20,200 km altitude)
eT = 0.001                # Near-circular
iT = deg2rad(55.0)        # GPS inclination
ΩT = deg2rad(30.0)
ωT = 0.0
νT = 0.0

kep0 = Keplerian(a0, e0, i0, Ω0, ω0, ν0)
kepT = Keplerian(aT, eT, iT, ΩT, ωT, νT)

oe0 = ModEq(kep0, μ)
oeT = ModEq(kepT, μ)

# Spacecraft: electric propulsion
Tmax = 0.5     # Maximum thrust [N] (lower thrust → longer transfer)
Isp = 3000.0  # Specific impulse [s] (Hall thruster class)
m0 = 500.0   # Initial mass [kg]
m_dry = 100.0   # Dry mass [kg]
spacecraft = QLawSpacecraft(m_dry, m0, Tmax, Isp)

# Time span (allow up to 200 days for this demanding transfer)
tspan = (0.0, 200.0 * 86400.0)

# Epoch
JD0 = date_to_jd(2025, 3, 21, 0, 0, 0)  # Spring equinox 2025

println("=" ^ 70)
println("HIGH-FIDELITY LEO → INCLINED MEO TRANSFER")
println("=" ^ 70)
println()
println("Spacecraft:")
println("  Dry mass:   $(m_dry) kg")
println("  Wet mass:   $(m0) kg")
println("  Thrust:     $(Tmax*1000) mN")
println("  Isp:        $(Isp) s")
println("  Vex:        $(round(exhaust_velocity(spacecraft), digits=3)) km/s")
println()
println("Initial orbit (equatorial LEO):")
println("  a = $(round(a0, digits=1)) km  (alt = $(round(a0 - 6378.137, digits=1)) km)")
println("  e = $(e0)")
println("  i = $(rad2deg(i0))°")
println()
println("Target orbit (inclined MEO / GPS-like):")
println("  a = $(round(aT, digits=1)) km  (alt = $(round(aT - 6378.137, digits=1)) km)")
println("  e = $(eT)")
println("  i = $(rad2deg(iT))°")
println()

# =============================================================================
# High-Fidelity Dynamics Model
# =============================================================================

println("Setting up high-fidelity dynamics model...")

# --- Earth Orientation Parameters ---
println("  Fetching IERS EOP data...")
eop_data = fetch_iers_eop()

# --- Gravity: 36×36 EGM96 ---
println("  Fetching EGM96 gravity model (36×36)...")
egm96_file = fetch_icgem_file(:EGM96)
gravity_coeffs = GravityModels.load(IcgemFile, egm96_file)

gravity_model = GravityHarmonicsAstroModel(;
    gravity_model = gravity_coeffs,
    eop_data = eop_data,
    degree = 36,
    order = 36,
)

# --- Third-body: Moon and Sun ---
moon_model = ThirdBodyModel(; body = MoonBody(), eop_data = eop_data)
sun_model = ThirdBodyModel(; body = SunBody(), eop_data = eop_data)

# --- Solar Radiation Pressure ---
# Cannonball model: CR = 1.3 (typical), effective area from equivalent sphere
# For a 2000 kg spacecraft with ~10 m² cross-section:
#   radius ≈ sqrt(A/π) ≈ 1.784 m
sat_srp = CannonballFixedSRP(1.784, m0, 1.3)   # radius [m], mass [kg], CR

srp_model = SRPAstroModel(;
    satellite_srp_model = sat_srp,
    sun_data = sun_model,
    eop_data = eop_data,
    shadow_model = Conical(),
)

# --- Atmospheric Drag ---
# Cannonball model: CD = 2.2 (typical), same cross-section
# Drag is significant in LEO but negligible once altitude rises above ~800 km.
sat_drag = CannonballFixedDrag(1.784, m0, 2.2)  # radius [m], mass [kg], CD

drag_model = DragAstroModel(;
    satellite_drag_model = sat_drag,
    atmosphere_model = ExpAtmo(),   # Exponential atmosphere (fast, sufficient for Q-Law)
    eop_data = eop_data,
)

# --- Combined dynamics model ---
dynamics_model =
    CentralBodyDynamicsModel(gravity_model, (moon_model, sun_model, srp_model, drag_model))

println()
println("Perturbation environment:")
println("  Gravity:     EGM96 36×36 harmonics")
println("  Third-body:  Moon + Sun")
println("  SRP:         Cannonball (CR=1.3, A=10 m²), conical shadow")
println("  Drag:        Cannonball (CD=2.2, A=10 m²), exponential atmosphere")
println()

# =============================================================================
# Q-Law Parameters
# =============================================================================

# Weights: emphasize inclination change (Wh, Wk) since the 55° plane change
# is the most expensive part of this transfer.
weights = QLawWeights(0.0902, 0.8902, 0.9137, 0.9686, 0.9882)
ηth = -0.01

params = QLawParameters(;
    Wp = 1.0,
    rp_min = 6378.137 + 200.0,     # Min periapsis: 200 km altitude
    η_threshold = ηth,             # Moderate coasting threshold
    η_smoothness = 1e-4,
    effectivity_type = AbsoluteEffectivity(),
    n_search_points = 50,
    convergence_criterion = SummedErrorConvergence(0.05),
)

# =============================================================================
# Create and Solve
# =============================================================================

println("Creating Q-Law problem...")
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

println("Solving (this may take several minutes)...")
println()

@time sol = solve(prob)

# =============================================================================
# Results
# =============================================================================

println()
println("=" ^ 70)
println("RESULTS")
println("=" ^ 70)

# Extract trajectory data
t_days = sol.trajectory.t ./ 86400.0
states = sol.trajectory.u

a_hist = [QLawController.get_sma(ModEq(u[1], u[2], u[3], u[4], u[5], u[6])) for u in states]
e_hist = [sqrt(u[2]^2 + u[3]^2) for u in states]
i_hist = [rad2deg(2 * atan(sqrt(u[4]^2 + u[5]^2))) for u in states]
m_hist = [u[7] for u in states]

# Final orbital elements
final_a = QLawController.get_sma(sol.final_oe)
final_e = sqrt(sol.final_oe.f^2 + sol.final_oe.g^2)
final_i = rad2deg(2 * atan(sqrt(sol.final_oe.h^2 + sol.final_oe.k^2)))

println()
println("  Converged:       $(sol.converged)")
println("  Transfer time:   $(round(sol.elapsed_time / 86400.0, digits=2)) days")
println("  Propellant used: $(round(m0 - sol.final_mass, digits=2)) kg")
println("  Final mass:      $(round(sol.final_mass, digits=2)) kg")
println()
println("  Final orbital elements:")
println(
    "    a = $(round(final_a, digits=2)) km   (target: $(round(aT, digits=2)) km,  Δ = $(round(final_a - aT, digits=2)) km)",
)
println(
    "    e = $(round(final_e, digits=6))        (target: $(round(eT, digits=6)),  Δ = $(round(final_e - eT, sigdigits=3)))",
)
println(
    "    i = $(round(final_i, digits=4))°      (target: $(round(rad2deg(iT), digits=4))°,  Δ = $(round(final_i - rad2deg(iT), digits=4))°)",
)
println()

# =============================================================================
# Compute Control History
# =============================================================================

println("Computing control history for plots...")

shadow = Conical()
α_hist = Float64[]
β_hist = Float64[]
thrust_hist = Float64[]
η_hist = Float64[]
γ_hist = Float64[]

for (idx, u) in enumerate(states)
    oe_i = ModEq(u[1], u[2], u[3], u[4], u[5], u[6])
    m_i = u[7]
    r_i = QLawController.compute_radius(oe_i)
    t_i = sol.trajectory.t[idx]

    F_max = max_thrust_acceleration(spacecraft, m_i, r_i)

    α_opt, β_opt, _ = QLawController.compute_thrust_direction(oe_i, oeT, weights, μ, F_max, params)
    η, _, _, _ = QLawController.compute_effectivity(oe_i, oeT, weights, μ, F_max, params)
    activation = QLawController.effectivity_activation(η, params.η_threshold, params.η_smoothness)

    JD_i = JD0 + t_i / 86400.0
    sun_pos = sun_model(JD_i, Position())
    γ = QLawController.compute_sunlight_fraction(oe_i, μ, sun_pos, shadow)

    throttle = activation * γ

    push!(α_hist, α_opt)
    push!(β_hist, β_opt)
    push!(thrust_hist, throttle * F_max * 1e6)  # mm/s²
    push!(η_hist, η)
    push!(γ_hist, γ)
end

# Cartesian positions for orbit plot
x_hist = Float64[]
y_hist = Float64[]
z_hist = Float64[]
for u in states
    cart = Cartesian(ModEq(u[1], u[2], u[3], u[4], u[5], u[6]), μ)
    push!(x_hist, cart.x)
    push!(y_hist, cart.y)
    push!(z_hist, cart.z)
end

# Altitude history (for drag context)
alt_hist = [
    QLawController.compute_radius(ModEq(u[1], u[2], u[3], u[4], u[5], u[6])) - 6378.137 for
    u in states
]

# =============================================================================
# Plot 1: Orbital Elements vs Time
# =============================================================================
println("Generating plots...")

p_a = plot(
    t_days,
    a_hist ./ 1e3,
    ylabel = "a [×10³ km]",
    label = false,
    linewidth = 0.5,
    color = :blue,
)
hline!([aT / 1e3], linestyle = :dash, color = :red, linewidth = 1.5, label = "Target")

p_e = plot(t_days, e_hist, ylabel = "e [-]", label = false, linewidth = 0.5, color = :blue)
hline!([eT], linestyle = :dash, color = :red, linewidth = 1.5, label = false)

p_i =
    plot(t_days, i_hist, ylabel = "i [deg]", label = false, linewidth = 0.5, color = :blue)
hline!([rad2deg(iT)], linestyle = :dash, color = :red, linewidth = 1.5, label = false)

p_m = plot(
    t_days,
    m_hist,
    xlabel = "Time [days]",
    ylabel = "Mass [kg]",
    label = false,
    linewidth = 1,
    color = :blue,
)
hline!([m_dry], linestyle = :dash, color = :red, linewidth = 1.5, label = "Dry mass")

fig_oe = plot(
    p_a,
    p_e,
    p_i,
    p_m,
    layout = (2, 2),
    size = (1100, 750),
    plot_title = "Orbital Elements — LEO → MEO (Hi-Fi)",
)
savefig(fig_oe, "meo_orbital_elements.png")
println("  Saved: meo_orbital_elements.png")

# =============================================================================
# Plot 2: Controls
# =============================================================================

p_thr = plot(
    t_days,
    thrust_hist,
    ylabel = "Thrust Accel\n[mm/s²]",
    label = false,
    linewidth = 0.3,
    color = :blue,
)

p_alp = plot(
    t_days,
    rad2deg.(α_hist),
    ylabel = "α* [deg]",
    label = false,
    linewidth = 0.3,
    color = :blue,
)

p_bet = plot(
    t_days,
    rad2deg.(β_hist),
    ylabel = "β* [deg]",
    label = false,
    linewidth = 0.3,
    color = :blue,
)

p_eta = plot(
    t_days,
    η_hist,
    ylabel = "η [-]",
    xlabel = "Time [days]",
    label = false,
    linewidth = 0.3,
    color = :blue,
)
hline!([params.η_threshold], linestyle = :dash, color = :red, linewidth = 1, label = "ηₜₕ")

fig_ctrl = plot(
    p_thr,
    p_alp,
    p_bet,
    p_eta,
    layout = (4, 1),
    size = (1100, 900),
    plot_title = "Controls — LEO → MEO (Hi-Fi)",
)
savefig(fig_ctrl, "meo_controls.png")
println("  Saved: meo_controls.png")

# =============================================================================
# Plot 3: 3D Orbit Shape (three 2D projections)
# =============================================================================

p_xy = plot(
    x_hist ./ 1e3,
    y_hist ./ 1e3,
    xlabel = "X [×10³ km]",
    ylabel = "Y [×10³ km]",
    linewidth = 0.2,
    color = :blue,
    legend = false,
    title = "X-Y (Equatorial)",
    aspect_ratio = :equal,
)
scatter!(
    [x_hist[1] / 1e3],
    [y_hist[1] / 1e3],
    markersize = 4,
    color = :green,
    label = "Start",
)
scatter!(
    [x_hist[end] / 1e3],
    [y_hist[end] / 1e3],
    markersize = 4,
    color = :red,
    label = "End",
)

p_yz = plot(
    y_hist ./ 1e3,
    z_hist ./ 1e3,
    xlabel = "Y [×10³ km]",
    ylabel = "Z [×10³ km]",
    linewidth = 0.2,
    color = :blue,
    legend = false,
    title = "Y-Z",
    aspect_ratio = :equal,
)
scatter!([y_hist[1] / 1e3], [z_hist[1] / 1e3], markersize = 4, color = :green)
scatter!([y_hist[end] / 1e3], [z_hist[end] / 1e3], markersize = 4, color = :red)

p_xz = plot(
    x_hist ./ 1e3,
    z_hist ./ 1e3,
    xlabel = "X [×10³ km]",
    ylabel = "Z [×10³ km]",
    linewidth = 0.2,
    color = :blue,
    legend = false,
    title = "X-Z",
    aspect_ratio = :equal,
)
scatter!([x_hist[1] / 1e3], [z_hist[1] / 1e3], markersize = 4, color = :green)
scatter!([x_hist[end] / 1e3], [z_hist[end] / 1e3], markersize = 4, color = :red)

fig_orbit = plot(
    p_xy,
    p_yz,
    p_xz,
    layout = (1, 3),
    size = (1500, 500),
    plot_title = "Orbit Shape — LEO → MEO (Hi-Fi)",
)
savefig(fig_orbit, "meo_orbit_shape.png")
println("  Saved: meo_orbit_shape.png")

# =============================================================================
# Plot 4: Altitude and Eclipse History (perturbation context)
# =============================================================================

p_alt = plot(
    t_days,
    alt_hist,
    ylabel = "Altitude [km]",
    label = false,
    linewidth = 0.5,
    color = :blue,
    title = "Altitude History",
)
hline!([400.0], linestyle = :dot, color = :gray, linewidth = 1, label = "Initial alt")
hline!(
    [aT - 6378.137],
    linestyle = :dash,
    color = :red,
    linewidth = 1,
    label = "Target alt",
)

p_sun = plot(
    t_days,
    γ_hist,
    ylabel = "Sunlight γ [-]",
    xlabel = "Time [days]",
    label = false,
    linewidth = 0.3,
    color = :orange,
    title = "Eclipse History (γ=0 → shadow)",
)
ylims!((-0.05, 1.1))

fig_env = plot(
    p_alt,
    p_sun,
    layout = (2, 1),
    size = (1100, 600),
    plot_title = "Environment — LEO → MEO (Hi-Fi)",
)
savefig(fig_env, "meo_environment.png")
println("  Saved: meo_environment.png")

# =============================================================================
# Display
# =============================================================================

display(fig_oe)
display(fig_ctrl)
display(fig_orbit)
display(fig_env)

println()
println("All plots saved to current directory.")
println("=" ^ 70)

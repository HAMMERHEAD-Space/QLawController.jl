# =============================================================================
# Earth-to-Mars Heliocentric SEP Transfer
#
# Demonstrates Q-Law for an interplanetary low-thrust transfer using Solar
# Electric Propulsion (SEP). Thrust scales as (1 AU / r)² with distance
# from the Sun.
#
# Central body:  Sun
# Initial orbit: Earth-like heliocentric orbit (1 AU, e≈0.017, i≈0°)
# Target orbit:  Mars-like heliocentric orbit (1.524 AU, e≈0.093, i≈1.85°)
#
# No perturbations are applied (two-body Sun + Q-Law thrust only).
# Eclipse effects are disabled since the spacecraft is in heliocentric space.
#
# To run:
#   using Pkg
#   Pkg.activate("examples")
#   Pkg.instantiate()
#   include("examples/earth_to_mars_sep.jl")
# =============================================================================

using QLawController

using AstroCoords
using AstroForceModels
using Plots

# =============================================================================
# Constants
# =============================================================================

AU = 1.495978707e8           # Astronomical unit [km]
μ_sun = 1.32712440018e11     # Sun gravitational parameter [km³/s²]

# =============================================================================
# Heliocentric Orbits (ecliptic J2000)
# =============================================================================

# Earth orbit (approximate mean elements)
a_earth = 1.000 * AU        # Semi-major axis [km]
e_earth = 0.0167            # Eccentricity
i_earth = deg2rad(0.0)      # Inclination to ecliptic [rad]

# Mars orbit (approximate mean elements)
a_mars = 1.524 * AU        # Semi-major axis [km]
e_mars = 0.0934            # Eccentricity
i_mars = deg2rad(1.85)     # Inclination to ecliptic [rad]

kep0 = Keplerian(a_earth, e_earth, i_earth, 0.0, 0.0, 0.0)
kepT = Keplerian(a_mars, e_mars, i_mars, 0.0, 0.0, 0.0)

oe0 = ModEq(kep0, μ_sun)
oeT = ModEq(kepT, μ_sun)

# =============================================================================
# SEP Spacecraft
#
# Representative of a science/cargo mission with a Hall-effect thruster:
#   - 500 mN thrust at 1 AU (scales as 1/r²)
#   - Isp = 2500 s
#   - 2000 kg wet mass, 1200 kg dry mass
# =============================================================================

Tmax_1AU = 0.5               # Thrust at 1 AU [N]
Isp = 2500.0            # Specific impulse [s]
m0 = 1200.0            # Initial mass [kg]
m_dry = 400.0             # Dry mass [kg]

spacecraft = SEPQLawSpacecraft(m_dry, m0, Tmax_1AU, Isp, AU)

println("=" ^ 70)
println("EARTH → MARS HELIOCENTRIC SEP TRANSFER")
println("=" ^ 70)
println()
println("Spacecraft (SEP):")
println("  Dry mass:       $(m_dry) kg")
println("  Wet mass:       $(m0) kg")
println("  Thrust @ 1 AU:  $(Tmax_1AU * 1000) mN")
println("  Isp:            $(Isp) s")
println("  Vex:            $(round(exhaust_velocity(spacecraft), digits=3)) km/s")
println("  Thrust @ Mars:  $(round(Tmax_1AU * (1.0/1.524)^2 * 1000, digits=1)) mN")
println()
println("Initial orbit (Earth-like):")
println("  a = $(round(a_earth / AU, digits=4)) AU  ($(round(a_earth, digits=0)) km)")
println("  e = $(e_earth)")
println("  i = $(round(rad2deg(i_earth), digits=2))°")
println()
println("Target orbit (Mars-like):")
println("  a = $(round(a_mars / AU, digits=4)) AU  ($(round(a_mars, digits=0)) km)")
println("  e = $(e_mars)")
println("  i = $(round(rad2deg(i_mars), digits=2))°")
println()

# =============================================================================
# Q-Law Parameters
#
# For a heliocentric transfer the semi-major axis change dominates.
# The inclination change (1.85°) is small, so we weight a heavily.
# The periapsis penalty is set far below Earth orbit so it never activates.
# =============================================================================

weights = QLawWeights(1.0, 0.5, 0.5, 0.3, 0.3)

# rp_min well below Earth perihelion so the penalty never fires
rp_min_helio = 0.5 * AU

params = QLawParameters(;
    Wp = 0.0,                               # No periapsis penalty
    rp_min = rp_min_helio,
    η_threshold = 0.1,                       # Moderate coasting
    η_smoothness = 1e-4,
    effectivity_type = AbsoluteEffectivity(),
    n_search_points = 50,
    convergence_criterion = SummedErrorConvergence(0.05),
)

# =============================================================================
# Time span
#
# Hohmann transfer time ≈ 259 days. Low-thrust takes longer.
# Allow up to 800 days.
# =============================================================================

tspan = (0.0, 1200.0 * 86400.0)

# =============================================================================
# Create and Solve
#
# No dynamics_model needed — the default Keplerian gravity with μ_sun is
# sufficient for a two-body heliocentric transfer.
# No sun_model or eclipse (spacecraft is always in sunlight).
# =============================================================================

println("Creating Q-Law problem (heliocentric, two-body + SEP thrust)...")

prob = qlaw_problem(
    oe0,
    oeT,
    tspan,
    μ_sun,
    spacecraft;
    weights = weights,
    qlaw_params = params,
    shadow_model_type = No_Shadow(),    # Always sunlit in heliocentric space
)

println("Solving (this may take a few minutes)...")
println()

@time sol = solve(prob)

# =============================================================================
# Results
# =============================================================================

println()
println("=" ^ 70)
println("RESULTS")
println("=" ^ 70)

t_days = sol.trajectory.t ./ 86400.0
states = sol.trajectory.u

a_hist = [QLawController.get_sma(ModEq(u[1], u[2], u[3], u[4], u[5], u[6])) for u in states]
e_hist = [sqrt(u[2]^2 + u[3]^2) for u in states]
i_hist = [rad2deg(2 * atan(sqrt(u[4]^2 + u[5]^2))) for u in states]
m_hist = [u[7] for u in states]

final_a = QLawController.get_sma(sol.final_oe)
final_e = sqrt(sol.final_oe.f^2 + sol.final_oe.g^2)
final_i = rad2deg(2 * atan(sqrt(sol.final_oe.h^2 + sol.final_oe.k^2)))

println()
println("  Converged:       $(sol.converged)")
println("  Transfer time:   $(round(sol.elapsed_time / 86400.0, digits=2)) days")
println("  Propellant used: $(round(m0 - sol.final_mass, digits=2)) kg")
println("  Final mass:      $(round(sol.final_mass, digits=2)) kg")
println("  ΔV (Tsiolkovsky):$(round(sol.Δv_total, digits=3)) km/s")
println()
println("  Final orbital elements:")
println(
    "    a = $(round(final_a / AU, digits=4)) AU   (target: $(round(a_mars / AU, digits=4)) AU,  Δ = $(round((final_a - a_mars) / AU, digits=4)) AU)",
)
println(
    "    e = $(round(final_e, digits=6))        (target: $(round(e_mars, digits=6)),  Δ = $(round(final_e - e_mars, sigdigits=3)))",
)
println(
    "    i = $(round(final_i, digits=4))°      (target: $(round(rad2deg(i_mars), digits=4))°,  Δ = $(round(final_i - rad2deg(i_mars), digits=4))°)",
)
println()

# =============================================================================
# Compute Control History
# =============================================================================

println("Computing control history for plots...")

α_hist = Float64[]
β_hist = Float64[]
thrust_hist = Float64[]  # actual thrust acceleration [mm/s²]
η_hist = Float64[]

for (idx, u) in enumerate(states)
    oe_i = ModEq(u[1], u[2], u[3], u[4], u[5], u[6])
    m_i = u[7]
    r_i = QLawController.compute_radius(oe_i)

    F_max = max_thrust_acceleration(spacecraft, m_i, r_i)

    α_opt, β_opt, _ =
        QLawController.compute_thrust_direction(oe_i, oeT, weights, μ_sun, F_max, params)
    η, _, _, _ = QLawController.compute_effectivity(oe_i, oeT, weights, μ_sun, F_max, params)
    activation = QLawController.effectivity_activation(η, params.η_threshold, params.η_smoothness)

    push!(α_hist, α_opt)
    push!(β_hist, β_opt)
    push!(thrust_hist, activation * F_max * 1e6)  # mm/s²
    push!(η_hist, η)
end

# Cartesian positions for orbit plot (heliocentric)
x_hist = Float64[]
y_hist = Float64[]
z_hist = Float64[]
for u in states
    cart = Cartesian(ModEq(u[1], u[2], u[3], u[4], u[5], u[6]), μ_sun)
    push!(x_hist, cart.x)
    push!(y_hist, cart.y)
    push!(z_hist, cart.z)
end

# Distance from Sun
r_hist = [QLawController.compute_radius(ModEq(u[1], u[2], u[3], u[4], u[5], u[6])) for u in states]

# =============================================================================
# Plot 1: Orbital Elements vs Time
# =============================================================================
println("Generating plots...")

p_a = plot(
    t_days,
    a_hist ./ AU,
    ylabel = "a [AU]",
    label = false,
    linewidth = 1,
    color = :blue,
)
hline!([a_mars / AU], linestyle = :dash, color = :red, linewidth = 1.5, label = "Mars")
hline!([a_earth / AU], linestyle = :dot, color = :green, linewidth = 1, label = "Earth")

p_e = plot(t_days, e_hist, ylabel = "e [-]", label = false, linewidth = 1, color = :blue)
hline!([e_mars], linestyle = :dash, color = :red, linewidth = 1.5, label = false)

p_i = plot(t_days, i_hist, ylabel = "i [deg]", label = false, linewidth = 1, color = :blue)
hline!([rad2deg(i_mars)], linestyle = :dash, color = :red, linewidth = 1.5, label = false)

p_m = plot(
    t_days,
    m_hist,
    xlabel = "Time [days]",
    ylabel = "Mass [kg]",
    label = false,
    linewidth = 1.5,
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
    plot_title = "Orbital Elements — Earth → Mars (SEP)",
)
savefig(fig_oe, "mars_orbital_elements.png")
println("  Saved: mars_orbital_elements.png")

# =============================================================================
# Plot 2: Controls
# =============================================================================

p_thr = plot(
    t_days,
    thrust_hist,
    ylabel = "Thrust Accel\n[mm/s²]",
    label = false,
    linewidth = 0.5,
    color = :blue,
)

p_alp = plot(
    t_days,
    rad2deg.(α_hist),
    ylabel = "α* [deg]",
    label = false,
    linewidth = 0.5,
    color = :blue,
)

p_bet = plot(
    t_days,
    rad2deg.(β_hist),
    ylabel = "β* [deg]",
    label = false,
    linewidth = 0.5,
    color = :blue,
)

p_eta = plot(
    t_days,
    η_hist,
    ylabel = "η [-]",
    xlabel = "Time [days]",
    label = false,
    linewidth = 0.5,
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
    plot_title = "Controls — Earth → Mars (SEP)",
)
savefig(fig_ctrl, "mars_controls.png")
println("  Saved: mars_controls.png")

# =============================================================================
# Plot 3: Heliocentric Orbit (top-down ecliptic view + side view)
# =============================================================================

# Reference circles for Earth and Mars orbits
θ_ref = range(0, 2π, length = 361)
earth_x = a_earth .* cos.(θ_ref) ./ AU
earth_y = a_earth .* sin.(θ_ref) ./ AU
mars_x = a_mars .* cos.(θ_ref) ./ AU
mars_y = a_mars .* sin.(θ_ref) ./ AU

p_top = plot(
    x_hist ./ AU,
    y_hist ./ AU,
    xlabel = "X [AU]",
    ylabel = "Y [AU]",
    linewidth = 1,
    color = :blue,
    label = "Trajectory",
    title = "Ecliptic Plane (top-down)",
    aspect_ratio = :equal,
    legend = :topright,
)
plot!(
    earth_x,
    earth_y,
    linestyle = :dot,
    color = :green,
    linewidth = 0.8,
    label = "Earth orbit",
)
plot!(mars_x, mars_y, linestyle = :dot, color = :red, linewidth = 0.8, label = "Mars orbit")
scatter!([0], [0], markersize = 6, color = :orange, markershape = :star5, label = "Sun")
scatter!(
    [x_hist[1] / AU],
    [y_hist[1] / AU],
    markersize = 5,
    color = :green,
    label = "Departure",
)
scatter!(
    [x_hist[end] / AU],
    [y_hist[end] / AU],
    markersize = 5,
    color = :red,
    label = "Arrival",
)

p_side = plot(
    x_hist ./ AU,
    z_hist ./ AU,
    xlabel = "X [AU]",
    ylabel = "Z [AU]",
    linewidth = 1,
    color = :blue,
    label = false,
    title = "Side View (ecliptic edge-on)",
    aspect_ratio = :equal,
)
scatter!([0], [0], markersize = 6, color = :orange, markershape = :star5, label = "Sun")
scatter!([x_hist[1] / AU], [z_hist[1] / AU], markersize = 5, color = :green, label = "Dep")
scatter!(
    [x_hist[end] / AU],
    [z_hist[end] / AU],
    markersize = 5,
    color = :red,
    label = "Arr",
)

fig_orbit = plot(
    p_top,
    p_side,
    layout = (1, 2),
    size = (1400, 600),
    plot_title = "Heliocentric Trajectory — Earth → Mars (SEP)",
)
savefig(fig_orbit, "mars_orbit.png")
println("  Saved: mars_orbit.png")

# =============================================================================
# Plot 4: Solar Distance and Thrust Available
# =============================================================================

thrust_avail = [Tmax_1AU * (AU / r)^2 * 1000 for r in r_hist]  # mN

p_r = plot(
    t_days,
    r_hist ./ AU,
    ylabel = "r [AU]",
    label = false,
    linewidth = 1.5,
    color = :blue,
    title = "Heliocentric Distance",
)
hline!([1.0], linestyle = :dot, color = :green, linewidth = 1, label = "1 AU")
hline!([1.524], linestyle = :dash, color = :red, linewidth = 1, label = "Mars")

p_T = plot(
    t_days,
    thrust_avail,
    ylabel = "Available Thrust [mN]",
    xlabel = "Time [days]",
    label = false,
    linewidth = 1.5,
    color = :purple,
    title = "SEP Thrust vs Distance",
)

fig_sep = plot(
    p_r,
    p_T,
    layout = (2, 1),
    size = (1100, 600),
    plot_title = "SEP Environment — Earth → Mars",
)
savefig(fig_sep, "mars_sep_environment.png")
println("  Saved: mars_sep_environment.png")

# =============================================================================
# Display
# =============================================================================

display(fig_oe)
display(fig_ctrl)
display(fig_orbit)
display(fig_sep)

println()
println("All plots saved to current directory.")
println("=" ^ 70)

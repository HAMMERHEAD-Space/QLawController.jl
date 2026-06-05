@testset "Aqua.jl" begin
    Aqua.test_all(
        QLawController; ambiguities=(recursive = false), deps_compat=(check_extras = false)
    )
end

# On Julia 1.12+, JET reports spurious errors originating in SciMLBase internals
# (e.g. `has_initialization_data`/`numargs`) that are reached when the out-of-place
# `eom` closure is wrapped into an `ODEProblem`. These accesses are provably
# unreachable at runtime (guarded by `hasfield(...)` short-circuits), but JET on
# 1.12 fails to constant-fold the guard and flags the dead `getproperty`. The
# package's own code analyzes cleanly, so JET is run on 1.10/1.11 and skipped on
# 1.12+ until the upstream JET/SciMLBase interaction is resolved.
const _SKIP_JET = VERSION >= v"1.12"

if _SKIP_JET
    @info "Skipping JET.test_package on Julia $(VERSION): spurious SciMLBase-internal reports under JET on 1.12+."
else
    @testset "JET Testing" begin
        rep = JET.test_package(
            QLawController; toplevel_logger=nothing, target_modules=(@__MODULE__,)
        )
    end
end

# AllocCheck reports spurious `jl_get_pgcstack_static` "allocating runtime
# call"s on macOS aarch64 with Julia 1.12+. These are not real heap allocations:
# the analyzed code is allocation-free on every other platform/version (Linux,
# Windows, and macOS on Julia 1.10/1.11). This is a known AllocCheck/Julia
# limitation, so the checks are skipped on the affected platform.
# Ref: https://github.com/SciML/SciMLStructures.jl/issues/59
const _SKIP_ALLOCCHECK = Sys.isapple() && Sys.ARCH === :aarch64 && VERSION >= v"1.12"

if _SKIP_ALLOCCHECK
    @info "Skipping AllocCheck allocation tests (spurious jl_get_pgcstack_static reports on macOS aarch64 + Julia 1.12+; see SciML/SciMLStructures.jl#59)."
end

# Wrapper around `check_allocs` that honors the platform skip and, when real
# allocations are detected, dumps the full vector (with backtraces) to stdout so
# CI logs reveal exactly what is allocating.
function checked_allocs(f, types)
    _SKIP_ALLOCCHECK && return ()
    allocs = check_allocs(f, types)
    if !isempty(allocs)
        printstyled(stdout, "\n[ALLOC] "; color=:red, bold=true)
        println(stdout, f, " with ", types, " => ", length(allocs), " allocation(s)")
        for (i, a) in enumerate(allocs)
            println(stdout, "  ──────── allocation ", i, " ────────")
            show(stdout, MIME"text/plain"(), a)
            println(stdout)
        end
        flush(stdout)
    end
    return allocs
end

@testset "Core Function Allocations" begin
    μ = 398600.4418
    oe0 = ModEq(Keplerian(6778.0, 0.001, deg2rad(28.5), 0.0, 0.0, 0.0), μ)
    oeT = ModEq(Keplerian(42164.0, 0.001, 0.0, 0.0, 0.0, 0.0), μ)

    spacecraft = QLawSpacecraft(500.0, 1000.0, 1.0, 3000.0)
    weights = QLawWeights(1.0)
    params = QLawParameters(; effectivity_search=GridSearch(), n_search_points=20)

    F_max = max_thrust_acceleration(spacecraft, 1000.0, 6778.0)
    a0 = get_sma(oe0)
    aT = get_sma(oeT)

    oe_vec = SVector{5,Float64}(a0, oe0.f, oe0.g, oe0.h, oe0.k)
    oeT_vec = SVector{5,Float64}(aT, oeT.f, oeT.g, oeT.h, oeT.k)
    W_vec = SVector{5,Float64}(1.0, 1.0, 1.0, 1.0, 1.0)
    max_rates = compute_max_rates_analytical(a0, oe0.f, oe0.g, oe0.h, oe0.k, μ, F_max)

    sun_pos = SVector{3,Float64}(1.495978707e8, 0.0, 0.0)

    @testset "compute_max_rates_analytical" begin
        @test length(
            checked_allocs(
                compute_max_rates_analytical,
                (Float64, Float64, Float64, Float64, Float64, Float64, Float64),
            ),
        ) == 0
    end

    @testset "compute_scaling" begin
        @test length(checked_allocs(compute_scaling, (Float64, Float64))) == 0
    end

    @testset "compute_penalty" begin
        @test length(
            checked_allocs(compute_penalty, (Float64, Float64, Float64, Float64))
        ) == 0
    end

    @testset "compute_Q_from_vec_with_rates" begin
        @test length(
            checked_allocs(
                compute_Q_from_vec_with_rates,
                (
                    typeof(oe_vec),
                    typeof(oeT_vec),
                    typeof(W_vec),
                    typeof(max_rates),
                    Float64,
                    Float64,
                ),
            ),
        ) == 0
    end

    @testset "compute_dQ_doe_analytical" begin
        @test length(
            checked_allocs(
                compute_dQ_doe_analytical,
                (
                    typeof(oe_vec),
                    typeof(oeT_vec),
                    typeof(W_vec),
                    typeof(max_rates),
                    Float64,
                    Float64,
                ),
            ),
        ) == 0
    end

    @testset "equinoctial_gve_partials" begin
        @test length(checked_allocs(equinoctial_gve_partials, (typeof(oe0), Float64))) == 0
    end

    @testset "thrust_direction_to_rtn" begin
        @test length(checked_allocs(thrust_direction_to_rtn, (Float64, Float64))) == 0
    end

    @testset "effectivity_activation" begin
        @test length(checked_allocs(effectivity_activation, (Float64, Float64, Float64))) ==
            0
    end

    @testset "apply_frame_rotation" begin
        @test length(checked_allocs(apply_frame_rotation, (SVector{3,Float64}, Float64))) ==
            0
    end

    @testset "compute_sunlight_fraction (ModEq)" begin
        @test length(
            checked_allocs(
                compute_sunlight_fraction,
                (typeof(oe0), Float64, typeof(sun_pos), typeof(Conical())),
            ),
        ) == 0
    end

    @testset "compute_sunlight_fraction (SVector)" begin
        sat_pos_sv = SVector{3,Float64}(6778.0, 0.0, 0.0)
        @test length(
            checked_allocs(
                compute_sunlight_fraction,
                (typeof(sat_pos_sv), typeof(sun_pos), typeof(Conical())),
            ),
        ) == 0
    end
end

@testset "EOM Allocations" begin
    μ = 398600.4418
    JD = 2451545.0

    oe0 = ModEq(Keplerian(6778.0, 0.001, deg2rad(28.5), 0.0, 0.0, 0.0), μ)
    oeT = ModEq(Keplerian(42164.0, 0.001, 0.0, 0.0, 0.0, 0.0), μ)

    spacecraft = QLawSpacecraft(500.0, 1000.0, 1.0, 3000.0)
    weights = QLawWeights(1.0)
    params = QLawParameters(; effectivity_search=GridSearch(), n_search_points=20)

    dynamics_model = CentralBodyDynamicsModel(KeplerianGravityAstroModel(; μ=μ), ())

    prob = qlaw_problem(
        oe0,
        oeT,
        (0.0, 365.25 * 86400.0),
        μ,
        spacecraft;
        weights=weights,
        qlaw_params=params,
        dynamics_model=dynamics_model,
    )

    ps = ComponentVector(; μ=μ, JD=JD)
    u0 = SVector{7,Float64}(oe0.p, oe0.f, oe0.g, oe0.h, oe0.k, oe0.L, 1000.0)

    @testset "qlaw_eom" begin
        @test length(
            checked_allocs(
                (u, p, t) -> qlaw_eom(u, p, t, prob), (typeof(u0), typeof(ps), Float64)
            ),
        ) == 0
    end
end

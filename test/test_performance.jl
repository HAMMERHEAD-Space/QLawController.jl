@testset "Aqua.jl" begin
    Aqua.test_all(
        QLawController; ambiguities=(recursive = false), deps_compat=(check_extras = false)
    )
end

@testset "JET Testing" begin
    rep = JET.test_package(
        QLawController; toplevel_logger=nothing, target_modules=(@__MODULE__,)
    )
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
            check_allocs(
                compute_max_rates_analytical,
                (Float64, Float64, Float64, Float64, Float64, Float64, Float64),
            ),
        ) == 0
    end

    @testset "compute_scaling" begin
        @test length(check_allocs(compute_scaling, (Float64, Float64))) == 0
    end

    @testset "compute_penalty" begin
        @test length(check_allocs(compute_penalty, (Float64, Float64, Float64, Float64))) ==
            0
    end

    @testset "compute_Q_from_vec_with_rates" begin
        @test length(
            check_allocs(
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
            check_allocs(
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
        @test length(check_allocs(equinoctial_gve_partials, (typeof(oe0), Float64))) == 0
    end

    @testset "thrust_direction_to_rtn" begin
        @test length(check_allocs(thrust_direction_to_rtn, (Float64, Float64))) == 0
    end

    @testset "effectivity_activation" begin
        @test length(check_allocs(effectivity_activation, (Float64, Float64, Float64))) == 0
    end

    @testset "apply_frame_rotation" begin
        @test length(check_allocs(apply_frame_rotation, (SVector{3,Float64}, Float64))) == 0
    end

    @testset "compute_sunlight_fraction (ModEq)" begin
        @test length(
            check_allocs(
                compute_sunlight_fraction,
                (typeof(oe0), Float64, typeof(sun_pos), typeof(Conical())),
            ),
        ) == 0
    end

    @testset "compute_sunlight_fraction (SVector)" begin
        sat_pos_sv = SVector{3,Float64}(6778.0, 0.0, 0.0)
        @test length(
            check_allocs(
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
            check_allocs(
                (u, p, t) -> qlaw_eom(u, p, t, prob), (typeof(u0), typeof(ps), Float64)
            ),
        ) == 0
    end
end

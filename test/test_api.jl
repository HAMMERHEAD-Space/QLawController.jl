# ==========================================================================
# Tests for Q-Law API: Problem construction
# ==========================================================================

@testset "Problem Construction" begin
    μ = 398600.4418

    kep0 = Keplerian(7000.0, 0.01, deg2rad(28.5), 0.0, 0.0, 0.0)
    kepT = Keplerian(42164.0, 0.001, deg2rad(0.1), 0.0, 0.0, 0.0)

    oe0 = ModEq(kep0, μ)
    oeT = ModEq(kepT, μ)

    sc = QLawSpacecraft(500.0, 1000.0, 1.0, 3000.0)
    tspan = (0.0, 86400.0 * 180.0)  # 180 days

    @testset "Basic construction" begin
        prob = qlaw_problem(oe0, oeT, tspan, μ, sc)

        @test prob.μ == μ
        @test prob.m0 == 1000.0
        @test prob.tspan == tspan
    end

    @testset "With custom weights" begin
        weights = QLawWeights(0.5, 0.6, 0.7, 0.8, 0.9)
        prob = qlaw_problem(oe0, oeT, tspan, μ, sc; weights = weights)

        @test prob.weights.Wa == 0.5
        @test prob.weights.Wk == 0.9
    end

    @testset "With custom parameters" begin
        params = QLawParameters(; η_threshold = 0.2)
        prob = qlaw_problem(oe0, oeT, tspan, μ, sc; qlaw_params = params)

        @test prob.params.η_threshold == 0.2
    end

    @testset "From Keplerian elements" begin
        prob = qlaw_problem(kep0, kepT, tspan, μ, sc)

        @test prob.μ == μ
        @test QLaw.get_sma(prob.oe0) ≈ 7000.0 rtol=0.01
    end
end

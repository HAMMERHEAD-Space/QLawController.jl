# ==========================================================================
# Tests for QLaw types: Spacecraft, Weights, Parameters
# ==========================================================================

@testset "Spacecraft Types" begin
    @testset "QLawSpacecraft" begin
        sc = QLawSpacecraft(500.0, 1000.0, 0.5, 3000.0)
        @test mass(sc) == 1000.0
        @test sc.dry_mass == 500.0
        @test sc.wet_mass == 1000.0
        @test sc.thrust == 0.5
        @test sc.Isp == 3000.0
        @test exhaust_velocity(sc) ≈ 3000.0 * 9.80665e-3
        @test max_thrust(sc, 7000.0) == 0.5
        @test max_thrust(sc, 42000.0) == 0.5  # Constant thrust

        # Max acceleration
        a_max = max_thrust_acceleration(sc, 1000.0, 7000.0)
        @test a_max ≈ 0.5 / 1000.0 * 1e-3  # km/s²
    end

    @testset "SEPQLawSpacecraft" begin
        AU = 1.495978707e8
        sep = SEPQLawSpacecraft(500.0, 1000.0, 0.5, 3000.0, AU)
        @test mass(sep) == 1000.0
        @test max_thrust(sep, AU) == 0.5
        @test max_thrust(sep, 2*AU) ≈ 0.5 / 4  # Inverse square law
        @test max_thrust(sep, AU/2) ≈ 0.5 * 4  # Closer = more power
    end
end

@testset "Q-Law Weights" begin
    # Default weights
    w = QLawWeights()
    @test w.Wa == 1.0
    @test w.Wf == 1.0
    @test w.Wg == 1.0
    @test w.Wh == 1.0
    @test w.Wk == 1.0

    # Custom weights
    w2 = QLawWeights(0.5, 0.6, 0.7, 0.8, 0.9)
    @test w2.Wa == 0.5
    @test w2.Wf == 0.6
    @test w2.Wg == 0.7
    @test w2.Wh == 0.8
    @test w2.Wk == 0.9

    # Uniform weights
    w3 = QLawWeights(0.5)
    @test w3.Wa == 0.5
    @test w3.Wf == 0.5
    @test w3.Wg == 0.5
    @test w3.Wh == 0.5
    @test w3.Wk == 0.5
end

@testset "Q-Law Parameters" begin
    params = QLawParameters()
    @test params.Wp == 1.0
    @test params.rp_min == 6578.0
    @test params.k_penalty == 100.0
    @test params.η_threshold == -0.01  # Paper: -0.01 for constant thrust
    @test params.η_smoothness == 1e-4
    @test params.Θrot == 0.0
    @test params.effectivity_type isa AbsoluteEffectivity
    @test params.effectivity_search isa RefinedSearch
    @test params.n_search_points == 50
    @test params.m_scaling == 1.0
    @test params.n_scaling == 4.0
    @test params.r_scaling == 2.0
    @test params.convergence_criterion isa SummedErrorConvergence

    params2 = QLawParameters(;
        η_threshold = 0.2,
        effectivity_type = RelativeEffectivity(),
        rp_min = 6500.0,
        Wp = 2.0,
    )
    @test params2.η_threshold == 0.2
    @test params2.effectivity_type isa RelativeEffectivity
    @test params2.rp_min == 6500.0
    @test params2.Wp == 2.0

    # Varga convergence
    params3 = QLawParameters(; convergence_criterion = VargaConvergence(0.01))
    @test params3.convergence_criterion isa VargaConvergence
    @test params3.convergence_criterion.Rc == 0.01

    # Custom scaling
    params4 =
        QLawParameters(; m_scaling = 0.5, n_scaling = 2.0, r_scaling = 1.0, Θrot = 0.1)
    @test params4.m_scaling == 0.5
    @test params4.n_scaling == 2.0
    @test params4.r_scaling == 1.0
    @test params4.Θrot == 0.1

    # Custom penalty steepness and effectivity search
    params5 = QLawParameters(; k_penalty = 50.0, effectivity_search = GridSearch())
    @test params5.k_penalty == 50.0
    @test params5.effectivity_search isa GridSearch
end

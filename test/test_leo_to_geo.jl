# ==========================================================================
# LEO to GEO Transfer (Paper Test Case - Table 1)
# From: "Q-Law for Rapid Assessment of Low Thrust Cislunar Trajectories 
#        via Automatic Differentiation" by Steffen, Falck, and Faller
# ==========================================================================

@testset "LEO to GEO Transfer (Paper Values)" begin
    # From Table 1 in the paper
    μ = 398600.4418  # Earth gravitational parameter [km³/s²]
    
    # Initial orbit (LEO)
    a0 = 6878.0  # km (approximating 500 km altitude)
    e0 = 0.001   # Nearly circular
    i0 = deg2rad(28.5)  # Cape Canaveral latitude
    
    # Target orbit (GEO)
    aT = 42164.0  # km
    eT = 0.001    # Nearly circular
    iT = deg2rad(0.1)  # Nearly equatorial
    
    # Spacecraft parameters (from paper)
    Tmax = 1.0   # N
    Isp = 1500.0 # s
    m0 = 1000.0  # kg
    
    kep0 = Keplerian(a0, e0, i0, 0.0, 0.0, 0.0)
    kepT = Keplerian(aT, eT, iT, 0.0, 0.0, 0.0)
    
    oe0 = ModEq(kep0, μ)
    oeT = ModEq(kepT, μ)
    
    sc = QLawSpacecraft(500.0, m0, Tmax, Isp)
    
    @testset "Spacecraft acceleration calculation" begin
        # F_max = T / m in km/s² = 1 N / 1000 kg * 1e-3 km/m = 1e-6 km/s²
        F_max = max_thrust_acceleration(sc, m0, a0)
        @test F_max ≈ Tmax / m0 * 1e-3 rtol=1e-10
    end
    
    @testset "Q value formula verification" begin
        weights = QLawWeights(0.0785, 0.7926, 0.6876, 0.3862, 0.5)  # DOE from Table 2
        F_max = max_thrust_acceleration(sc, m0, a0)
        
        # Compute Q using implementation
        Q = compute_Q(oe0, oeT, weights, μ, F_max, 0.0, 6378.0)  # Wp=0, no penalty
        
        # Verify by computing Q manually from formula
        # Q = Σ Wᵢ * Sᵢ * ((oeᵢ - oeᵢᵀ) / ȯeᵢ_max)² for max_rates[i] > eps
        a_curr, a_tgt = QLaw.get_sma(oe0), QLaw.get_sma(oeT)
        oe_vec = [a_curr, oe0.f, oe0.g, oe0.h, oe0.k]
        oeT_vec = [a_tgt, oeT.f, oeT.g, oeT.h, oeT.k]
        W_vec = [0.0785, 0.7926, 0.6876, 0.3862, 0.5]
        
        max_rates = QLaw.compute_max_rates(oe0, μ, F_max)
        S = QLaw.compute_scaling(oe0, oeT)
        
        # Match implementation: only include terms where max_rates[i] > eps
        Q_expected = 0.0
        for i in 1:5
            if max_rates[i] > eps(Float64)
                Q_expected += W_vec[i] * S[i] * ((oe_vec[i] - oeT_vec[i]) / max_rates[i])^2
            end
        end
        
        @test Q ≈ Q_expected rtol=1e-10
    end
    
    @testset "Q = 0 at target orbit" begin
        weights = QLawWeights(0.0785, 0.7926, 0.6876, 0.3862, 0.5)
        F_max = max_thrust_acceleration(sc, m0, a0)
        
        Q_at_target = compute_Q(oeT, oeT, weights, μ, F_max, 0.0, 6378.0)
        @test Q_at_target == 0.0
    end
    
    @testset "Thrust direction Qdot formula" begin
        weights = QLawWeights(0.0785, 0.7926, 0.6876, 0.3862, 0.5)
        F_max = max_thrust_acceleration(sc, m0, a0)
        
        α, β, Qdot = compute_thrust_direction(oe0, oeT, weights, μ, F_max, 1.0, 6378.0)
        
        # Verify Qdot = D1*cos(β)*cos(α) + D2*cos(β)*sin(α) + D3*sin(β)
        D1, D2, D3 = QLaw.compute_Qdot_coefficients(oe0, oeT, weights, μ, F_max, 1.0, 6378.0)
        Qdot_expected = D1*cos(β)*cos(α) + D2*cos(β)*sin(α) + D3*sin(β)
        
        @test Qdot ≈ Qdot_expected rtol=1e-10
        @test Qdot < 0  # Q must decrease under optimal thrust
    end
    
    @testset "Thrust direction unit vector formula" begin
        weights = QLawWeights(0.0785, 0.7926, 0.6876, 0.3862, 0.5)
        F_max = max_thrust_acceleration(sc, m0, a0)
        
        α, β, _ = compute_thrust_direction(oe0, oeT, weights, μ, F_max, 1.0, 6378.0)
        u_rtn = QLaw.thrust_direction_to_rtn(α, β)
        
        # Verify unit vector: ||u|| = 1
        @test norm(u_rtn) ≈ 1.0 atol=1e-15
        
        # Verify component formulas
        @test u_rtn[1] ≈ cos(β) * sin(α) atol=1e-15  # Radial
        @test u_rtn[2] ≈ cos(β) * cos(α) atol=1e-15  # Tangential
        @test u_rtn[3] ≈ sin(β) atol=1e-15           # Normal
    end
    
    @testset "Effectivity definition" begin
        weights = QLawWeights(0.0120, 0.5846, 0.1189, 0.9475, 0.3578)  # DOE with effectivity from Table 3
        F_max = max_thrust_acceleration(sc, m0, a0)
        
        η_abs, Qdot_n, Qdot_nn, Qdot_nx = compute_effectivity(
            oe0, oeT, weights, μ, F_max, 1.0, 6378.0, 50, :absolute
        )
        
        # Verify η_absolute = Qdot_n / Qdot_nn
        @test η_abs ≈ Qdot_n / Qdot_nn rtol=1e-10
        
        # Verify ordering by construction: Qdot_nn ≤ Qdot_n ≤ Qdot_nx
        @test Qdot_nn ≤ Qdot_n
        @test Qdot_n ≤ Qdot_nx
        
        # Relative effectivity formula
        η_rel, _, Qdot_nn_r, Qdot_nx_r = compute_effectivity(
            oe0, oeT, weights, μ, F_max, 1.0, 6378.0, 50, :relative
        )
        
        @test η_rel ≈ (Qdot_n - Qdot_nx_r) / (Qdot_nn_r - Qdot_nx_r) rtol=1e-10
    end
    
    @testset "Different paper weights produce different Q" begin
        F_max = max_thrust_acceleration(sc, m0, a0)
        
        # Table 2: Different optimization methods gave different weights
        weights_doe = QLawWeights(0.0785, 0.7926, 0.6876, 0.3862, 0.5)
        weights_ipopt = QLawWeights(0.1167, 0.3801, 0.6256, 0.1981, 0.5)
        weights_ga = QLawWeights(0.1137, 0.8431, 1.0000, 0.4902, 0.5)
        
        Q_doe = compute_Q(oe0, oeT, weights_doe, μ, F_max, 0.0, 6378.0)
        Q_ipopt = compute_Q(oe0, oeT, weights_ipopt, μ, F_max, 0.0, 6378.0)
        Q_ga = compute_Q(oe0, oeT, weights_ga, μ, F_max, 0.0, 6378.0)
        
        @test Q_doe != Q_ipopt
        @test Q_ipopt != Q_ga
        @test Q_doe != Q_ga
    end
    
    @testset "Problem construction stores parameters correctly" begin
        tspan = (0.0, 86400.0 * 60.0)
        weights = QLawWeights(0.0785, 0.7926, 0.6876, 0.3862, 0.5)
        params = QLawParameters(; η_threshold=-0.01, rp_min=6578.0)
        
        prob = qlaw_problem(oe0, oeT, tspan, μ, sc; weights=weights, qlaw_params=params)
        
        @test prob.μ == μ
        @test prob.m0 == m0
        @test prob.tspan == tspan
        @test QLaw.get_sma(prob.oe0) ≈ a0 rtol=1e-10
        @test QLaw.get_sma(prob.oeT) ≈ aT rtol=1e-10
        @test prob.weights.Wa ≈ 0.0785
        @test prob.params.η_threshold ≈ -0.01
        @test prob.params.rp_min ≈ 6578.0
    end
end

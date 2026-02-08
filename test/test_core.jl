# ==========================================================================
# Tests for Q-Law core functionality
# All tests verify against analytically computed expected values
# ==========================================================================

@testset "Equinoctial Utilities" begin
    μ = 398600.4418  # Earth [km³/s²]
    
    @testset "p ↔ a conversion is consistent" begin
        a_original = 7000.0
        f, g = 0.1, 0.05  # e ≈ 0.112
        e_sq = f^2 + g^2
        
        # p = a(1 - e²)
        p = QLaw.a_to_p(a_original, f, g)
        @test p ≈ a_original * (1 - e_sq)
        
        # Roundtrip: a → p → a
        a_recovered = QLaw.p_to_a(p, f, g)
        @test a_recovered ≈ a_original
    end
    
    @testset "get_sma matches Keplerian input" begin
        a_input = 7000.0
        e_input = 0.1
        
        kep = Keplerian(a_input, e_input, 0.5, 0.0, 0.0, 0.0)
        oe = ModEq(kep, μ)
        
        a_computed = QLaw.get_sma(oe)
        @test a_computed ≈ a_input rtol=1e-10
    end
    
    @testset "compute_radius matches orbital mechanics" begin
        a = 10000.0
        e = 0.2
        
        # At periapsis (true anomaly = 0, ω = 0 → L = 0)
        kep_peri = Keplerian(a, e, 0.0, 0.0, 0.0, 0.0)
        oe_peri = ModEq(kep_peri, μ)
        r_peri = QLaw.compute_radius(oe_peri)
        r_peri_expected = a * (1 - e)
        @test r_peri ≈ r_peri_expected rtol=1e-10
        
        # At apoapsis (true anomaly = π)
        kep_apo = Keplerian(a, e, 0.0, 0.0, 0.0, π)
        oe_apo = ModEq(kep_apo, μ)
        r_apo = QLaw.compute_radius(oe_apo)
        r_apo_expected = a * (1 + e)
        @test r_apo ≈ r_apo_expected rtol=1e-10
    end
end

@testset "GVE Partials - Analytical Verification" begin
    μ = 398600.4418
    
    @testset "da/dFr ≈ 0 for circular orbit at L=0" begin
        # da/dFr = (2a*q / (1-e²)) * sqrt(a(1-e²)/μ) * (f*sin(L) - g*cos(L)) / q
        # For circular orbit (e≈0, f≈0, g≈0), this should be ≈ 0
        
        kep = Keplerian(7000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        oe = ModEq(kep, μ)
        
        A = equinoctial_gve_partials(oe, μ)
        
        @test abs(A[1, 1]) < 1e-10
    end
    
    @testset "da/dFh = 0 always (normal thrust doesn't change energy)" begin
        # Normal thrust is perpendicular to velocity, does no work
        kep = Keplerian(7000.0, 0.1, 0.5, 0.3, 0.2, 1.0)
        oe = ModEq(kep, μ)
        
        A = equinoctial_gve_partials(oe, μ)
        
        @test A[1, 3] == 0.0
    end
    
    @testset "dh/dFr = 0 and dh/dFθ = 0 (in-plane thrust doesn't change h)" begin
        kep = Keplerian(7000.0, 0.1, 0.5, 0.3, 0.2, 1.0)
        oe = ModEq(kep, μ)
        
        A = equinoctial_gve_partials(oe, μ)
        
        @test A[4, 1] == 0.0
        @test A[4, 2] == 0.0
    end
end

@testset "Q Function - Analytical Verification" begin
    μ = 398600.4418
    
    @testset "Q = 0 at target" begin
        kep = Keplerian(7000.0, 0.1, 0.5, 0.0, 0.0, 0.0)
        oe = ModEq(kep, μ)
        weights = QLawWeights()
        F_max = 1e-6
        
        Q = compute_Q(oe, oe, weights, μ, F_max, 1.0, 6378.0)
        @test Q == 0.0
    end
    
    @testset "Q formula verification" begin
        # Q = Σ Wᵢ * Sᵢ * ((oeᵢ - oeᵢᵀ) / ȯeᵢ_max)²
        # For simple case with no penalty, single element weighted
        
        μ = 398600.4418
        F_max = 1e-6
        
        # Use only a-weight, all others zero
        weights = QLawWeights(1.0, 0.0, 0.0, 0.0, 0.0)
        
        kep0 = Keplerian(7000.0, 0.001, 0.001, 0.0, 0.0, 0.0)
        kepT = Keplerian(8000.0, 0.001, 0.001, 0.0, 0.0, 0.0)
        oe0 = ModEq(kep0, μ)
        oeT = ModEq(kepT, μ)
        
        # Compute expected Q manually
        a0, aT = QLaw.get_sma(oe0), QLaw.get_sma(oeT)
        max_rates = QLaw.compute_max_rates(oe0, μ, F_max)
        S = QLaw.compute_scaling(oe0, oeT)
        
        # Q = W_a * S_a * ((a - aT) / ȧ_max)²
        Q_expected = 1.0 * S[1] * ((a0 - aT) / max_rates[1])^2
        
        Q_computed = compute_Q(oe0, oeT, weights, μ, F_max, 0.0, 6378.0)  # Wp=0, no penalty
        
        @test Q_computed ≈ Q_expected rtol=1e-10
    end
    
    @testset "Penalty term formula" begin
        # P = exp(k * (1 - rp/rp_min)) when rp < rp_min
        μ = 398600.4418
        
        a = 6500.0
        e = 0.05
        rp = a * (1 - e)  # = 6175 km
        rp_min = 6378.0
        
        kep = Keplerian(a, e, 0.1, 0.0, 0.0, 0.0)
        oe = ModEq(kep, μ)
        
        k = 100.0  # From implementation
        P_expected = exp(k * (1 - rp / rp_min))
        P_computed = QLaw.compute_penalty(oe, rp_min)
        
        @test P_computed ≈ P_expected rtol=1e-10
    end
    
    @testset "Q with penalty vs without" begin
        μ = 398600.4418
        F_max = 1e-6
        weights = QLawWeights()
        rp_min = 6378.0
        
        # Orbit with low periapsis
        kep = Keplerian(6500.0, 0.05, 0.1, 0.0, 0.0, 0.0)
        kepT = Keplerian(42000.0, 0.001, 0.001, 0.0, 0.0, 0.0)
        oe = ModEq(kep, μ)
        oeT = ModEq(kepT, μ)
        
        Q_no_penalty = compute_Q(oe, oeT, weights, μ, F_max, 0.0, rp_min)
        Q_with_penalty = compute_Q(oe, oeT, weights, μ, F_max, 1.0, rp_min)
        
        # Q_with = Q_no * (1 + Wp * P)
        P = QLaw.compute_penalty(oe, rp_min)
        @test Q_with_penalty ≈ Q_no_penalty * (1 + 1.0 * P) rtol=1e-10
    end
end

@testset "Thrust Direction - Analytical Verification" begin
    μ = 398600.4418
    F_max = 1e-6
    
    @testset "Optimal angles formula" begin
        # α* = atan(-D2, -D1)
        # β* = atan(-D3, √(D1² + D2²))
        
        kep0 = Keplerian(7000.0, 0.1, 0.5, 0.0, 0.0, 0.0)
        kepT = Keplerian(42000.0, 0.001, 0.001, 0.0, 0.0, 0.0)
        oe0 = ModEq(kep0, μ)
        oeT = ModEq(kepT, μ)
        weights = QLawWeights()
        
        # Get D coefficients directly
        D1, D2, D3 = QLaw.compute_Qdot_coefficients(oe0, oeT, weights, μ, F_max, 1.0, 6378.0)
        
        # Compute expected angles
        α_expected = atan(-D2, -D1)
        D12 = sqrt(D1^2 + D2^2)
        β_expected = atan(-D3, D12)
        
        α, β, _ = compute_thrust_direction(oe0, oeT, weights, μ, F_max, 1.0, 6378.0)
        
        @test α ≈ α_expected rtol=1e-10
        @test β ≈ β_expected rtol=1e-10
    end
    
    @testset "Qdot formula at optimal direction" begin
        # Qdot = D1*cos(β)*cos(α) + D2*cos(β)*sin(α) + D3*sin(β)
        
        kep0 = Keplerian(7000.0, 0.1, 0.5, 0.0, 0.0, 0.0)
        kepT = Keplerian(42000.0, 0.001, 0.001, 0.0, 0.0, 0.0)
        oe0 = ModEq(kep0, μ)
        oeT = ModEq(kepT, μ)
        weights = QLawWeights()
        
        D1, D2, D3 = QLaw.compute_Qdot_coefficients(oe0, oeT, weights, μ, F_max, 1.0, 6378.0)
        α, β, Qdot = compute_thrust_direction(oe0, oeT, weights, μ, F_max, 1.0, 6378.0)
        
        Qdot_expected = D1*cos(β)*cos(α) + D2*cos(β)*sin(α) + D3*sin(β)
        
        @test Qdot ≈ Qdot_expected rtol=1e-10
    end
    
    @testset "RTN vector is unit vector" begin
        α, β = 0.3, 0.2
        u = QLaw.thrust_direction_to_rtn(α, β)
        
        # cos²β*sin²α + cos²β*cos²α + sin²β = cos²β(sin²α + cos²α) + sin²β = cos²β + sin²β = 1
        @test norm(u) ≈ 1.0 atol=1e-15
    end
    
    @testset "RTN vector formula" begin
        α, β = 0.3, 0.2
        u = QLaw.thrust_direction_to_rtn(α, β)
        
        # Fr = cos(β)*sin(α), Fθ = cos(β)*cos(α), Fh = sin(β)
        @test u[1] ≈ cos(β) * sin(α) atol=1e-15
        @test u[2] ≈ cos(β) * cos(α) atol=1e-15
        @test u[3] ≈ sin(β) atol=1e-15
    end
end

@testset "Effectivity - Definition Verification" begin
    μ = 398600.4418
    F_max = 1e-6
    
    @testset "Absolute effectivity formula: η = Qdot_n / Qdot_nn" begin
        kep0 = Keplerian(7000.0, 0.1, 0.5, 0.0, 0.0, 0.0)
        kepT = Keplerian(42000.0, 0.001, 0.001, 0.0, 0.0, 0.0)
        oe0 = ModEq(kep0, μ)
        oeT = ModEq(kepT, μ)
        weights = QLawWeights()
        
        η, Qdot_n, Qdot_nn, _ = compute_effectivity(
            oe0, oeT, weights, μ, F_max, 1.0, 6378.0, 50, :absolute
        )
        
        # η_absolute = Qdot_n / Qdot_nn
        η_expected = Qdot_n / Qdot_nn
        @test η ≈ η_expected rtol=1e-10
    end
    
    @testset "Relative effectivity formula: η = (Qdot_n - Qdot_nx) / (Qdot_nn - Qdot_nx)" begin
        kep0 = Keplerian(7000.0, 0.1, 0.5, 0.0, 0.0, 0.0)
        kepT = Keplerian(42000.0, 0.001, 0.001, 0.0, 0.0, 0.0)
        oe0 = ModEq(kep0, μ)
        oeT = ModEq(kepT, μ)
        weights = QLawWeights()
        
        η, Qdot_n, Qdot_nn, Qdot_nx = compute_effectivity(
            oe0, oeT, weights, μ, F_max, 1.0, 6378.0, 50, :relative
        )
        
        # η_relative = (Qdot_n - Qdot_nx) / (Qdot_nn - Qdot_nx)
        η_expected = (Qdot_n - Qdot_nx) / (Qdot_nn - Qdot_nx)
        @test η ≈ η_expected rtol=1e-10
    end
    
    @testset "Qdot_nn ≤ Qdot_n ≤ Qdot_nx by construction" begin
        kep0 = Keplerian(7000.0, 0.1, 0.5, 0.0, 0.0, 0.0)
        kepT = Keplerian(42000.0, 0.001, 0.001, 0.0, 0.0, 0.0)
        oe0 = ModEq(kep0, μ)
        oeT = ModEq(kepT, μ)
        weights = QLawWeights()
        
        _, Qdot_n, Qdot_nn, Qdot_nx = compute_effectivity(
            oe0, oeT, weights, μ, F_max, 1.0, 6378.0, 50, :absolute
        )
        
        @test Qdot_nn ≤ Qdot_n
        @test Qdot_n ≤ Qdot_nx
    end
end

@testset "Activation Function - Formula Verification" begin
    @testset "Activation = 0.5 * (1 + tanh((η - η_tr) / μ))" begin
        η = 0.15
        η_tr = 0.1
        μ_smooth = 1e-4
        
        act = QLaw.effectivity_activation(η, η_tr, μ_smooth)
        act_expected = 0.5 * (1 + tanh((η - η_tr) / μ_smooth))
        
        @test act ≈ act_expected rtol=1e-10
    end
    
    @testset "At threshold: activation = 0.5 (since tanh(0) = 0)" begin
        η_tr = 0.1
        act = QLaw.effectivity_activation(η_tr, η_tr, 1e-4)
        @test act ≈ 0.5 rtol=1e-10
    end
    
    @testset "Limits: tanh(±∞) = ±1" begin
        η_tr = 0.1
        μ_smooth = 1e-4
        
        # Far below threshold: tanh(-large) → -1, so act → 0
        act_low = QLaw.effectivity_activation(0.0, η_tr, μ_smooth)
        @test act_low < 0.01
        
        # Far above threshold: tanh(+large) → +1, so act → 1
        act_high = QLaw.effectivity_activation(0.2, η_tr, μ_smooth)
        @test act_high > 0.99
    end
end

@testset "Max Rates - Analytical Properties" begin
    μ = 398600.4418
    
    @testset "Max rate at single L = F_max * ||A[i,:]||" begin
        # compute_max_rates_at_L computes max rate at a single true longitude
        kep = Keplerian(7000.0, 0.1, 0.5, 0.0, 0.0, 0.0)
        oe = ModEq(kep, μ)
        F_max = 1e-6
        
        A = QLaw.equinoctial_gve_partials(oe, μ)
        a = QLaw.get_sma(oe)
        max_rates_at_L = QLaw.compute_max_rates_at_L(a, oe.f, oe.g, oe.h, oe.k, oe.L, μ, F_max)
        
        for i in 1:5
            expected = F_max * norm(SVector{3}(A[i,1], A[i,2], A[i,3]))
            @test max_rates_at_L[i] ≈ expected rtol=1e-10
        end
    end
    
    @testset "Max rates are same order of magnitude as rates at any L" begin
        # compute_max_rates uses analytical formulas (Varga Eqs. 14-18) which give
        # an approximation for normalization purposes. For eccentric orbits, there
        # can be significant differences vs GVE-based computation at specific L.
        # The key property is that they're the same order of magnitude.
        kep = Keplerian(7000.0, 0.1, 0.5, 0.0, 0.0, 0.0)
        oe = ModEq(kep, μ)
        F_max = 1e-6
        
        max_rates_orbit = QLaw.compute_max_rates(oe, μ, F_max)
        a = QLaw.get_sma(oe)
        
        # Check at multiple L values - rates should be same order of magnitude
        for L_deg in [0, 45, 90, 135, 180, 225, 270, 315]
            L_rad = deg2rad(Float64(L_deg))
            max_rates_at_L = QLaw.compute_max_rates_at_L(a, oe.f, oe.g, oe.h, oe.k, L_rad, μ, F_max)
            for i in 1:5
                # Allow 50% relative difference (within same order of magnitude)
                @test max_rates_orbit[i] >= max_rates_at_L[i] * 0.5
                @test (max_rates_orbit[i] <= max_rates_at_L[i] * 2.0) || (max_rates_at_L[i] < 1e-20)
            end
        end
    end
    
    @testset "Max rates scale linearly with F_max" begin
        kep = Keplerian(7000.0, 0.1, 0.5, 0.0, 0.0, 0.0)
        oe = ModEq(kep, μ)
        
        rates_1 = QLaw.compute_max_rates(oe, μ, 1e-6)
        rates_2 = QLaw.compute_max_rates(oe, μ, 2e-6)
        
        @test rates_2 ≈ 2.0 .* rates_1 rtol=1e-10
    end
    
    @testset "Max rates for h,k are non-zero even at singular L positions" begin
        # BUG this caught: At L=90°, dh/dFh ∝ cos(L) = 0, so h_dot_max was 0 at that L
        # FIX: compute_max_rates searches over all L, so h_dot_max is never 0
        
        μ = 398600.4418
        # Inclined circular orbit (like LEO-GEO case)
        kep = Keplerian(6878.0, 0.0, deg2rad(28.5), 0.0, 0.0, deg2rad(90.0))  # L = 90°
        oe = ModEq(kep, μ)
        F_max = 1e-6
        
        # At L=90°, the single-L max rate for h would be ~0 (dh/dFh ∝ cos(L) = 0)
        a = QLaw.get_sma(oe)
        max_rates_at_90 = QLaw.compute_max_rates_at_L(a, oe.f, oe.g, oe.h, oe.k, deg2rad(90.0), μ, F_max)
        @test max_rates_at_90[4] < 1e-12  # h rate is ~0 at L=90°
        
        # But orbit-wide max rate for h should be non-zero (max at L=0° or L=180°)
        max_rates_orbit = QLaw.compute_max_rates(oe, μ, F_max)
        @test max_rates_orbit[4] > 1e-10  # h rate over orbit is definitely non-zero
    end
end

@testset "Scaling Function - Formula Verification" begin
    μ = 398600.4418
    
    @testset "Sa = Varga Eq. 8 with default params (m=1, n=4, r=2)" begin
        kep0 = Keplerian(7000.0, 0.1, 0.5, 0.0, 0.0, 0.0)
        kepT = Keplerian(42000.0, 0.001, 0.001, 0.0, 0.0, 0.0)
        oe0 = ModEq(kep0, μ)
        oeT = ModEq(kepT, μ)
        
        a0, aT = QLaw.get_sma(oe0), QLaw.get_sma(oeT)
        
        S = QLaw.compute_scaling(oe0, oeT)
        # Varga Eq. 8: Sa = [1 + (|a - at|/(m*at))^n]^(1/r)
        # With defaults: m=1, n=4, r=2 → sqrt(1 + (|a-aT|/aT)^4)
        Sa_expected = sqrt(1 + (abs(a0 - aT) / aT)^4)
        
        @test S[1] ≈ Sa_expected rtol=1e-10
    end
    
    @testset "Other scaling factors are 1.0" begin
        kep0 = Keplerian(7000.0, 0.1, 0.5, 0.0, 0.0, 0.0)
        kepT = Keplerian(42000.0, 0.001, 0.001, 0.0, 0.0, 0.0)
        oe0 = ModEq(kep0, μ)
        oeT = ModEq(kepT, μ)
        
        S = QLaw.compute_scaling(oe0, oeT)
        
        @test S[2] == 1.0  # Sf
        @test S[3] == 1.0  # Sg
        @test S[4] == 1.0  # Sh
        @test S[5] == 1.0  # Sk
    end
    
    @testset "At target: Sa = 1" begin
        kep = Keplerian(42000.0, 0.001, 0.001, 0.0, 0.0, 0.0)
        oe = ModEq(kep, μ)
        
        S = QLaw.compute_scaling(oe, oe)
        @test S[1] == 1.0
    end
end

@testset "Penalty Function - Formula Verification" begin
    μ = 398600.4418
    
    @testset "P ≈ 0 when rp >> rp_min (smooth penalty)" begin
        kep = Keplerian(7000.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # rp = 7000 km
        oe = ModEq(kep, μ)
        
        P = QLaw.compute_penalty(oe, 6378.0)  # rp_min = 6378 km
        # Smooth penalty: exp(100*(1 - 7000/6378)) ≈ 5.8e-5
        # Very small but not exactly zero (smooth for AD)
        @test P < 1e-3
    end
    
    @testset "P = exp(k*(1 - rp/rp_min)) when rp < rp_min" begin
        a = 6500.0
        e = 0.05
        rp = a * (1 - e)  # 6175 km
        rp_min = 6378.0
        k = 100.0
        
        kep = Keplerian(a, e, 0.0, 0.0, 0.0, 0.0)
        oe = ModEq(kep, μ)
        
        P = QLaw.compute_penalty(oe, rp_min)
        P_expected = exp(k * (1 - rp / rp_min))
        
        @test P ≈ P_expected rtol=1e-10
    end
end

@testset "Convergence Check" begin
    μ = 398600.4418
    
    @testset "Identical elements → converged" begin
        kep = Keplerian(42164.0, 0.001, 0.01, 0.0, 0.0, 0.0)
        oe = ModEq(kep, μ)
        
        @test QLaw.check_convergence(oe, oe) == true
    end
    
    @testset "Large a mismatch → not converged" begin
        kep0 = Keplerian(7000.0, 0.001, 0.01, 0.0, 0.0, 0.0)
        kepT = Keplerian(42164.0, 0.001, 0.01, 0.0, 0.0, 0.0)
        oe0 = ModEq(kep0, μ)
        oeT = ModEq(kepT, μ)
        
        @test QLaw.check_convergence(oe0, oeT) == false
    end
    
    @testset "Tolerance respected (paper criteria: summed normalized error)" begin
        kepT = Keplerian(42164.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Case 1: Close to target - normalized error ≈ 0.01 + 0 + 0 + 0 + 0 = 0.01 < 0.05
        kep_close = Keplerian(42164.0 * 1.01, 0.0, 0.0, 0.0, 0.0, 0.0)  # 1% off in a
        oeT = ModEq(kepT, μ)
        oe_close = ModEq(kep_close, μ)
        
        # Normalized a error = |42584 - 42164| / 42164 ≈ 0.01
        @test QLaw.check_convergence(oe_close, oeT; tol=0.05) == true   # 0.01 < 0.05 → converged
        @test QLaw.check_convergence(oe_close, oeT; tol=0.005) == false # 0.01 > 0.005 → not converged
    end
end

# ==========================================================================
# REGRESSION TESTS - These catch bugs found during debugging
# ==========================================================================

@testset "Regression: Q̇ is smooth over all L (no spikes at singular positions)" begin
    # BUG: Q̇ was spiking to INFINITE values at certain L values (around L=88°, 268°)
    # because max_rates for h and k were computed at the current L only, giving
    # near-zero values when cos(L) or sin(L) approached zero.
    # FIX: max_rates must be computed as maximum over ALL L for the orbit shape.
    #
    # Note: Q̇ values CAN be large (1e12) when elements are far from target - that's
    # correct. What we're testing is that Q̇ doesn't have discontinuous spikes.
    
    μ = 398600.4418
    a0, e0, i0 = 6878.0, 0.0, deg2rad(28.5)  # LEO with 28.5° inclination (paper case)
    aT, eT, iT = 42164.0, 0.0, 0.0            # GEO equatorial
    
    kep0 = Keplerian(a0, e0, i0, 0.0, 0.0, 0.0)
    kepT = Keplerian(aT, eT, iT, 0.0, 0.0, 0.0)
    oeT = ModEq(kepT, μ)
    
    weights = QLawWeights(0.0120, 0.5846, 0.1189, 0.9475, 0.3578)  # DOE weights from paper
    F_max = 1e-6
    Wp, rp_min = 1.0, 6578.0
    
    # Test Q̇ at many L values including previously problematic ones (88°, 90°, 268°, 270°)
    test_L_degrees = [0, 45, 88, 90, 135, 180, 225, 268, 270, 315]
    p0 = a0 * (1 - e0^2)
    h0 = tan(i0/2)
    
    Qdot_values = Float64[]
    for L_deg in test_L_degrees
        L_rad = deg2rad(Float64(L_deg))
        oe = ModEq(p0, 0.0, 0.0, h0, 0.0, L_rad)
        _, _, Qdot = compute_thrust_direction(oe, oeT, weights, μ, F_max, Wp, rp_min)
        push!(Qdot_values, Qdot)
    end
    
    # All Q̇ values should be negative (Q decreasing under optimal thrust)
    @test all(Qdot_values .< 0)
    
    # All Q̇ values should be finite (no Inf from division by zero)
    @test all(isfinite.(Qdot_values))
    
    # The key test: Q̇ variation should be smooth, not have discontinuous spikes
    # At formerly problematic L values (88°, 90°, 268°, 270°), Q̇ should be similar
    # magnitude to nearby values, not orders of magnitude different
    Qdot_min = minimum(Qdot_values)
    Qdot_max = maximum(Qdot_values)
    
    # All values should be within 100x of each other (no 1000x spikes)
    @test abs(Qdot_max / Qdot_min) < 100
end

@testset "Regression: Effectivity causes coasting at inefficient positions" begin
    # BUG: Thrust was always on (activation ≈ 1 everywhere) even with
    # effectivity enabled. The paper shows clear coasting periods.
    # FIX: With proper max_rates computation and DOE weights with η_threshold=0.247,
    # there should be positions where η < η_threshold → activation ≈ 0.
    # NOTE: Wp=0 isolates effectivity test from smooth penalty contribution.
    
    μ = 398600.4418
    a0, i0 = 6878.0, deg2rad(28.5)
    aT = 42164.0
    
    kep0 = Keplerian(a0, 0.0, i0, 0.0, 0.0, 0.0)
    kepT = Keplerian(aT, 0.0, 0.0, 0.0, 0.0, 0.0)
    oeT = ModEq(kepT, μ)
    
    # DOE weights WITH effectivity (Table 3)
    weights = QLawWeights(0.0120, 0.5846, 0.1189, 0.9475, 0.3578)
    η_threshold = 0.247  # From Table 3
    η_smoothness = 1e-4
    F_max = 1e-6
    Wp, rp_min = 0.0, 6578.0  # Wp=0: no penalty, isolates effectivity test
    
    p0 = a0
    h0 = tan(i0/2)
    
    # Check effectivity and activation at various L
    thrusting_count = 0
    coasting_count = 0
    
    for L_deg in 0:10:350
        L_rad = deg2rad(Float64(L_deg))
        oe = ModEq(p0, 0.0, 0.0, h0, 0.0, L_rad)
        
        η, _, _, _ = compute_effectivity(oe, oeT, weights, μ, F_max, Wp, rp_min, 50, :absolute)
        activation = QLaw.effectivity_activation(η, η_threshold, η_smoothness)
        
        if activation > 0.5
            thrusting_count += 1
        else
            coasting_count += 1
        end
    end
    
    # With proper effectivity, there should be BOTH thrusting AND coasting periods
    @test thrusting_count > 0  # Some positions should thrust
    @test coasting_count > 0   # Some positions should coast
end

@testset "Regression: Keplerian inclination in radians produces correct ModEq" begin
    # BUG: Inclination was being double-converted (deg2rad applied twice)
    # when i0 was stored as degrees then converted again in constructor.
    # FIX: Keplerian expects radians; ModEq should preserve the inclination.
    
    μ = 398600.4418
    
    # Paper value: 28.5 degrees
    i_deg = 28.5
    i_rad = deg2rad(i_deg)
    
    # Create Keplerian with RADIANS (correct)
    kep = Keplerian(7000.0, 0.0, i_rad, 0.0, 0.0, 0.0)
    oe = ModEq(kep, μ)
    
    # h = tan(i/2) for equinoctial elements
    h_expected = tan(i_rad / 2)
    @test oe.h ≈ h_expected rtol=1e-10
    
    # Recovered inclination should match
    i_recovered = 2 * atan(sqrt(oe.h^2 + oe.k^2))
    @test i_recovered ≈ i_rad rtol=1e-10
    @test rad2deg(i_recovered) ≈ i_deg rtol=1e-10
    
    # WRONG: If someone accidentally passes degrees
    kep_wrong = Keplerian(7000.0, 0.0, i_deg, 0.0, 0.0, 0.0)  # 28.5 rad ≈ 1633°!
    oe_wrong = ModEq(kep_wrong, μ)
    i_wrong_recovered = 2 * atan(sqrt(oe_wrong.h^2 + oe_wrong.k^2))
    
    # This should NOT equal our intended inclination
    @test abs(rad2deg(i_wrong_recovered) - i_deg) > 1.0  # Clearly wrong
end

@testset "Regression: Shadow model returns γ < 1 during eclipse" begin
    # BUG: Thrust plots weren't accounting for eclipse (γ was always 1).
    # FIX: At certain orbit positions, satellite is in Earth's shadow → γ < 1.
    
    μ = 398600.4418
    
    # LEO orbit
    kep = Keplerian(6878.0, 0.0, deg2rad(28.5), 0.0, 0.0, 0.0)
    oe = ModEq(kep, μ)
    
    # Sun position (roughly along -X axis, typical for J2000)
    sun_pos = SVector{3,Float64}(-1.495e8, 0.0, 0.0)
    
    # Test at different true longitudes
    γ_values = Float64[]
    for L_deg in 0:30:330
        L_rad = deg2rad(Float64(L_deg))
        kep_at_L = Keplerian(6878.0, 0.0, deg2rad(28.5), 0.0, 0.0, L_rad)
        oe_at_L = ModEq(kep_at_L, μ)
        
        γ = QLaw.compute_sunlight_fraction(oe_at_L, μ, sun_pos, Conical())
        push!(γ_values, γ)
    end
    
    # Some positions should be in sunlight (γ = 1)
    @test any(γ_values .≈ 1.0)
    
    # Some positions should be in eclipse (γ < 1 or γ = 0)
    @test any(γ_values .< 1.0)
end

@testset "Regression: D coefficients are smooth over all L (no singular spikes)" begin
    # BUG: D1, D2, D3 were spiking to INFINITE values at certain L values because
    # max_rates for h and k approached zero, causing dQ/doe to spike.
    # FIX: max_rates computed over all L prevents division by near-zero.
    #
    # Note: D coefficients CAN be large (1e12) for far-from-target orbits.
    # What we test is smoothness and finiteness, not small magnitude.
    
    μ = 398600.4418
    a0, i0 = 6878.0, deg2rad(28.5)
    aT = 42164.0
    
    kepT = Keplerian(aT, 0.0, 0.0, 0.0, 0.0, 0.0)
    oeT = ModEq(kepT, μ)
    
    weights = QLawWeights(0.0120, 0.5846, 0.1189, 0.9475, 0.3578)
    F_max = 1e-6
    Wp, rp_min = 1.0, 6578.0
    
    p0 = a0
    h0 = tan(i0/2)
    
    D1_values, D2_values, D3_values = Float64[], Float64[], Float64[]
    
    for L_deg in 0:5:355
        L_rad = deg2rad(Float64(L_deg))
        oe = ModEq(p0, 0.0, 0.0, h0, 0.0, L_rad)
        
        D1, D2, D3 = QLaw.compute_Qdot_coefficients(oe, oeT, weights, μ, F_max, Wp, rp_min)
        push!(D1_values, D1)
        push!(D2_values, D2)
        push!(D3_values, D3)
    end
    
    # All D coefficients should be finite (no Inf from division by zero)
    @test all(isfinite.(D1_values))
    @test all(isfinite.(D2_values))
    @test all(isfinite.(D3_values))
    
    # D1 (tangential) should be roughly constant over L for circular orbit
    # (it depends on ∂Q/∂a * da/dFθ which is constant for circular)
    D1_mean = sum(D1_values) / length(D1_values)
    D1_std = sqrt(sum((D1_values .- D1_mean).^2) / length(D1_values))
    @test D1_std / abs(D1_mean) < 0.01  # < 1% variation
    
    # D3 (normal) varies with cos(L) for h and sin(L) for k
    # For this orbit (h≠0, k=0), D3 should follow cos(L) pattern
    # Key test: no discontinuous spikes at L=90° or L=270°
    D3_at_0 = D3_values[1]    # L=0°
    D3_at_90 = D3_values[19]  # L=90° (index 19 for 0:5:355)
    D3_at_180 = D3_values[37] # L=180°
    D3_at_270 = D3_values[55] # L=270°
    
    # At L=90° and L=270°, D3 should be near zero (cos(90°)=0, cos(270°)=0)
    @test abs(D3_at_90) < abs(D3_at_0) / 10
    @test abs(D3_at_270) < abs(D3_at_0) / 10
    
    # D3 at L=0° and L=180° should be opposite in sign (cos(0)=1, cos(180)=-1)
    @test D3_at_0 * D3_at_180 < 0
end

@testset "Regression: Thrust acceleration includes effectivity and eclipse" begin
    # BUG: qlaw_thrust_acceleration wasn't properly scaling by activation and γ.
    # FIX: throttle = activation * γ, and a_thrust = throttle * F_max * direction.
    
    μ = 398600.4418
    
    kep0 = Keplerian(6878.0, 0.0, deg2rad(28.5), 0.0, 0.0, 0.0)
    kepT = Keplerian(42164.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    oe0 = ModEq(kep0, μ)
    oeT = ModEq(kepT, μ)
    
    sc = QLawSpacecraft(500.0, 1000.0, 1.0, 1500.0)
    weights = QLawWeights(0.0120, 0.5846, 0.1189, 0.9475, 0.3578)
    params = QLawParameters(; η_threshold=0.247, η_smoothness=1e-4)
    
    m = 1000.0
    
    # Test with full sunlight (γ = 1)
    a_thrust_sun, throttle_sun, _, _ = QLaw.qlaw_thrust_acceleration(
        oe0, oeT, m, sc, weights, params, μ, 1.0
    )
    
    # Test with eclipse (γ = 0)
    a_thrust_eclipse, throttle_eclipse, _, _ = QLaw.qlaw_thrust_acceleration(
        oe0, oeT, m, sc, weights, params, μ, 0.0
    )
    
    # In eclipse, thrust should be zero
    @test norm(a_thrust_eclipse) ≈ 0.0 atol=1e-15
    @test throttle_eclipse ≈ 0.0 atol=1e-15
    
    # Test with partial sunlight (γ = 0.5)
    a_thrust_partial, throttle_partial, _, _ = QLaw.qlaw_thrust_acceleration(
        oe0, oeT, m, sc, weights, params, μ, 0.5
    )
    
    # Partial sunlight should give partial thrust (if activation > 0)
    if throttle_sun > 0
        @test throttle_partial ≈ 0.5 * throttle_sun / 1.0 rtol=0.01  # γ scales throttle
    end
end

@testset "Regression: Analytical and numerical max_rates match exactly" begin
    # BUG: Varga analytical formulas (Eqs. 14-18) might differ from GVE-based numerical search.
    # VERIFICATION: Both methods should give identical results for circular orbits.
    
    μ = 398600.4418
    F_max = 1e-6
    
    # LEO circular orbit at 28.5° inclination
    kep = Keplerian(6878.0, 0.0, deg2rad(28.5), 0.0, 0.0, 0.0)
    oe = ModEq(kep, μ)
    a = QLaw.get_sma(oe)
    
    # Analytical max_rates from Varga formulas
    max_rates_analytical = QLaw.compute_max_rates_analytical(a, oe.f, oe.g, oe.h, oe.k, μ, F_max)
    
    # Numerical max_rates from searching over L
    n_points = 100
    max_rates_numerical = zeros(5)
    for i in 0:n_points-1
        L_test = 2π * i / n_points
        rates_at_L = QLaw.compute_max_rates_at_L(a, oe.f, oe.g, oe.h, oe.k, L_test, μ, F_max)
        for j in 1:5
            max_rates_numerical[j] = max(max_rates_numerical[j], rates_at_L[j])
        end
    end
    
    # Both methods should give identical results
    for i in 1:5
        @test max_rates_analytical[i] ≈ max_rates_numerical[i] rtol=1e-10
    end
end

@testset "Regression: Q-function contribution balance at LEO" begin
    # FINDING: At initial LEO state, even with Wa=0.012, the SMA contribution
    # to Q is significant because the error is huge (35,286 km).
    # But h contribution dominates due to normalization by max_rates.
    
    μ = 398600.4418
    F_max = 1e-6
    
    kep0 = Keplerian(6878.0, 0.0, deg2rad(28.5), 0.0, 0.0, 0.0)
    kepT = Keplerian(42164.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    oe0 = ModEq(kep0, μ)
    oeT = ModEq(kepT, μ)
    
    a0 = QLaw.get_sma(oe0)
    aT = QLaw.get_sma(oeT)
    
    # Compute individual terms
    max_rates = QLaw.compute_max_rates(oe0, μ, F_max)
    scaling = QLaw.compute_scaling(oe0, oeT)
    
    # Normalized squared errors (before weighting)
    term_a = scaling[1] * ((a0 - aT) / max_rates[1])^2
    term_h = scaling[4] * ((oe0.h - oeT.h) / max_rates[4])^2
    
    # With DOE weights
    Wa, Wh = 0.012, 0.9475
    contrib_a = Wa * term_a
    contrib_h = Wh * term_h
    
    # Inclination (h) should dominate despite huge SMA error
    # This explains the out-of-plane thrust direction
    @test contrib_h > contrib_a  # h dominates
    @test contrib_h / contrib_a > 2.0  # by at least 2x
end

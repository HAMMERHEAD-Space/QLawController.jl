_diff_tspan = 200.0
_diff_params = QLawParameters(;
    effectivity_search=GridSearch(), n_search_points=20, η_smoothness=0.1
)

@testset "Full Solve Initial State Sensitivity" begin
    @testset "ForwardDiff vs FiniteDiff" begin
        function solve_from_initial_sma(x)
            a0 = x[1]
            p0 = a0 * (1.0 - 0.001^2)
            oe0_local = ModEq(p0, 0.001, 0.0, _oe0.h, _oe0.k, _oe0.L)

            prob_local = qlaw_problem(
                oe0_local,
                _oeT,
                (0.0, _diff_tspan),
                _μ,
                _spacecraft;
                weights=_weights,
                qlaw_params=_diff_params,
                dynamics_model=_dynamics_model,
            )

            sol = solve(prob_local; abstol=1e-10, reltol=1e-10, ODE_solver=Vern9())
            u_final = sol.trajectory.u[end]
            return [u_final[1], u_final[2], u_final[3], u_final[4], u_final[5], u_final[7]]
        end

        x0 = [6778.0]

        J_fd = FiniteDiff.finite_difference_jacobian(solve_from_initial_sma, x0)
        J_ad = ForwardDiff.jacobian(solve_from_initial_sma, x0)

        @test all(isfinite, J_ad)
        @test J_fd ≈ J_ad rtol = 1e-3
    end
end

@testset "Full Solve Weight Sensitivity" begin
    @testset "ForwardDiff vs FiniteDiff" begin
        function solve_from_weights(x)
            w_local = QLawWeights(x[1], x[2], x[3], x[4], x[5])

            prob_local = qlaw_problem(
                _oe0,
                _oeT,
                (0.0, _diff_tspan),
                _μ,
                _spacecraft;
                weights=w_local,
                qlaw_params=_diff_params,
                dynamics_model=_dynamics_model,
            )

            sol = solve(prob_local; abstol=1e-10, reltol=1e-10, ODE_solver=Vern9())
            u_final = sol.trajectory.u[end]
            return [u_final[1], u_final[2], u_final[3], u_final[4], u_final[5], u_final[7]]
        end

        x0 = [1.0, 1.0, 1.0, 1.0, 1.0]

        J_fd = FiniteDiff.finite_difference_jacobian(solve_from_weights, x0)
        J_ad = ForwardDiff.jacobian(solve_from_weights, x0)

        @test all(isfinite, J_ad)
        @test J_fd ≈ J_ad rtol = 1e-3
    end
end

@testset "Full Solve Spacecraft Parameter Sensitivity" begin
    @testset "ForwardDiff vs FiniteDiff" begin
        function solve_from_spacecraft_params(x)
            sc_local = QLawSpacecraft(500.0, 1000.0, x[1], x[2])

            prob_local = qlaw_problem(
                _oe0,
                _oeT,
                (0.0, _diff_tspan),
                _μ,
                sc_local;
                weights=_weights,
                qlaw_params=_diff_params,
                dynamics_model=_dynamics_model,
            )

            sol = solve(prob_local; abstol=1e-10, reltol=1e-10, ODE_solver=Vern9())
            u_final = sol.trajectory.u[end]
            return [u_final[1], u_final[2], u_final[3], u_final[4], u_final[5], u_final[7]]
        end

        x0 = [1.0, 3000.0]

        J_fd = FiniteDiff.finite_difference_jacobian(solve_from_spacecraft_params, x0)
        J_ad = ForwardDiff.jacobian(solve_from_spacecraft_params, x0)

        @test all(isfinite, J_ad)
        @test J_fd ≈ J_ad rtol = 1e-3
    end
end

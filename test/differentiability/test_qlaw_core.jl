@testset "compute_Q_from_vec_with_rates State Differentiability" begin
    for backend in _BACKENDS
        testname = "compute_Q_from_vec_with_rates " * backend[1]
        @testset "$testname" begin
            f_fd, df_fd = value_and_jacobian(
                (x) -> [
                    compute_Q_from_vec_with_rates(
                        SVector{5}(x...), _oeT_vec, _W_vec, _max_rates, 1.0, 6578.0
                    ),
                ],
                AutoFiniteDiff(),
                Array(_oe_vec),
            )

            f_ad, df_ad = value_and_jacobian(
                (x) -> [
                    compute_Q_from_vec_with_rates(
                        SVector{5}(x...), _oeT_vec, _W_vec, _max_rates, 1.0, 6578.0
                    ),
                ],
                backend[2],
                Array(_oe_vec),
            )

            @test f_fd ≈ f_ad
            @test df_fd ≈ df_ad rtol = 1e-4
        end
    end
end

@testset "compute_dQ_doe_analytical State Differentiability" begin
    for backend in _BACKENDS
        testname = "compute_dQ_doe_analytical " * backend[1]
        @testset "$testname" begin
            f_fd, df_fd = value_and_jacobian(
                (x) -> Array(
                    compute_dQ_doe_analytical(
                        SVector{5}(x...), _oeT_vec, _W_vec, _max_rates, 1.0, 6578.0
                    ),
                ),
                AutoFiniteDiff(),
                Array(_oe_vec),
            )

            f_ad, df_ad = value_and_jacobian(
                (x) -> Array(
                    compute_dQ_doe_analytical(
                        SVector{5}(x...), _oeT_vec, _W_vec, _max_rates, 1.0, 6578.0
                    ),
                ),
                backend[2],
                Array(_oe_vec),
            )

            @test f_fd ≈ f_ad
            @test df_fd ≈ df_ad rtol = 1e-4
        end
    end
end

@testset "compute_thrust_direction State Differentiability" begin
    for backend in _BACKENDS
        testname = "compute_thrust_direction " * backend[1]
        @testset "$testname" begin
            function thrust_dir_wrapper(x)
                oe_test = ModEq(x[1], x[2], x[3], x[4], x[5], x[6])
                α, β, Qdot_n = compute_thrust_direction(
                    oe_test, _oeT, _weights, _μ, _F_max, _params
                )
                return [α, β, Qdot_n]
            end

            x0 = [_oe0.p, _oe0.f, _oe0.g, _oe0.h, _oe0.k, _oe0.L]

            f_fd, df_fd = value_and_jacobian(thrust_dir_wrapper, AutoFiniteDiff(), x0)
            f_ad, df_ad = value_and_jacobian(thrust_dir_wrapper, backend[2], x0)

            @test f_fd ≈ f_ad
            @test isapprox(df_fd, df_ad; rtol=0.05)
        end
    end
end

@testset "effectivity_activation Differentiability" begin
    for backend in _BACKENDS
        testname = "effectivity_activation " * backend[1]
        @testset "$testname" begin
            f_fd, df_fd = value_and_derivative(
                (x) -> effectivity_activation(x, -0.01, 1e-4), AutoFiniteDiff(), 0.5
            )

            f_ad, df_ad = value_and_derivative(
                (x) -> effectivity_activation(x, -0.01, 1e-4), backend[2], 0.5
            )

            @test f_fd ≈ f_ad
            @test df_fd ≈ df_ad atol = 1e-5
        end
    end
end

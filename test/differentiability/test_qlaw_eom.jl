@testset "Q-Law EOM State Differentiability" begin
    for backend in _BACKENDS
        testname = "Q-Law EOM State " * backend[1]
        @testset "$testname" begin
            f_fd, df_fd = value_and_jacobian(
                (x) -> Array(qlaw_eom(SVector{7}(x...), _ps, _t, _prob)),
                AutoFiniteDiff(),
                Array(_u0),
            )

            f_ad, df_ad = value_and_jacobian(
                (x) -> Array(qlaw_eom(SVector{7}(x...), _ps, _t, _prob)),
                backend[2],
                Array(_u0),
            )

            @test f_fd ≈ f_ad
            @test df_fd ≈ df_ad atol = 1e-5
        end
    end
end

@testset "Q-Law EOM Time Differentiability" begin
    for backend in _BACKENDS
        testname = "Q-Law EOM Time " * backend[1]
        @testset "$testname" begin
            f_fd, df_fd = value_and_derivative(
                (x) -> Array(qlaw_eom(_u0, _ps, x, _prob)), AutoFiniteDiff(), _t
            )

            f_ad, df_ad = value_and_derivative(
                (x) -> Array(qlaw_eom(_u0, _ps, x, _prob)), backend[2], _t
            )

            @test f_fd ≈ f_ad
            @test df_fd ≈ df_ad atol = 1e-5
        end
    end
end

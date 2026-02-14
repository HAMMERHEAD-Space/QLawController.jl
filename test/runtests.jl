using QLawController

using AstroCoords
using AstroForceModels
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Test

@testset "QLawController.jl" begin
    include("test_types.jl")
    include("test_core.jl")
    include("test_api.jl")
    include("test_leo_to_geo.jl")
end

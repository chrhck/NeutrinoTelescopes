using NeutrinoTelescopes
using Random
using Test
using StaticArrays
using LinearAlgebra
using Distributions
using DataStructures

@testset "NeutrinoTelescopes.jl" begin

    @testset "Utils" begin
        using NeutrinoTelescopes.Utils

        @testset "integrate_gauss_quad" begin
            let f = x->x^3
                @test integrate_gauss_quad(f, 0., 1.) ≈ 1/4
            end

            let f = cos, a=0., b=0.5
                @test integrate_gauss_quad(f, a, b) ≈ sin(b) - sin(a)
            end

        end

        @testset "sph_to_cart" begin
            let theta = 0.0, phi = 0.0
                @test sph_to_cart(theta, phi) ≈ SA[0.0, 0.0, 1.0]
            end

            let theta = 0.0, phi = π
                @test sph_to_cart(theta, phi) ≈ SA[0.0, 0.0, 1.0]
            end

            let theta = π / 2, phi = 0
                @test sph_to_cart(theta, phi) ≈ SA[1.0, 0.0, 0.0]
            end

            let theta = π / 2, phi = π
                @test sph_to_cart(theta, phi) ≈ SA[-1.0, 0.0, 0.0]
            end

            let theta = π / 2, phi = π / 2
                @test sph_to_cart(theta, phi) ≈ SA[0.0, 1.0, 0.0]
            end
        end

        @testset "CategoricalSetDistribution" begin
            let pdist = CategoricalSetDistribution(OrderedSet([:EMinus, :EPlus]), [1.0, 0.0])
                @test rand(pdist) === :EMinus
            end

            let pdist = CategoricalSetDistribution(OrderedSet([:EMinus, :EPlus]), Categorical([1.0, 0.0]))
                @test rand(pdist) === :EMinus
            end

            let err = nothing
                try
                    CategoricalSetDistribution(OrderedSet([:EMinus, :EPlus]), Categorical([1,]))
                catch err
                end
                @test err isa Exception
            end

            let err = nothing
                try
                    CategoricalSetDistribution(OrderedSet[:EMinus, :EPlus], [1.0])
                catch err
                end
                @test err isa Exception
            end
        end
    end

    @testset "Injectors" begin
        using NeutrinoTelescopes.EventGeneration.Injectors
        using NeutrinoTelescopes.Types


        struct TestVolumeType{T} <: VolumeType end

        let err = nothing, vol = TestVolumeType{Real}()
            try
                sample_volume(vol)
            catch err
            end

            @test err isa Exception
        end

        let center = SA[10.0, 10, 50], height = 100.0, radius = 20.0
            vol = Cylinder(center, height, radius)
            points = [rand(vol) for _ in 1:10000]

            function is_inside(p)
                isin_height = p[3] <= (height / 2 + center[3])
                isin_radius = norm(p[1:2] - center[1:2]) <= radius
                return isin_height && isin_radius
            end

            @test all(map(is_inside, points))
        end

        let pdist, vol, ang_dist, edist, inj, particle
            pdist = CategoricalSetDistribution(OrderedSet(ParticleTypes), [0.1, 0.1, 0.8])
            vol = Cylinder(SA[10.0, 10, 50], 100.0, 20.0)

            ang_dist = UniformAngularDistribution()
            edist = Pareto(1, 100) + 100

            inj = VolumeInjector(vol, edist, pdist, ang_dist)
            particle = rand(inj)

            @test norm(particle.direction) ≈ 1


        end
    end





end
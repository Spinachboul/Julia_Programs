using LinearAlgebra
using Random
using Profile
using ProfileView
using FlameGraphs
# using Kriging: AbstractKriging

abstract type  AbstractKriging end
mutable struct SGP <: AbstractKriging
    name::String

    # Options
    options::Dict{String, Any}

    # Inducing inputs
    Z::Union{Nothing, Matrix{Float64}}

    # Woodbury data
    woodbury_data::Dict{String, Union{Nothing, Vector{Float64}, Matrix{Float64}}}

    # Optimal parameters
    optimal_par::Dict{String, Any}

    # Optimal noise
    optimal_noise::Union{Nothing, Float64}
end

function SGP()
    options = Dict(
        "corr" => "squar_exp",
        "poly" => "constant",
        "theta_bounds" => [1e-6, 1e2],
        "noise0" => [1e-2],
        "hyper_opt" => "Cobyla",
        "eval_noise" => true,
        "nugget" => 1000.0 * eps(Float64),
        "method" => "FITC",
        "n_inducing" => 10
    )

    woodbury_data = Dict("vec" => nothing, "inv" => nothing)

    optimal_par = Dict()

    optimal_noise = nothing

    sgp = SGP("SGP", options, nothing, woodbury_data, optimal_par, optimal_noise)

    return sgp
end

function set_inducing_inputs!(sgp::SGP, Z::Union{Nothing, Matrix{Float64}}=nothing, normalize::Bool=false)
    
    if isnothing(Z)
        nt = size(sgp.optimal_par["training_points"][1][1], 1)
        nz = sgp.options["n_inducing"]
        X = sgp.optimal_par["training_points"][1][1]  # [nt,nx]
        random_idx = randperm(nt)[1:nz]
        sgp.Z = X[random_idx, :]  # [nz,nx]
    else
        Z = copy(Z)
        if size(Z, 2) != sgp.nx
            throw(DimensionMismatch("Z.shape[2] != X.shape[2]"))
        end
        sgp.Z = Z  # [nz,nx]
        if normalize
            X = sgp.optimal_par["training_points"][1][1]  # [nt,nx]
            y = sgp.optimal_par["training_points"][1][2]
            X_offset, X_scale, _, _, _ = standardization(X, y)
            sgp.Z = (sgp.Z .- X_offset') ./ X_scale'
        end

    end
    
end

function _new_train(sgp::SGP)
    if isnothing(sgp.Z)
        set_inducing_inputs!(sgp)
    end

    Y = sgp.optimal_par["training_points"][1][2]
    _, output_dim = size(Y)
    if output_dim > 1
        throw(NotImplementedError("SGP does not support vector-valued function"))
    end

    if sgp.options["use_het_noise"]
        throw(NotImplementedError("SGP does not support heteroscedastic noise"))
    end

    if !sgp.is_continuous
        throw(NotImplementedError("SGP does not support mixed-integer variables"))
    end

    if sgp.options["hyper_opt"] != "Cobyla"
        throw(NotImplementedError("SGP works only with COBYLA internal optimizer"))
    end

    return
end

function _compute_K(A::Matrix{Float64}, B::Matrix{Float64}, theta::Vector{Float64}, sigma2::Float64)
    dx = differences(A, B)
    d = componentwise_distance(dx)
    r = correlation_types[sgp.options["corr"]](theta, d)
    R = reshape(r, size(A, 1), size(B, 1))
    return sigma2 * R
end
                                                                                                                                                                                                                                                                                                    w
function _fitc(sgp::SGP, X::Matrix{Float64}, Y::Matrix{Float64}, Z::Matrix{Float64}, theta::Vector{Float64}, sigma2::Float64, noise::Float64, nugget::Float64)
    Knn = fill(sigma2, size(Y, 1))
    Kmm = _compute_K(Z, Z, theta, sigma2) + I * nugget
    Kmn = _compute_K(Z, X, theta, sigma2)

    U = cholesky(Kmm).L

    Ui = inv(U)
    V = Ui * Kmn

    eta2 = noise

    nu = Knn - sum(V.^2, dims=1)' + eta2
    beta = 1.0 ./ nu

    A = I + V * diagm(beta) * V'

    L = cholesky(A).L
    Li = inv(L)

    a = Y .* beta'
    b = Li * (V * a)

    likelihood = -0.5 * (
        # num_data * log(2.0 * π)   # constant term ignored in reduced likelihood
        + sum(log.(nu))
        + 2.0 * sum(log.(diag(L)))
        + sum(a' * Y)
        - sum(b.^2)
    )

    LiUi = Li * Ui
    woodbury_vec = LiUi' * b
    woodbury_inv = Ui' * Ui - LiUi' * LiUi

    return likelihood, woodbury_vec, woodbury_inv
end

function _vfe(sgp::SGP, X::Matrix{Float64}, Y::Matrix{Float64}, Z::Matrix{Float64}, theta::Vector{Float64}, sigma2::Float64, noise::Float64, nugget::Float64)
    mean = zeros(size(Y, 1), 1)
    Y .-= mean

    Kmm = _compute_K(Z, Z, theta, sigma2) + I * nugget
    Kmn = _compute_K(Z, X, theta, sigma2)

    U = cholesky(Kmm).L     

    Ui = inv(U)
    V = Ui * Kmn

    beta = 1.0 ./ max.(noise, nugget)

    A = beta * V * V'

    B = I + A
    L = cholesky(B).L
    Li = inv(L)

    b = beta * Li * (V * Y)

    likelihood = -0.5 * (
        # nt * log(2.0 * π)   # constant term ignored in reduced likelihood
        -size(Y, 1) * log(beta)
        + 2.0 * sum(log.(diag(L)))
        + beta * sum(Y.^2)
        - sum(b.^2)
        + size(Y, 1) * beta * sigma2
        - sum(A)
    )

    LiUi = Li * Ui
    Bi = I + Li' * Li
    woodbury_vec = LiUi' * b
    woodbury_inv = Ui' * Bi * Ui

    return likelihood, woodbury_vec, woodbury_inv
end

function _predict_values(sgp::SGP, x::Matrix{Float64})
    Kx = _compute_K(x, sgp.Z, sgp.optimal_par["theta"], sgp.optimal_par["sigma2"])
    mu = Kx * sgp.woodbury_data["vec"]
    return mu
end

function _predict_variances(sgp::SGP, x::Matrix{Float64})
    Kx = _compute_K(sgp.Z, x, sgp.optimal_par["theta"], sgp.optimal_par["sigma2"])
    Kxx = fill(sgp.optimal_par["sigma2"], size(x, 1))
    var = (Kxx .- sum((sgp.woodbury_data["inv"]' * Kx) .* Kx, dims=1)') .+ sgp.optimal_noise
    return max.(var, 1e-15)
end


# Create an instance of SGP with appropriate parameters
sgp = SGP()

# Profile the set_inducing_inputs! function
@ProfileView set_inducing_inputs!(sgp)

# Print profiling results
Profile.print()

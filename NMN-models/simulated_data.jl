using Lux, Zygote, Distributions, Optimisers, Random, MLUtils
using ComponentArrays
using Statistics
using DataFrames, CSV

include("functions.jl")

rng = Random.default_rng()
Random.seed!(1)

Nid = 3000
Nid_val = 1000
Nruns = 100
Sim_covs = 2

time_const = []
time_lin = []
time_NN = []

loss_const = []
loss_lin = []
loss_NN = []

loss_const_val = []
loss_lin_val = []
loss_NN_val = []

BS_const_vec = []
BS_lin_vec = []
BS_NN_vec = []

MSE11_const_val_vec = []
MSE11_lin_val_vec = []
MSE11_NN_val_vec = []

MSE12_const_val_vec = []
MSE12_lin_val_vec = []
MSE12_NN_val_vec = []

MSE13_const_val_vec = []
MSE13_lin_val_vec = []
MSE13_NN_val_vec = []

for i in 1:Nruns
    ################################# Training data ##############################################
    TIME, trans12, trans13, state, covariates, lambda12, lambda13 = simulate_data(Nid, Sim_covs)
    lambda11 = 1 .-(lambda12 .+lambda13) ## for calculations later
    TIME = introduce_interval_censoring(TIME, trans12, trans13, 1)
    TIME = create_time_matrix(TIME)
    TIME, trans12, trans13 = right_censoring(TIME, trans12, trans13, 0.5)
    ################################# Validation data ##############################################
    TIME_val, trans12_val, trans13_val, state_val, COVS_val, lambda12_val, lambda13_val = simulate_data(Nid_val, Sim_covs)
    lambda11_val = 1 .-(lambda12_val .+lambda13_val) ## for calculations later
    TIME_val = introduce_interval_censoring(TIME_val, trans12_val, trans13_val, 1)
    TIME_val = create_time_matrix(TIME_val)
    TIME_val, trans12_val, trans13_val = right_censoring(TIME_val, trans12_val, trans13_val, 0.5)

    COVS = covariates
    train_loader = DataLoader((data=COVS, label=TIME'), batchsize=128, shuffle=true)
    #################################################################################################################
    ############## ANN MODEL #####################################################################################
    #################################################################################################################
    NN_model = NN_model_two_layer(2, 50, swish)
    ps_λ12, NN_ls = Lux.setup(rng, NN_model)
    ps_λ13, _ = Lux.setup(rng, NN_model)
    NN_parameters = ComponentArray{Float32}()
    NN_parameters = ComponentArray(NN_parameters;ps_λ12)
    NN_parameters = ComponentArray(NN_parameters;ps_λ13)
    opt = Adam(0.001)
    opt_state = Optimisers.setup(opt, NN_parameters)
    NN_parameters, NN_lossvec, NN_time1 = training(200, NN_model, trans12, trans13, COVS, TIME, NN_parameters, NN_ls, train_loader, opt_state)
    opt = Adam(0.0001)
    opt_state = Optimisers.setup(opt, NN_parameters)
    NN_parameters, NN_lossvec, NN_time2 = training(300, NN_model, trans12, trans13, COVS, TIME, NN_parameters, NN_ls, train_loader, opt_state)

    NN_loss = NN_lossvec[end]
    NN_loss_val = loss_fn(NN_parameters, NN_ls, trans12_val, trans13_val, COVS_val, TIME_val', NN_model)[1]
    ######################################## Prediction #############################
    NN_pred_11_val, NN_pred_12_val, NN_pred_13_val = predict(NN_model, NN_parameters, COVS_val, NN_ls)
    MSE12_NN_val = sum(sqrt.((lambda12_val - NN_pred_12_val).^2) )* (1/length(lambda12_val))
    MSE13_NN_val = sum(sqrt.((lambda13_val - NN_pred_13_val).^2) )* (1/length(lambda13_val))
    MSE11_NN_val = sum(sqrt.((lambda11_val - NN_pred_11_val).^2) )* (1/length(lambda11_val))

    BS_NN = Brier_score_multiple_timepoints(TIME_val, NN_pred_12_val, NN_pred_13_val, trans12_val, trans13_val, [5])[1][1]

    NN_time = NN_time1 + NN_time2
    push!(time_NN, NN_time)
    push!(loss_NN, NN_loss)
    push!(loss_NN_val, NN_loss_val)
    push!(MSE11_NN_val_vec, MSE11_NN_val)
    push!(MSE12_NN_val_vec, MSE12_NN_val)
    push!(MSE13_NN_val_vec, MSE13_NN_val)
    push!(BS_NN_vec, BS_NN)
    ##################################################################################################################
    #################################################################################################################
    #################################################################################################################
    ############## LINEAR MODEL #####################################################################################
    #################################################################################################################

    lin_model = Dense(2, 1)
    ps_λ12, lin_ls = Lux.setup(rng, lin_model)
    ps_λ13, _ = Lux.setup(rng, lin_model)

    lin_parameters = ComponentArray{Float32}()
    lin_parameters = ComponentArray(lin_parameters;ps_λ12)
    lin_parameters = ComponentArray(lin_parameters;ps_λ13)
    opt = Adam(0.01)
    opt_state = Optimisers.setup(opt, lin_parameters)

    ### training ##########################################
    lin_parameters, lin_lossvec, lin_time = training(500, lin_model, trans12, trans13, COVS, TIME, lin_parameters, lin_ls,train_loader, opt_state)

    lin_loss = lin_lossvec[end]
    lin_loss_val = loss_fn(lin_parameters, lin_ls, trans12_val, trans13_val, COVS_val, TIME_val', lin_model)[1]
    ######################################## Prediction #############################
    lin_pred_11_val, lin_pred_12_val, lin_pred_13_val = predict(lin_model, lin_parameters, COVS_val, lin_ls)
    MSE12_lin_val = sum(sqrt.((lambda12_val - lin_pred_12_val).^2) )* (1/length(lambda12_val))
    MSE13_lin_val = sum(sqrt.((lambda13_val - lin_pred_13_val).^2) )* (1/length(lambda13_val))
    MSE11_lin_val = sum(sqrt.((lambda11_val - lin_pred_11_val).^2) )* (1/length(lambda11_val))
    
    BS_lin = Brier_score_multiple_timepoints(TIME_val, lin_pred_12_val, lin_pred_13_val, trans12_val, trans13_val, [5])[1][1]

    push!(time_lin, lin_time)
    push!(loss_lin, lin_loss)
    push!(loss_lin_val, lin_loss_val)
    push!(MSE11_lin_val_vec, MSE11_lin_val)
    push!(MSE12_lin_val_vec, MSE12_lin_val)
    push!(MSE13_lin_val_vec, MSE13_lin_val)
    push!(BS_lin_vec, BS_lin)
    #################################################################################################################
    #################################################################################################################
    ############## Constant MODEL #####################################################################################
    #################################################################################################################
    COVS = ones(1, Nid)
    COVS_val = ones(1, Nid_val)
    train_loader = DataLoader((data=COVS, label=TIME'), batchsize=128, shuffle=true)
    const_model = Dense(1, 1)
    ps_λ12, const_ls = Lux.setup(rng, const_model)
    ps_λ13, _ = Lux.setup(rng, const_model)

    const_parameters = ComponentArray{Float32}()
    const_parameters = ComponentArray(const_parameters;ps_λ12)
    const_parameters = ComponentArray(const_parameters;ps_λ13)
    opt = Adam(0.01)
    opt_state = Optimisers.setup(opt, const_parameters)

    ### training ##########################################
    const_parameters, const_lossvec, const_time = training(500, const_model, trans12, trans13, COVS, TIME, const_parameters, const_ls,train_loader, opt_state)

    const_loss = const_lossvec[end]
    const_loss_val = loss_fn(const_parameters, const_ls, trans12_val, trans13_val, COVS_val, TIME_val', const_model)[1]
    ######################################## Prediction #############################
    const_pred_11_val, const_pred_12_val, const_pred_13_val = predict(const_model, const_parameters, COVS_val, const_ls)
    MSE12_const_val = sum(sqrt.((lambda12_val - const_pred_12_val).^2) )* (1/length(lambda12_val))
    MSE13_const_val = sum(sqrt.((lambda13_val - const_pred_13_val).^2) )* (1/length(lambda13_val))
    MSE11_const_val = sum(sqrt.((lambda11_val - const_pred_11_val).^2) )* (1/length(lambda11_val))
    ############################ BS ##############################
    BS_const = Brier_score_multiple_timepoints(TIME_val, const_pred_12_val, const_pred_13_val, trans12_val, trans13_val, [5])[1][1]

    push!(time_const, const_time)
    push!(loss_const, const_loss)
    push!(loss_const_val, const_loss_val)
    push!(MSE11_const_val_vec, MSE11_const_val)
    push!(MSE12_const_val_vec, MSE12_const_val)
    push!(MSE13_const_val_vec, MSE13_const_val)
    push!(BS_const_vec, BS_const)
end


df = DataFrame(time_const = time_const,
                time_lin = time_lin,
                time_NN = time_NN,
                loss_const = loss_const, loss_lin = loss_lin, loss_NN = loss_NN,
                loss_const_val = loss_const_val, loss_lin_val = loss_lin_val, loss_NN_val = loss_NN_val,
                BS_const = BS_const_vec,
                BS_lin = BS_lin_vec,
                BS_NN = BS_NN_vec,
                MSE11_const_val = MSE11_const_val_vec,
                MSE12_const_val = MSE12_const_val_vec,
                MSE13_const_val = MSE13_const_val_vec,
                MSE11_lin_val = MSE11_lin_val_vec,
                MSE12_lin_val = MSE12_lin_val_vec,
                MSE13_lin_val = MSE13_lin_val_vec,
                MSE11_NN_val = MSE11_NN_val_vec,
                MSE12_NN_val = MSE12_NN_val_vec,
                MSE13_NN_val = MSE13_NN_val_vec)
##

#CSV.write("performance_metrics.csv", df)

#############################################################

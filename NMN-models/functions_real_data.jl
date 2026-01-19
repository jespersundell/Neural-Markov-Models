function train_test_data_split(df, frac)
    state = df.state
    unique_state = unique(state)
    Number_of_groups = length(unique_state)
    Nobs_grouped_state = similar(unique_state)

    for i in eachindex(unique_state)
        Nobs_grouped_state[i] = (count(state .== unique_state[i]) )
    end 
    Nobs_group_testdata = Int.(round.(frac .* Nobs_grouped_state ) )

    tot = 0.0
    accumulating_Nobs = zeros(length(Nobs_grouped_state))
    for i in eachindex(Nobs_grouped_state)
        tot += Nobs_grouped_state[i]
        accumulating_Nobs[i] = tot
    end
    accumulating_Nobs = Int.(accumulating_Nobs)

    Index_vector = []
    for i in 1:Number_of_groups
        if i ==1
            sample_group = sample(1:accumulating_Nobs[i], Nobs_group_testdata[i], replace=false)
        elseif i > 1 
            sample_group = sample(accumulating_Nobs[i-1]+1:accumulating_Nobs[i], Nobs_group_testdata[i], replace=false)
        end
        push!(Index_vector, sample_group)
    end
    Index_vector = reduce(vcat, Index_vector)

    testdata = df[Index_vector, :]
    traindata = df[Not(Index_vector), :]

    return testdata, traindata
end


function multi_sigmoid(input1, input2, input3, input4, input5, input6)
    λ1 = exp(input1) / (1 +  exp(input1) + exp(input2)+  exp(input3) + exp(input4)+  exp(input5) + exp(input6))
    λ2 = exp(input2) / (1 +  exp(input1) + exp(input2)+  exp(input3) + exp(input4)+  exp(input5) + exp(input6))
    λ3 = exp(input3) / (1 +  exp(input1) + exp(input2)+  exp(input3) + exp(input4)+  exp(input5) + exp(input6))
    λ4 = exp(input4) / (1 +  exp(input1) + exp(input2)+  exp(input3) + exp(input4)+  exp(input5) + exp(input6))
    λ5 = exp(input5) / (1 +  exp(input1) + exp(input2)+  exp(input3) + exp(input4)+  exp(input5) + exp(input6))
    λ6 = exp(input6) / (1 +  exp(input1) + exp(input2)+  exp(input3) + exp(input4)+  exp(input5) + exp(input6))
    return λ1, λ2, λ3, λ4, λ5, λ6
end

function init_ps()
    NN_model = NN_model_two_layer(4, 50, swish)
    ps_λ1, NN_ls = Lux.setup(rng, NN_model)
    ps_λ2, _ = Lux.setup(rng, NN_model)
    ps_λ3, _ = Lux.setup(rng, NN_model)
    ps_λ4, _ = Lux.setup(rng, NN_model)
    ps_λ5, _ = Lux.setup(rng, NN_model)
    ps_λ6, _ = Lux.setup(rng, NN_model)

    NN_parameters = ComponentArray{Float32}()
    NN_parameters = ComponentArray(NN_parameters;ps_λ1)
    NN_parameters = ComponentArray(NN_parameters;ps_λ2)
    NN_parameters = ComponentArray(NN_parameters;ps_λ3)
    NN_parameters = ComponentArray(NN_parameters;ps_λ4)
    NN_parameters = ComponentArray(NN_parameters;ps_λ5)
    NN_parameters = ComponentArray(NN_parameters;ps_λ6)

    loss = loss_fn(NN_parameters, NN_ls, state, COVS, TIME', NN_model)[1]

    println("--------------------------------------------")
    println("Loss is $loss.")
    println("--------------------------------------------")
    ## max try
    counter = 0
    while isinf(loss) && counter < 20
        counter += 1
        NN_model = NN_model_two_layer(4, 50, swish)
        ps_λ1, NN_ls = Lux.setup(rng, NN_model)
        ps_λ2, _ = Lux.setup(rng, NN_model)
        ps_λ3, _ = Lux.setup(rng, NN_model)
        ps_λ4, _ = Lux.setup(rng, NN_model)
        ps_λ5, _ = Lux.setup(rng, NN_model)
        ps_λ6, _ = Lux.setup(rng, NN_model)

        NN_parameters = ComponentArray{Float32}()
        NN_parameters = ComponentArray(NN_parameters;ps_λ1)
        NN_parameters = ComponentArray(NN_parameters;ps_λ2)
        NN_parameters = ComponentArray(NN_parameters;ps_λ3)
        NN_parameters = ComponentArray(NN_parameters;ps_λ4)
        NN_parameters = ComponentArray(NN_parameters;ps_λ5)
        NN_parameters = ComponentArray(NN_parameters;ps_λ6)

        loss = loss_fn(NN_parameters, NN_ls, state, COVS, TIME', NN_model)[1]

        if !isinf(loss)
            #println("--------------------------------------------")
            println("Final loss is $loss.")
            println("--------------------------------------------")
            break
        elseif counter == 20
            break 
        end
        
    end
    return NN_parameters
end

function loss_fn(params, ls, state, X, T, model)# X = covariates, T = time to event
    loss = 0.0f0
    λ1pred = model(X, params.ps_λ1, ls)[1]
    λ2pred = model(X, params.ps_λ2, ls)[1]
    λ3pred = model(X, params.ps_λ3, ls)[1]
    λ4pred = model(X, params.ps_λ4, ls)[1]
    λ5pred = model(X, params.ps_λ5, ls)[1]
    λ6pred = model(X, params.ps_λ6, ls)[1]

    for i in eachindex(T)
        λ1pred_i, λ2pred_i,λ3pred_i, λ4pred_i,λ5pred_i, λ6pred_i = multi_sigmoid((λ1pred[i]), (λ2pred[i]),(λ3pred[i]), (λ4pred[i]),(λ5pred[i]), (λ6pred[i]) )

        survival = (1-(λ1pred_i+λ2pred_i+λ3pred_i+λ4pred_i+λ5pred_i+λ6pred_i) ) + 0.00001
        if λ1pred_i <  0.0000000001
            λ1pred_i = 0.0000000001
        end
        
        if λ2pred_i <  0.0000000001
            λ2pred_i = 0.0000000001
        end

        if λ3pred_i <  0.0000000001
            λ3pred_i = 0.0000000001
        end

        if λ4pred_i <  0.0000000001
            λ4pred_i = 0.0000000001
        end

        if λ5pred_i <  0.0000000001
            λ5pred_i = 0.0000000001
        end

        if λ6pred_i <  0.0000000001
            λ6pred_i = 0.0000000001
        end

        if (state[i] == 0 )
            loss += (T[i] * log(survival) )
        elseif (state[i] == 1 )
            loss += log(λ1pred_i) + (T[i] * log(survival) )
        elseif (state[i] == 2 )
            loss += log(λ2pred_i) + (T[i] * log(survival) )
        elseif (state[i] == 3 )
            loss += log(λ3pred_i) + (T[i] * log(survival) )
        elseif (state[i] == 4 )
            loss += log(λ4pred_i) + (T[i] * log(survival) )
        elseif (state[i] == 5 )
            loss += log(λ5pred_i) + (T[i] * log(survival) )
        elseif (state[i] == 6 )
            loss += log(λ6pred_i) + (T[i] * log(survival) )
        end
    end

    return -1*loss, 1
end

## training with mini-batching
function training_1(epochs, model, state, COVS, TIME, ps, ls, train_loader, opt_state
    )
    lossvec = []
    push!(lossvec,loss_fn(ps, ls, state, COVS, TIME', model)[1] )

    t = time()
    for epoch in 1:epochs
        losscount = 0.0
        for (x, y) in train_loader
            (loss, _), back = pullback(loss_fn, ps, ls, state, x, y, model)# ## updated to train_loader (x=COVS, y=TIME)
            grad, _ = back((one(loss), nothing))

            opt_state, ps = Optimisers.update(opt_state, ps, grad)

            losscount=loss
        end
        
        if epoch % 10 == 0
            losscount = loss_fn(ps, ls, state, COVS, TIME', model)[1]
            dt = (round((time() - t), digits = 2) ) /60
            println("Epoch: $epoch, Loss: $losscount")
            println("Elapsed time is $dt min")
            println("-------------------------------------")
        end
        if epoch % 1 == 0
            push!(lossvec, loss_fn(ps, ls, state, COVS, TIME', model)[1] )
        end

    end
    #dt = round((time() - t), digits = 2) 
    return ps, lossvec     
end

## training without mini-batching
function training_2(epochs, model, state, COVS, TIME, ps, ls, train_loader, opt_state
    )
    lossvec = []
    push!(lossvec,loss_fn(ps, ls, state, COVS, TIME', model)[1] )

    t = time()
    for epoch in 1:epochs
        losscount = 0.0
        #for (x, y) in train_loader
            (loss, _), back = pullback(loss_fn, ps, ls, state, COVS, TIME', model)# ## updated to train_loader (x=COVS, y=TIME)
            grad, _ = back((one(loss), nothing))

            opt_state, ps = Optimisers.update(opt_state, ps, grad)

            losscount=loss
        #end
        
        if epoch % 10 == 0
            losscount = loss_fn(ps, ls, state, COVS, TIME', model)[1]
            dt = (round((time() - t), digits = 2) ) / 60
            println("Epoch: $epoch, Loss: $losscount")
            println("Elapsed time is $dt min")
            println("-------------------------------------")
        end
        if epoch % 1 == 0
            push!(lossvec, loss_fn(ps, ls, state, COVS, TIME', model)[1] )
        end

    end
    #dt = round((time() - t), digits = 2) 
    return ps, lossvec     
end

function predict(model, params, inputs, layerstates)
    pred1 = (vec(model(inputs, params.ps_λ1, layerstates)[1]) )
    pred2 = (vec(model(inputs, params.ps_λ2, layerstates)[1]) )
    pred3 = (vec(model(inputs, params.ps_λ3, layerstates)[1]) )
    pred4 = (vec(model(inputs, params.ps_λ4, layerstates)[1]) )
    pred5 = (vec(model(inputs, params.ps_λ5, layerstates)[1]) )
    pred6 = (vec(model(inputs, params.ps_λ6, layerstates)[1]) )

    Nid = length(pred1)
    tp1, tp2, tp3, tp4, tp5, tp6 = zeros(Nid), zeros(Nid), zeros(Nid), zeros(Nid), zeros(Nid), zeros(Nid)

    for i in eachindex(pred1)
        tp1[i], tp2[i], tp3[i], tp4[i], tp5[i], tp6[i] = multi_sigmoid(pred1[i], pred2[i], pred3[i], pred4[i], pred5[i], pred6[i])
    end
    tp0 = 1 .- (tp1 .+ tp2.+ tp3.+ tp4.+ tp5.+ tp6)
    return tp0, tp1, tp2, tp3, tp4, tp5, tp6
end

## Neural network model
function NN_model_two_layer(N_covs, Nnodes, af)
    m = Chain( Dense(N_covs, Nnodes, af),
        Dense(Nnodes, Nnodes, af),
        Dense(Nnodes, 1) )
    return m
end

function state_occupation_matrix(df::DataFrame, Time_point_evaluated::Int)
    
    filtered_df = filter(r -> !(r.time < Time_point_evaluated && r.state == 0), df)
    TIME = Int.(round.(filtered_df.time))
    state = filtered_df.state
    
    Nid = length(TIME)
    state_occupation_mat = zeros(Nid, 7)

    for i in 1:Nid
        state_occupation = zeros(7)
        if (TIME[i] > Time_point_evaluated) 
            state_occupation[1] = 1
        elseif (TIME[i] <= Time_point_evaluated)
            current_state = state[i] + 1 ## since initial state is not accounted for in data
            state_occupation[current_state] = 1
        end
        state_occupation_mat[i, :] = state_occupation
    end

    return state_occupation_mat, filtered_df.id
end

function brier_score_real_world(pred::Matrix, observations::Matrix, Time_point_evaluated::Int)
    Nid = size(pred)[1]
    BS = zeros(Nid)
    π0 = [1, 0, 0, 0, 0, 0, 0]'
    for i in 1:Nid
        TP = pred[i,:]
        P = [TP[1] TP[2] TP[3] TP[4] TP[5] TP[6] TP[7];
            0 1 0 0 0 0 0 ;
            0 0 1 0 0 0 0 ;
            0 0 0 1 0 0 0 ;
            0 0 0 0 1 0 0 ;
            0 0 0 0 0 1 0 ;
            0 0 0 0 0 0 1 ;]

        state_occupation_pred = vec(π0 * (P^Time_point_evaluated) )
        BS[i] = sum( (state_occupation_pred .- observations[i,:]) .^2  )
    end
    return BS
end

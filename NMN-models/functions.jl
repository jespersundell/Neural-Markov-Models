###########################################################
####### Functions for data generation and manipulation ####
###########################################################
function transform_transvec(transitionvector, statevector, endstate)
    transformed_transitionvector = deepcopy(transitionvector)
    for i in eachindex(statevector)
        if statevector[i] != endstate
            transformed_transitionvector[i] = 0
        end
    end 
    return transformed_transitionvector
end

### function to make interval censored time vector
function introduce_interval_censoring(timevector, transitionvector12, transitionvector13, P)
    interval_vector = []
    ϵ_distribution = Poisson(P)
    for i in eachindex(timevector)
        if (transitionvector12[i] == 1) && (timevector[i] >= 0)
            lower_interval = timevector[i] - rand(ϵ_distribution)
            if lower_interval < 0
                lower_interval = 0
            end
            upper_interval = timevector[i] + rand(ϵ_distribution)
            interval = (lower_interval, upper_interval)
            push!(interval_vector, interval)
        
        elseif (transitionvector13[i] == 1) && (timevector[i] >= 0)
            lower_interval = timevector[i] - rand(ϵ_distribution)
            if lower_interval < 0
                lower_interval = 0
            end
            upper_interval = timevector[i] + rand(ϵ_distribution)
            interval = (lower_interval, upper_interval)
            push!(interval_vector, interval)
        elseif (transitionvector12[i] == 0) && (transitionvector13[i] == 0)
            interval = (timevector[i], timevector[i])
            push!(interval_vector, interval)

        end
    end
    return interval_vector
end

## replace time vector of tuple with nx2 matrix
function create_time_matrix(timevector)
    timematrix = zeros(length(timevector), 2)
    for i in eachindex(timevector)
        timematrix[i,1] = timevector[i][1]
        timematrix[i,2] = timevector[i][2]
    end
    timematrix = Int.(timematrix)
    return timematrix
end

function simulate_data(Nid, Ncovs)
    time_vector = Int[]
    trans12 = zeros(Nid)
    trans13 = zeros(Nid)
    state_vec = Int[]

    cov_mat = zeros(Nid, Ncovs)
    lambda_vec12 = zeros(Nid)
    lambda_vec13 = zeros(Nid)

    h(X) = ( ( (0.5*(X[1]^3))/(X[1]^3+0.3^3 ) ) + ( 5*X[2]*(X[2]-0.9)*(X[2]-0.8) +0.05  ))   /4
    hz(X) = ( exp(X[1]*2+ X[1]^3-4) + ( 5*X[2]*(X[2]-0.9)*(X[2]-0.8) + 0.15  ) ) /3

    for i in 1:Nid
        day = 0
        ϕi = rand(0.02:0.01:0.98,Ncovs)

        cov_mat[i,:] = ϕi

        ϵ_λ12i = rand(Normal(0, 0.05))
        ϵ_λ13i = rand(Normal(0, 0.05))
        λ12i = h(ϕi) + (h(ϕi) * ϵ_λ12i)
        λ13i = hz(ϕi) + (hz(ϕi) * ϵ_λ13i) 

        lambda_vec12[i] = λ12i
        lambda_vec13[i] = λ13i
        ##########################################################
        ##########################################################
        λ11i = 1-λ12i-λ13i
        ## P = (λ11, λ12, λ13)
        d_event = Categorical(λ11i, λ12i, λ13i)

        event = rand(d_event)
        if event == 2
            state = 2
            trans12[i] = 1
        elseif event == 3
            state = 3
            trans13[i] = 1
        end

        while (event == 1)
            day += 1
            d_event = Categorical(λ11i, λ12i, λ13i)
            event = rand(d_event)
            if event == 3
                trans13[i] = 1
                state = 3
                break
            elseif event == 2
                trans12[i] = 1
                state = 2
                break
            end
            
        end

        push!(time_vector, day)
        push!(state_vec, state)
    end
    return time_vector, trans12, trans13, state_vec, cov_mat', lambda_vec12, lambda_vec13
end

function right_censoring(TIME, trans12, trans13, do_prob)
    d = Bernoulli(do_prob)
    d_rigthcensoring = Poisson(1)
    Nid = length(trans12)

    t12 = deepcopy(trans12)
    t13 = deepcopy(trans13)
    T = deepcopy(TIME)
    for i in 1:Nid
        dropout = rand(d)
        if dropout == true
            t12[i] = 0.0
            t13[i] = 0.0

            remove_time = rand(d_rigthcensoring)

            T[i,:] = T[i,:] .- [remove_time, remove_time]
            
            Tmax = T[i, :][2]
            if Tmax < 0
                Tmax = 0
            end
            T[i,:] .= Tmax
        end
    end

    return T, t12, t13

end
###########################################################
####### Functions ########################################
#########################################################
function loss_fn(params, ls, transition12, transition13, X, T, model)# X = covariates, T = time to event
    loss = 0.0f0
    lower = T[1,:]
    upper = T[2,:]
    λ12pred = model(X, params.ps_λ12, ls)[1]
    λ13pred = model(X, params.ps_λ13, ls)[1]

    for i in eachindex(lower)
        λ12pred_i, λ13pred_i = multi_sigmoid((λ12pred[i]), (λ13pred[i]) )
        survival = (1-(λ12pred_i+λ13pred_i) ) + 0.000000001

        if (transition12[i] == 1 ) && (lower[i] == upper[i]) ## we know the exact time of event 2   
            loss += log(λ12pred_i) + (lower[i] * log(survival) )
        elseif (transition13[i] == 1 ) && (lower[i] == upper[i]) ## we know the exact time of event 3
            
            loss += log(λ13pred_i) + (lower[i] * log(survival) )

        elseif (transition12[i] == 1) && (lower[i] != upper[i] ) ## interval censored λ12
            interval_prob = eps()

            for jj in (lower[i]):upper[i]
                interval_prob += (survival^jj) * λ12pred_i
            end
            loss += log(interval_prob)   

        elseif (transition13[i] == 1) && (lower[i] != upper[i] ) ## interval censored λ13
            interval_prob = eps()

            for jj in (lower[i]):upper[i]
                interval_prob += (survival^jj) * λ13pred_i
            end
            loss += log(interval_prob)
        elseif (transition13[i] == 0) && (transition12[i] == 0) ## right censored

            loss += (upper[i] * log(survival) )
        end
    end

    return -1*loss, 1
end

function training(epochs, model, transition12, transition13, COVS, TIME, parameters, layerstates, train_loader, opt_state
    )
    lossvec = []
    push!(lossvec,loss_fn(parameters, layerstates, transition12, transition13, COVS, TIME', model)[1] )

    t = time()
    for epoch in 1:epochs
        losscount = 0.0
        for (x, y) in train_loader
            (loss, _), back = pullback(loss_fn, parameters, layerstates, transition12, transition13, x, y, model)# ## updated to train_loader (x=COVS, y=TIME)
            grad, _ = back((one(loss), nothing))

            opt_state, parameters = Optimisers.update(opt_state, parameters, grad)
         ## reseting parameters if loss is NaN
            losscount=loss
        end
        
        # if isnan(losscount)
        #     parameters = reinitialize_params(parameters, layerstates)
        #     println("Loss is NaN")
        # elseif isinf(losscount)
        #     parameters = reinitialize_params(parameters, layerstates)
        # end
        if epoch % 20 == 0
            losscount = loss_fn(parameters, layerstates, transition12, transition13, COVS, TIME', model)[1]
            dt = round((time() - t), digits = 2)
            println("Epoch: $epoch, Loss: $losscount")
            println("Elapsed time is $dt")
            println("-------------------------------------")
        end
        if epoch % 1 == 0
            push!(lossvec, loss_fn(parameters, layerstates, transition12, transition13,COVS, TIME', model)[1] )
        end

    end
    dt = round((time() - t), digits = 2) 
    return parameters, lossvec, dt     
end

function predict(model, params, inputs, layerstates)
    pred12 = (vec(model(inputs, params.ps_λ12, layerstates)[1]) )
    pred13 = (vec(model(inputs, params.ps_λ13, layerstates)[1]) )

    tp12 = zeros(length(pred12))
    tp13 = zeros(length(pred13))
    for i in eachindex(pred12)
        tp12[i], tp13[i] = multi_sigmoid(pred12[i], pred13[i])
    end
    tp11 = 1 .- (tp12 .+ tp13)
    return tp11, tp12, tp13
end

function multi_sigmoid(input1, input2)
    λ1 = exp(input1) / (1 +  exp(input1) + exp(input2))
    λ2 = exp(input2) / (1 +  exp(input1) + exp(input2))
    return λ1, λ2
end

## Neural network models
function NN_model_two_layer(N_covs, Nnodes, af)
    m = Chain( Dense(N_covs, Nnodes, af),
        Dense(Nnodes, Nnodes, af),
        Dense(Nnodes, 1) )
    return m
end

function NN_model_three_layer(N_covs, Nnodes, af)
    m = Chain( Dense(N_covs, Nnodes, af),
        Dense(Nnodes, Nnodes, af),
        Dense(Nnodes, Nnodes, af),
        Dense(Nnodes, 1) )
    return m
end
###################################################################
##### Brier Score functions #######################################
###################################################################

function state_occupation_probability(T, λ12, λ13)
    π0 = [1, 0, 0]

    λ11 = 1 -(λ12+λ13)
    P = [λ11 λ12 λ13;
        0.0 1.0 0.0;
        0.0 0.0 1.0]
    P_t = P^T
    π_t = vec(π0' * P_t )
    return π_t
end


#state_occupation_probability(10, 0.1, 0.03)

function Brier_score_interval(T_interval::Vector, λ12, λ13, observed_state_occupation::Vector)
    interval_lenght = T_interval[2] - T_interval[1]
    BS = 0.0
    for i in T_interval[1]:T_interval[2]
        π_t = state_occupation_probability(i, λ12, λ13)
        BS += sum( (π_t .- observed_state_occupation).^2 )
    end
    return BS .* (1/interval_lenght)
end

#a = Brier_score_interval([3, 5], 0.1, 0.03, [0,1,0])


function Brier_score(T::Matrix, λ12_pred::Vector, λ13_pred::Vector, transition12, transition13, time_cutoff::Int)
    N_obs = length(λ12_pred)
    BS = zeros(N_obs)

    for i in 1:N_obs
        if (transition12[i] == 0.0) && (transition13[i] == 0.0) && T[i,:][1] < time_cutoff ## do not count IDs which dropout prior to time cutoff
            BS[i] = 0.0
        elseif (transition12[i] == 1.0) && T[i,:][2] < time_cutoff ## transition to state 2 did occur prior to time cutoff
            π_t = state_occupation_probability(time_cutoff, λ12_pred[i], λ13_pred[i])
            state_occupation = [0, 1, 0]
            BS[i] = sum( (π_t .- state_occupation).^2 )
        elseif (transition13[i] == 1.0) && T[i,:][2] < time_cutoff ## transition to state 3 did occur prior to time cutoff

            π_t = state_occupation_probability(time_cutoff, λ12_pred[i], λ13_pred[i])
            state_occupation = [0, 0, 1]
            BS[i] = sum( (π_t .- state_occupation).^2 )

        elseif T[i,:][1] > time_cutoff ## transition did not yet occur
            π_t = state_occupation_probability(time_cutoff, λ12_pred[i], λ13_pred[i])
            state_occupation = [1, 0, 0]
            BS[i] = sum( (π_t .- state_occupation).^2 )

        elseif (T[i,:][1] < time_cutoff) && (T[i,:][2] > time_cutoff) && (transition12[i] == 1)## cutoff occurs within interval trans12
            BS[i] = Brier_score_interval(T[i,:], λ12_pred[i], λ13_pred[i], [0,1,0])
    
        elseif (T[i,:][1] < time_cutoff) && (T[i,:][2] > time_cutoff) && (transition13[i] == 1)## cutoff occurs within interval trans12
            BS[i] = Brier_score_interval(T[i,:], λ12_pred[i], λ13_pred[i], [0,0,1])
        end
    end

    BS_event = filter(x -> x !=0.0, BS)

    return BS_event
end


function Brier_score_multiple_timepoints(T::Matrix, λ12_pred::Vector, λ13_pred::Vector, transition12, transition13, time_cutoff::Vector)
    
    mean_BS = []
    mode_BS = []
    for i in eachindex(time_cutoff)
        BS = Brier_score(T, λ12_pred, λ13_pred, transition12, transition13, time_cutoff[i])
        push!(mean_BS, mean(BS))
        push!(mode_BS, mode(BS))
    end

    return mean_BS, mode_BS
end


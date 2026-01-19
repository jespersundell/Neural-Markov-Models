using Lux, Zygote, Distributions, Optimisers, Random, MLUtils
using ComponentArrays
using Statistics
using DataFrames, CSV

include("functions_real_data.jl")


# df = CSV.read("e2.csv", DataFrame)
# rename!(df, :status => :state)
# df_validation, df_train = train_test_data_split(df, 0.2)

df_train = CSV.read("training_data.csv", DataFrame)
df_validation = CSV.read("validation_data.csv", DataFrame)

#names(df)
#df = filter(row -> row.status <=2, df)
Nid = size(df_train)[1]
Nid_val = size(df_validation)[1]
TIME = Int.(round.(df_train.time))
rng = Random.default_rng()
state = df_train.state

x1 = df_train.x1'
x2 = df_train.x2'
x3 = df_train.x3'
x4 = df_train.x4'

x1_val = df_validation.x1'
x2_val = df_validation.x2'
x3_val = df_validation.x3'
x4_val = df_validation.x4'

COVS_val = vcat(x1_val, x2_val, x3_val, x4_val)
TIME_val = Int.(round.(df_validation.time))
state_val = df_validation.state

##################################################################################################################
#################################################################################################################
#################################################################################################################
############## CONSTANT MODEL #####################################################################################
#################################################################################################################
COVS = ones(1, Nid)
train_loader = DataLoader((data=COVS, label=TIME'), batchsize=264, shuffle=true)
const_model = Dense(1, 1)
ps_λ1, const_ls = Lux.setup(rng, const_model)
ps_λ2, _ = Lux.setup(rng, const_model)
ps_λ3, _ = Lux.setup(rng, const_model)
ps_λ4, _ = Lux.setup(rng, const_model)
ps_λ5, _ = Lux.setup(rng, const_model)
ps_λ6, _ = Lux.setup(rng, const_model)

const_parameters = ComponentArray{Float32}()
const_parameters = ComponentArray(const_parameters;ps_λ1)
const_parameters = ComponentArray(const_parameters;ps_λ2)
const_parameters = ComponentArray(const_parameters;ps_λ3)
const_parameters = ComponentArray(const_parameters;ps_λ4)
const_parameters = ComponentArray(const_parameters;ps_λ5)
const_parameters = ComponentArray(const_parameters;ps_λ6)
opt = Adam(0.01)
opt_state = Optimisers.setup(opt, const_parameters)

loss_fn(const_parameters, const_ls, state, COVS, TIME', const_model)[1]

const_parameters, lossvec = training_1(100, const_model, state, COVS, TIME, const_parameters, const_ls, train_loader, opt_state)
const_parameters, lossvec = training_2(100, const_model, state, COVS, TIME, const_parameters, const_ls, train_loader, opt_state)
const_model_loss =loss_fn(const_parameters, const_ls, state, COVS, TIME', const_model)[1]

##################################################################################################################
#################################################################################################################
#################################################################################################################
############## LINEAR MODEL #####################################################################################
#################################################################################################################
COVS = vcat(x1, x2, x3, x4)
train_loader = DataLoader((data=COVS, label=TIME'), batchsize=264, shuffle=true)

lin_model = Dense(4, 1)
ps_λ1, lin_ls = Lux.setup(rng, lin_model)
ps_λ2, _ = Lux.setup(rng, lin_model)
ps_λ3, _ = Lux.setup(rng, lin_model)
ps_λ4, _ = Lux.setup(rng, lin_model)
ps_λ5, _ = Lux.setup(rng, lin_model)
ps_λ6, _ = Lux.setup(rng, lin_model)

ps_λ1.weight[1] = 0
ps_λ2.weight[1] = 0
ps_λ3.weight[1] = 0
ps_λ4.weight[1] = 0
ps_λ5.weight[1] = 0
ps_λ6.weight[1] = 0

lin_parameters = ComponentArray{Float32}()
lin_parameters = ComponentArray(lin_parameters;ps_λ1)
lin_parameters = ComponentArray(lin_parameters;ps_λ2)
lin_parameters = ComponentArray(lin_parameters;ps_λ3)
lin_parameters = ComponentArray(lin_parameters;ps_λ4)
lin_parameters = ComponentArray(lin_parameters;ps_λ5)
lin_parameters = ComponentArray(lin_parameters;ps_λ6)

opt = Adam(0.01)
opt_state = Optimisers.setup(opt, lin_parameters)

### training ##########################################
loss_fn(lin_parameters, lin_ls, state, COVS, TIME', lin_model)[1]
lin_parameters, lossvec = training_1(200, lin_model, state, COVS, TIME, lin_parameters, lin_ls, train_loader, opt_state)
lin_parameters, lossvec = training_2(200, lin_model, state, COVS, TIME, lin_parameters, lin_ls, train_loader, opt_state)
#plot(lossvec)
lin_model_loss =loss_fn(lin_parameters, lin_ls, state, COVS, TIME', lin_model)[1]


##################################################################################################################
#################################################################################################################
#################################################################################################################
############## ANN MODEL #####################################################################################
#################################################################################################################
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
opt = Adam(0.001)
opt_state = Optimisers.setup(opt, NN_parameters)

## Initialization
loss_fn(NN_parameters, NN_ls, state, COVS, TIME', NN_model)[1]
NN_parameters = init_ps()
loss_fn(NN_parameters, NN_ls, state, COVS, TIME', NN_model)[1]

### training ##########################################
NN_parameters, lossvec = training_1(200, NN_model, state, COVS, TIME, NN_parameters, NN_ls, train_loader, opt_state)

opt = Adam(0.0002)
opt_state = Optimisers.setup(opt, NN_parameters)
NN_parameters, lossvec = training_2(200, NN_model, state, COVS, TIME, NN_parameters, NN_ls, train_loader, opt_state)
#plot(lossvec)
NN_model_loss =loss_fn(NN_parameters, NN_ls, state, COVS, TIME', NN_model)[1]

COVS_const_val = ones(1, Nid_val)
const_loss_val =loss_fn(const_parameters, const_ls, state_val, COVS_const_val, TIME_val', const_model)[1]
lin_loss_val =loss_fn(lin_parameters, lin_ls, state_val, COVS_val, TIME_val', lin_model)[1]
NN_loss_val =loss_fn(NN_parameters, NN_ls, state_val, COVS_val, TIME_val', NN_model)[1]

##################################################################################################################
#################################################################################################################
#################################################################################################################
############## Predictions validation data ######################################################################
#################################################################################################################
NN_pred_0, NN_pred_1, NN_pred_2, NN_pred_3, NN_pred_4, NN_pred_5, NN_pred_6 = predict(NN_model, NN_parameters, COVS_val, NN_ls)
lin_pred_0, lin_pred_1, lin_pred_2, lin_pred_3, lin_pred_4, lin_pred_5, lin_pred_6 = predict(lin_model, lin_parameters, COVS_val, lin_ls)
const_pred_0, const_pred_1, const_pred_2, const_pred_3, const_pred_4, const_pred_5, const_pred_6 = predict(const_model, const_parameters, COVS_const_val, const_ls)

# df_pred_val = DataFrame(NN0 = NN_pred_0,
                    # NN1 = NN_pred_1,
                    # NN2 = NN_pred_2,
                    # NN3 = NN_pred_3,
                    # NN4 = NN_pred_4,
                    # NN5 = NN_pred_5,
                    # NN6 = NN_pred_6,
                    # lin0 = lin_pred_0,
                    # lin1 = lin_pred_1,
                    # lin2 = lin_pred_2,
                    # lin3 = lin_pred_3,
                    # lin4 = lin_pred_4,
                    # lin5 = lin_pred_5,
                    # lin6 = lin_pred_6,
                    # const0 = const_pred_0,
                    # const1 = const_pred_1,
                    # const2 = const_pred_2,
                    # const3 = const_pred_3,
                    # const4 = const_pred_4,
                    # const5 = const_pred_5,
                    # const6 = const_pred_6)

#CSV.write("prediction_validation.csv", df_pred_val)

NN_pred = hcat(NN_pred_0, NN_pred_1, NN_pred_2, NN_pred_3, NN_pred_4, NN_pred_5, NN_pred_6)
lin_pred = hcat(lin_pred_0, lin_pred_1, lin_pred_2, lin_pred_3, lin_pred_4, lin_pred_5, lin_pred_6)
const_pred = hcat(const_pred_0, const_pred_1, const_pred_2, const_pred_3, const_pred_4, const_pred_5, const_pred_6)

state_occupation_mat, ID_vec = state_occupation_matrix(df_validation, 60)
ID_vec
mask_val = in.(df_validation.id, Ref(ID_vec))
const_pred = const_pred[mask_val, :]
lin_pred = lin_pred[mask_val, :]
NN_pred = NN_pred[mask_val, :]

BS_const = brier_score_real_world(const_pred, state_occupation_mat, 60)
BS_lin = brier_score_real_world(lin_pred, state_occupation_mat, 60)
BS_NN = brier_score_real_world(NN_pred, state_occupation_mat, 60)


mean(BS_lin .- BS_NN)

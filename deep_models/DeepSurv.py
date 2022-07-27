import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
import json
import itertools

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch  # For building the networks
import torchtuples as tt  # Some useful functions

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong

from Pycox.cancer.pycox_models_utils import init_data, init_random_state, evaluate, create_transformers, \
    process_different_types_of_features, prepare_columns


def prepare_neural_network(x_train, num_nodes, dropout, with_categorical):
    if with_categorical:
        in_features = x_train[0].shape[1]
    else:
        in_features = x_train.shape[1]

    out_features = 1
    batch_norm = True
    output_bias = False

    if with_categorical:
        num_embeddings = x_train[1].max(0) + 1
        embedding_dims = num_embeddings // 2

        net = tt.practical.MixedInputMLP(in_features, num_embeddings, embedding_dims,
                                         num_nodes, out_features, batch_norm, dropout)
    else:
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

    return net


def fit(selected_features: list, params: dict):
    init_random_state(random_state=42)
    df_train, df_val, df_test = init_data(selected_features=selected_features)

    cols_standardize, cols_leave, cols_categorical = prepare_columns(selected_features=selected_features)
    x_mapper_float, x_mapper_long = process_different_types_of_features(cols_standardize=cols_standardize,
                                                                        cols_leave=cols_leave,
                                                                        cols_categorical=cols_categorical)

    with_categorical = len(cols_categorical) > 0

    # Turn them to binary features (0, 1)
    df_train[cols_leave] -= 1
    df_val[cols_leave] -= 1
    df_test[cols_leave] -= 1

    x_fit_transform, x_transform = create_transformers(x_mapper_float=x_mapper_float, x_mapper_long=x_mapper_long)

    x_train = x_fit_transform(df_train)
    x_val = x_transform(df_val)
    x_test = x_transform(df_test)

    get_target = lambda df: (df['Survival'].values, df['Status'].values)

    y_train = get_target(df_train)
    y_val = get_target(df_val)

    val = x_val, y_val

    net = prepare_neural_network(x_train=x_train,
                                 num_nodes=params['num_nodes'], dropout=params['dropout'],
                                 with_categorical=with_categorical)

    model = CoxPH(net, tt.optim.Adam)

    model.optimizer.set_lr(params['lr'])

    epochs = params['epoch']
    callbacks = [tt.cb.EarlyStoppingCycle()]
    verbose = False
    batch_size = params['batch_size']


    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                    val_data=val, val_batch_size=batch_size)

    # _ = log.to_pandas().iloc[1:].plot()

    time_grid = params['time_grid']  # np.arange(365, 1826, 30)  # np.arange(365, 2400, 30)

    _ = model.compute_baseline_hazards()

    surv_cdi_test = model.predict_surv_df(x_test)
    surv_cdi_train = model.predict_surv_df(x_train)
    surv_cdi_val = model.predict_surv_df(x_val)

    durations_test, events_test = get_target(df_test)
    eval_test = evaluate(surv=surv_cdi_test, time_grid=time_grid, durations=durations_test, events=events_test)

    durations_train, events_train = get_target(df_train)
    eval_train = evaluate(surv=surv_cdi_train, time_grid=time_grid, durations=durations_train, events=events_train)

    durations_val, events_val = get_target(df_val)
    eval_val = evaluate(surv=surv_cdi_val, time_grid=time_grid, durations=durations_val, events=events_val)

    return {'Train': eval_train, 'Val': eval_val, 'Test': eval_test}

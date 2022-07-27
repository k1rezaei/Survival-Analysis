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

from pycox.evaluation import EvalSurv
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong


class Settings:
    def __init__(self, features: list, method: str, feature_selection_class: str, params: dict):
        self.features = features
        self.params = params
        self.method = method
        self.feature_selection_class = feature_selection_class
        self.ev = {}

    def get_dict(self):
        # self.params.pop('time_grid', None)
        return {'Method': self.method,
                'Feature Selection Method': self.feature_selection_class,
                'Selected Features': self.features,
                'Model Params': self.params,
                'Evaluation': self.ev}


def init_random_state(random_state: int):
    np.random.seed(random_state)
    _ = torch.manual_seed(random_state)


def init_data(selected_features: list):
    path_to_file = './../dataset_and_other_models/dataset/'

    # Train
    df_train_x = pd.read_csv(path_to_file + 'train_x.csv')
    df_train_x = df_train_x[selected_features]
    df_train_y = pd.read_csv(path_to_file + 'train_y.csv')
    df_train_y['Status'].replace({True: 1, False: 0}, inplace=True)

    # Test
    df_test_x = pd.read_csv(path_to_file + 'test_x.csv')
    df_test_x = df_test_x[selected_features]
    df_test_y = pd.read_csv(path_to_file + 'test_y.csv')
    df_test_y['Status'].replace({True: 1, False: 0}, inplace=True)

    df_train = pd.concat([df_train_x, df_train_y], axis=1)
    df_train = df_train.astype('float32')

    df_test = pd.concat([df_test_x, df_test_y], axis=1)
    df_test = df_test.astype('float32')

    df_val = df_train.sample(frac=0.25)
    df_train = df_train.drop(df_val.index)

    return df_train, df_val, df_test


def intersect(l1: list, l2: list):
    l = []
    for x in l1:
        if x in l2:
            l.append(x)
    return l


def prepare_columns(selected_features: list):
    cols_standardize = intersect(['BE', 'BC', 'AY', 'Age'], selected_features)  # Numerical columns
    cols_leave = intersect(['Y', 'AF', 'BG', 'BH', 'BI', 'BL',
                            'BM', 'BP', 'CY', 'DC',
                            'DW', 'DX', 'DY', 'metastasis', 'secondprim'], selected_features)  # Binary columns
    cols_categorical = intersect(['AQ', 'DE'], selected_features)  # Categorical Columns
    return cols_standardize, cols_leave, cols_categorical


def process_different_types_of_features(cols_standardize: list, cols_leave: list, cols_categorical: list):
    # Preprocess different types of features.
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    categorical = [(col, OrderedCategoricalLong(min_per_category=5)) for col in cols_categorical]

    x_mapper_float = DataFrameMapper(standardize + leave)
    x_mapper_long = None

    if len(categorical) > 0:
        x_mapper_long = DataFrameMapper(categorical)  # we need a separate mapper to ensure the data type 'int64'

    return x_mapper_float, x_mapper_long


def create_transformers(x_mapper_float, x_mapper_long):
    if x_mapper_long is not None:
        x_fit_transform = lambda df: tt.tuplefy(x_mapper_float.fit_transform(df), x_mapper_long.fit_transform(df))
        x_transform = lambda df: tt.tuplefy(x_mapper_float.transform(df), x_mapper_long.transform(df))
        return x_fit_transform, x_transform
    else:
        x_fit_transform = lambda df: x_mapper_float.fit_transform(df)
        x_transform = lambda df: x_mapper_float.transform(df)
        return x_fit_transform, x_transform


def prepare_neural_network(x_train, labtrans, num_nodes, dropout, with_categorical):


    if with_categorical:
        in_features = x_train[0].shape[1]
    else:
        in_features = x_train.shape[1]

    out_features = labtrans.out_features
    batch_norm = True

    if with_categorical:
        num_embeddings = x_train[1].max(0) + 1
        embedding_dims = num_embeddings // 2

        net = tt.practical.MixedInputMLP(in_features, num_embeddings, embedding_dims,
                                         num_nodes, out_features, batch_norm, dropout)
    else:
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

    return net


def evaluate(surv, time_grid, durations, events):
    ev = EvalSurv(surv, durations, events, censor_surv='km')
    eval_result = {'c-index': ev.concordance_td(),
                   'ibs': ev.integrated_brier_score(time_grid)}
    return eval_result



def feature_selection(all_features: list, feature_selection_class: str, model_params: dict,
                      feature_selection_class_params: dict):
    all_settings = []

    if feature_selection_class == 'exhaustive_search':
        k = feature_selection_class_params['k']
        n = len(all_features)

        i = 0
        kol = 1
        for c in range(k):
            kol *= (n - c)
        for c in range(k):
            kol /= (c + 1)
        print(kol)

        for subset in itertools.combinations(all_features, n-k):
            settings = Settings(features=list(subset), method=model_params['name'],
                                feature_selection_class=feature_selection_class,
                                params=model_params)
            ev = fit(list(subset), model_params)
            settings.ev = ev

            all_settings.append(settings.get_dict())
            print(settings.get_dict())
            i += 1
            print(f'-------------  {int(100 * i / kol)}%')

    elif feature_selection_class == 'priority_search':
        l = feature_selection_class_params['min_features']
        r = feature_selection_class_params['max_features']
        ordered_features = feature_selection_class_params['ordered_features']

        for i in range(l, r + 1):
            subset = ordered_features[:i]
            settings = Settings(features=subset, method=model_params['name'],
                                feature_selection_class=feature_selection_class,
                                params=model_params)
            ev = fit(subset, model_params)
            settings.ev = ev

            all_settings.append(settings.get_dict())
            print(settings.get_dict())
            print(f'-------------  {int(100 * (i - l + 1) / (r - l + 1))}%')

    return all_settings


def log(result):
    print(result)
    print('---------------')


def save_to_file(runs, filename):
    with open(filename, "a") as file_object:
        for run in runs:
            run['Model Params']['time_grid'] = []
            file_object.write(json.dumps(run, indent=4))
            file_object.write(',\n')
    file_object.close()


def my_test_model_priority_based(min_features: int, max_features: int, ordered_features: list, model_params: dict,
                                 filename: str):
    feature_selection_class_params = {'min_features': min_features, 'max_features': max_features,
                                      'ordered_features': ordered_features}

    all_features = ['Y', 'AF', 'AQ', 'AY', 'BC', 'BE', 'BG', 'BH', 'BI', 'BL', 'BM', 'BP',
                    'CY', 'DC', 'DE', 'DW', 'DX', 'DY', 'Age', 'metastasis',
                    'secondprim']

    all_runs = feature_selection(all_features=all_features, feature_selection_class='priority_search',
                                 model_params=model_params,
                                 feature_selection_class_params=feature_selection_class_params)
    print('Done ...')
    save_to_file(all_runs, filename)


def my_test_model_exhaustive_search(k: int, model_params: dict, filename: str):
    feature_selection_class_params = {'k': k}

    all_features = ['Y', 'AF', 'AQ', 'AY', 'BC', 'BE', 'BG', 'BH', 'BI', 'BL', 'BM', 'BP',
                    'CY', 'DC', 'DE', 'DW', 'DX', 'DY', 'Age', 'metastasis',
                    'secondprim']

    all_runs = feature_selection(all_features=all_features, feature_selection_class='exhaustive_search',
                                 model_params=model_params,
                                 feature_selection_class_params=feature_selection_class_params)
    print('Done ...')
    save_to_file(all_runs, filename)


if __name__ == '__main__':
    print("Utils ...")

    # params = {'epoch': 512, 'batch_size': 128, 'time_grid': np.arange(365, 1826, 30),
    #           'lr': 0.05, 'num_nodes': [64, 64, 64, 64], 'dropout': 0.7, 'name': 'Deep Surv'}
    # test_model_priority_based(min_features=4, max_features=22,
    #                           ordered_features=['Age', 'BH', 'BC', 'DC', 'BE', 'BG', 'DE',
    #                                             'DW', 'AQ', 'BI', 'AY', 'CY', 'AF', 'BM', 'Y',
    #                                             'metastasis', 'DX', 'DY', 'BP', 'BL', 'secondprim'],
    #                           model_params=params, filename='runs/deep_surv.txt')
    #
    # # test_model_exhaustive_search(k=2, model_params=params, filename='runs/logistic_hazard.txt')
    # #
    # test_model_priority_based(min_features=4, max_features=22,
    #                           ordered_features=['BG', 'Age', 'BI', 'BC', 'BH', 'DE', 'CY', 'DC',
    #                                             'AQ', 'BE', 'BP', 'AY', 'DW', 'BM', 'secondprim',
    #                                             'DX', 'BL', 'metastasis', 'Y', 'DY', 'AF'],
    #                           model_params=params, filename='runs/logistic_hazard.txt')

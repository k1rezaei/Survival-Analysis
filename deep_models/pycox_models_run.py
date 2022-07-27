import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
import random
import json
import itertools

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch  # For building the networks
import torchtuples as tt  # Some useful functions


from pycox_models_utils import Settings
from LogisticHazard import fit as LogisticHazardFit
from DeepSurv import fit as DeepSurvFit


def fit(selected_features: list, params: dict):
    name = params['name']

    if name == 'Logistic Hazard':
        return LogisticHazardFit(selected_features=selected_features, params=params)
    elif name == 'Deep Surv':
        return DeepSurvFit(selected_features=selected_features, params=params)
    else:
        pass


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

        for subset in itertools.combinations(all_features, n - k):
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
    elif feature_selection_class == 'semi_exhaustive_search':
        k = feature_selection_class_params['k']
        e = feature_selection_class_params['experiments']

        n = len(all_features)
        seq = range(n)

        for i in range(e):
            print(i)
            sample = random.sample(seq, k)
            subset = [all_features[sample[i]] for i in range(k)]

            settings = Settings(features=subset, method=model_params['name'],
                                feature_selection_class=feature_selection_class,
                                params=model_params)
            ev = fit(subset, model_params)
            settings.ev = ev

            all_settings.append(settings.get_dict())
            print(settings.get_dict())
            print(f'-------------  {int(100 * (i + 1) / e)}%')

    return all_settings


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
                    'CY', 'DC', 'DE', 'DT', 'DW', 'DX', 'DY', 'Age', 'metastasis',
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


def my_test_model_semi_exhaustive_search(k: int, e:int, model_params: dict, filename: str):
    feature_selection_class_params = {'k': k, 'experiments': e}

    all_features = ['Y', 'AF', 'AQ', 'AY', 'BC', 'BE', 'BG', 'BH', 'BI', 'BL', 'BM', 'BP',
                    'CY', 'DC', 'DE', 'DW', 'DX', 'DY', 'Age', 'metastasis',
                    'secondprim']

    all_runs = feature_selection(all_features=all_features, feature_selection_class='semi_exhaustive_search',
                                 model_params=model_params,
                                 feature_selection_class_params=feature_selection_class_params)
    print('Done ...')
    save_to_file(all_runs, filename)

if __name__ == '__main__':
    params = {'epoch': 512, 'batch_size': 128, 'time_grid': np.arange(365, 1826, 30),
              'lr': 0.05, 'num_nodes': [64, 128, 128, 64], 'dropout': 0.65, 'name': 'Logistic Hazard'}

    # test_model_priority_based(min_features=4, max_features=22,
    #                           ordered_features=['Age', 'BH', 'BC', 'DC', 'BE', 'BG', 'DE', 'DT',
    #                                             'DW', 'AQ', 'BI', 'AY', 'CY', 'AF', 'BM', 'Y',
    #                                             'metastasis', 'DX', 'DY', 'BP', 'BL', 'secondprim'],
    #                           model_params=params, filename='runs/deep_surv.txt')

    # test_model_priority_based(min_features=4, max_features=22,
    #                           ordered_features=['BG', 'Age', 'BI', 'BC', 'BH', 'DE', 'CY', 'DC',
    #                                             'AQ', 'BE', 'BP', 'AY', 'DW', 'BM', 'DT', 'secondprim',
    #                                             'DX', 'BL', 'metastasis', 'Y', 'DY', 'AF'],
    #                           model_params=params, filename='runs/deep_surv.txt')

    # my_test_model_semi_exhaustive_search(k=15, e=1000, model_params=params, filename='runs/log_hazard_new_experiments.txt')
    my_test_model_exhaustive_search(k=2, model_params=params, filename='runs/log_hazard_new_experiments_exhaustive.txt')
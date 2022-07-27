import pandas as pd
import numpy as np


def prepare_data(path_to_x, path_to_y):
    X = pd.read_csv(path_to_x)
    y = pd.read_csv(path_to_y)

    yt = np.array(list(zip(y['Status'], y['Survival'])),
                  dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    Xt = X.to_numpy()

    return Xt, yt


def get_train_test():
    path_to_x_train = 'dataset/train_x.csv'
    path_to_y_train = 'dataset/train_y.csv'

    path_to_x_test = 'dataset/test_x.csv'
    path_to_y_test = 'dataset/test_y.csv'

    X_train, y_train = prepare_data(path_to_x_train, path_to_y_train)
    X_test, y_test = prepare_data(path_to_x_test, path_to_y_test)

    return X_train, y_train, X_test, y_test


def get_event_indicator_time(y_train, y_test):
    event_indicator_test = np.array([False for i in range(len(y_test))], dtype=bool)
    event_time_test = np.zeros(len(y_test), dtype=float)

    event_indicator_train = np.array([False for i in range(len(y_train))], dtype=bool)
    event_time_train = np.zeros(len(y_train), dtype=float)

    for i in range(len(y_test)):
        event_indicator_test[i] = y_test[i][0]
        event_time_test[i] = y_test[i][1]

    for i in range(len(y_train)):
        event_indicator_train[i] = y_train[i][0]
        event_time_train[i] = y_train[i][1]

    return event_indicator_train, event_time_train, event_indicator_test, event_time_test

import pandas as pd
import numpy as np

def prepare_data(path_to_x, path_to_y):
    X = pd.read_csv(path_to_x)
    y = pd.read_csv(path_to_y)

    yt = np.array(list(zip(y['Status'], y['Survival'])),
                 dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    Xt = X.to_numpy()

    return Xt, yt
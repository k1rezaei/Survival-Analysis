import metrics, prepare_data
import numpy as np
from table import draw_plot
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import GridSearchCV

X_train, y_train, X_test, y_test = prepare_data.get_train_test()

random_state = 64

param_grid = {
    'ties': ['efron', 'breslow'],
    'n_iter': [50, 100, 200, 500, 1000],
    'alpha' : [0, 10, 50, 100, 100],
    'tol' : [1e-9, 1e-8]
}

coxph = CoxPHSurvivalAnalysis()

CV_coxph = GridSearchCV(estimator=coxph, param_grid=param_grid, cv=5, verbose=3)
CV_coxph.fit(X_train, y_train)

print(CV_coxph.best_params_)
# {'alpha': 10, 'n_iter': 50, 'ties': 'breslow'}
# {'alpha': 0, 'n_iter': 50, 'ties': 'efron'}
# {'alpha': 10, 'n_iter': 50, 'ties': 'efron'}
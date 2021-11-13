from models import metrics, prepare_data
import numpy as np
from table import draw_plot
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV

X_train, y_train, X_test, y_test = prepare_data.get_train_test()

random_state = 64

param_grid = {
    'l1_ratio': [0.05, 0.3, 0.5, 0.6, 0.7, 0.95],
    'n_alphas' : [50, 100, 200],
}

coxnet = CoxnetSurvivalAnalysis(fit_baseline_model=True, normalize=True)

CV_coxnet = GridSearchCV(estimator=coxnet, param_grid=param_grid, cv=5, verbose=3)
CV_coxnet.fit(X_train, y_train)

print(CV_coxnet.best_params_)
# {'l1_ratio': 0.6, 'n_alphas': 200}
from models import metrics, prepare_data
import numpy as np
from table import draw_plot
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sklearn.model_selection import GridSearchCV

X_train, y_train, X_test, y_test = prepare_data.get_train_test()

random_state = 64

param_grid = {
    'loss': ['ipcwls', 'coxph', 'squared'],
    'learning_rate': [0.05, 0.1],
    'n_estimators' : [100, 500, 1000],
    'dropout_rate' : [0.0, 0.1]
}

cwgbs = ComponentwiseGradientBoostingSurvivalAnalysis(random_state=random_state)

CV_cwgbs = GridSearchCV(estimator=cwgbs, param_grid=param_grid, cv=5, verbose=3)
CV_cwgbs.fit(X_train, y_train)

print(CV_cwgbs.best_params_)
# {'dropout_rate': 0.0, 'learning_rate': 0.1, 'loss': 'ipcwls', 'n_estimators': 1000}
# {'dropout_rate': 0.0, 'learning_rate': 0.1, 'loss': 'coxph', 'n_estimators': 100}
# {'dropout_rate': 0.0, 'learning_rate': 0.1, 'loss': 'coxph', 'n_estimators': 1000}
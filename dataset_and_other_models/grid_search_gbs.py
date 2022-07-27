import metrics, prepare_data
import numpy as np
from table import draw_plot
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.model_selection import GridSearchCV

X_train, y_train, X_test, y_test = prepare_data.get_train_test()

random_state = 64

param_grid = {
    'loss': ['ipcwls', 'coxph', 'squared'],
    'learning_rate': [0.05, 0.1],
    'n_estimators' : [100, 500, 1000],
    'min_samples_leaf' : [1, 4, 8],
    'min_samples_split' : [2, 4, 8]
}

gbs = GradientBoostingSurvivalAnalysis(random_state=random_state)

CV_gbs = GridSearchCV(estimator=gbs, param_grid=param_grid, cv=5, verbose=3)
CV_gbs.fit(X_train, y_train)

print(CV_gbs.best_params_)
print(CV_gbs.best_score_)
print(CV_gbs.best_estimator_
      )
# {'learning_rate': 0.1, 'loss': 'coxph', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 1000}
# {'learning_rate': 0.1, 'loss': 'coxph', 'min_samples_leaf': 8, 'min_samples_split': 2, 'n_estimators': 100}
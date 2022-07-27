import metrics, prepare_data
import numpy as np
from table import draw_plot
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

X_train, y_train, X_test, y_test = prepare_data.get_train_test()

random_state = 64

param_grid = {
    'n_estimators': [50, 100, 500, 1000],
    'max_features': ['sqrt'],
    'min_samples_split' : [3, 6, 10, 15, 20],
    'min_samples_leaf' : [3, 10, 15, 20]
}

rsf = RandomSurvivalForest(random_state=random_state, n_jobs=1)

CV_rfc = GridSearchCV(estimator=rsf, param_grid=param_grid, cv=5, verbose=3)
CV_rfc.fit(X_train, y_train)

print(CV_rfc.best_params_)
print(CV_rfc.best_score_)
print(CV_rfc.best_estimator_)
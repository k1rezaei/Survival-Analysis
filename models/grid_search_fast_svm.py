from sklearn.model_selection import GridSearchCV

from models import metrics, prepare_data
import numpy as np
from table import draw_plot2
from sksurv.svm import FastSurvivalSVM

X_train, y_train, X_test, y_test = prepare_data.get_train_test()

random_state = 64

param_grid = {
    'rank_ratio': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'max_iter' : [20, 40, 100],
    'optimizer': ['avltree', 'direct-count', 'PRSVM', 'rbtree', 'simple']
}

fast_svm = FastSurvivalSVM(random_state=random_state, alpha=1)

CV_fast_svm = GridSearchCV(estimator=fast_svm, param_grid=param_grid, cv=5, verbose=3)
CV_fast_svm.fit(X_train, y_train)

print(CV_fast_svm.best_params_)
# {'max_iter': 20, 'optimizer': 'avltree', 'rank_ratio': 1.0}

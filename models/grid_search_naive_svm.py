from models import metrics, prepare_data
import numpy as np
from table import draw_plot2
from sksurv.svm import NaiveSurvivalSVM
from sklearn.model_selection import GridSearchCV

X_train, y_train, X_test, y_test = prepare_data.get_train_test()

random_state = 64

param_grid = {
    'loss': ['hinge', 'squared_hinge'],
    'penalty' : ['l2'],
    'alpha' : [1.0, 0.1, 10]
}

naivesvm = NaiveSurvivalSVM(random_state=random_state, dual=True, max_iter=10000)

CV_naivesvm = GridSearchCV(estimator=naivesvm, param_grid=param_grid, cv=5, verbose=3)
CV_naivesvm.fit(X_train, y_train)

print(CV_naivesvm.best_params_)
# {'alpha': 0.1, 'loss': 'hinge', 'penalty': 'l2'}
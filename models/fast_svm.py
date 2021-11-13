from models import metrics, prepare_data
import numpy as np
from table import draw_plot2
from sksurv.svm import FastSurvivalSVM

X_train, y_train, X_test, y_test = prepare_data.get_train_test()

random_state = 64

fast_svm = FastSurvivalSVM(alpha=1, rank_ratio=1.0, optimizer='avltree', max_iter=20, random_state=random_state)

fast_svm.fit(X_train, y_train)
print(fast_svm.score(X_train, y_train))
print(fast_svm.score(X_test, y_test))

train_acc_cindex, test_acc_cindex = metrics.c_index_censored(fast_svm, X_train, y_train, X_test, y_test)
train_acc_icpw, test_acc_icpw = metrics.c_index_icpw(fast_svm, X_train, y_train, X_test, y_test)

draw_plot2(['Train', train_acc_cindex, train_acc_icpw],
          ['Test', test_acc_cindex, test_acc_icpw],
          'Fast SVM model')

from models import metrics, prepare_data
import numpy as np
from table import draw_plot2
from sksurv.svm import NaiveSurvivalSVM

X_train, y_train, X_test, y_test = prepare_data.get_train_test()

random_state = 64

naive_svm = NaiveSurvivalSVM(alpha=0.1,
                             penalty='l2',
                             loss='squared_hinge',
                             random_state=random_state,
                             max_iter=10000)

naive_svm.fit(X_train, y_train)
print(naive_svm.score(X_train, y_train))
print(naive_svm.score(X_test, y_test))
train_acc_cindex, test_acc_cindex = metrics.c_index_censored(naive_svm, X_train, y_train, X_test, y_test)
train_acc_icpw, test_acc_icpw = metrics.c_index_icpw(naive_svm, X_train, y_train, X_test, y_test)

draw_plot2(['Train', train_acc_cindex, train_acc_icpw],
          ['Test', test_acc_cindex, test_acc_icpw],
          'Naive SVM model')
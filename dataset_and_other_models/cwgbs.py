from models import metrics, prepare_data
import numpy as np
from table import draw_plot
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis

X_train, y_train, X_test, y_test = prepare_data.get_train_test()

random_state = 64

cw_gbs = ComponentwiseGradientBoostingSurvivalAnalysis(loss='coxph',
                                                       learning_rate=0.1,
                                                       n_estimators=1000,
                                                       dropout_rate=0.0,
                                                       random_state=random_state)

times = np.arange(365, 1826, 30)
cw_gbs.fit(X_train, y_train)

((train_plot, train_acc_auc), (test_plot, test_acc_auc)) = metrics.c_auc_score(cw_gbs, X_train, y_train, X_test, y_test, times)
train_acc_bs, test_acc_bs = metrics.i_brier_score(cw_gbs, X_train, y_train, X_test, y_test, times)
train_acc_icpw, test_acc_icpw = metrics.c_index_icpw(cw_gbs, X_train, y_train, X_test, y_test)
train_acc_cindex, test_acc_cindex = metrics.c_index_censored(cw_gbs, X_train, y_train, X_test, y_test)

draw_plot(['Train', train_acc_cindex, train_acc_icpw, train_acc_auc, train_acc_bs],
          ['Test', test_acc_cindex, test_acc_icpw, test_acc_auc, test_acc_bs],
          'CWGBS model')

with open('files_cancer_gov/cwgbs.npy', 'wb') as f:
    np.save(f, train_plot)
    np.save(f, test_plot)
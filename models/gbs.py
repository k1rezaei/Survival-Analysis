from models import metrics, prepare_data
import numpy as np
from table import draw_plot
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

X_train, y_train, X_test, y_test = prepare_data.get_train_test()

random_state = 64

gbs = GradientBoostingSurvivalAnalysis(loss="coxph",
                                       learning_rate=0.1,
                                       n_estimators=1000,
                                       min_samples_split=4,
                                       min_samples_leaf=1)

times = np.arange(365, 1826, 30)
gbs.fit(X_train, y_train)

((train_plot, train_acc_auc), (test_plot, test_acc_auc)) = metrics.c_auc_score(gbs, X_train, y_train, X_test, y_test, times)
train_acc_bs, test_acc_bs = metrics.i_brier_score(gbs, X_train, y_train, X_test, y_test, times)
train_acc_icpw, test_acc_icpw = metrics.c_index_icpw(gbs, X_train, y_train, X_test, y_test)
train_acc_cindex, test_acc_cindex = metrics.c_index_censored(gbs, X_train, y_train, X_test, y_test)

draw_plot(['Train', train_acc_cindex, train_acc_icpw, train_acc_auc, train_acc_bs],
          ['Test', test_acc_cindex, test_acc_icpw, test_acc_auc, test_acc_bs],
          'GBS model')

with open('files/gbs.npy', 'wb') as f:
    np.save(f, train_plot)
    np.save(f, test_plot)

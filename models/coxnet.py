from models import metrics, prepare_data
import numpy as np
from table import draw_plot
from sksurv.linear_model import CoxnetSurvivalAnalysis

X_train, y_train, X_test, y_test = prepare_data.get_train_test()

random_state = 64

cox_net = CoxnetSurvivalAnalysis(n_alphas=200,
                                 l1_ratio=0.6,
                                 normalize=True,
                                 fit_baseline_model=True)

times = np.arange(365, 1826, 30)
cox_net.fit(X_train, y_train)

((train_plot, train_acc_auc), (test_plot, test_acc_auc)) = metrics.c_auc_score(cox_net, X_train, y_train, X_test, y_test, times)
train_acc_bs, test_acc_bs = metrics.i_brier_score(cox_net, X_train, y_train, X_test, y_test, times)
train_acc_icpw, test_acc_icpw = metrics.c_index_icpw(cox_net, X_train, y_train, X_test, y_test)
train_acc_cindex, test_acc_cindex = metrics.c_index_censored(cox_net, X_train, y_train, X_test, y_test)

draw_plot(['Train', train_acc_cindex, train_acc_icpw, train_acc_auc, train_acc_bs],
          ['Test', test_acc_cindex, test_acc_icpw, test_acc_auc, test_acc_bs],
          'CoxNet model')

with open('files/coxnet.npy', 'wb') as f:
    np.save(f, train_plot)
    np.save(f, test_plot)

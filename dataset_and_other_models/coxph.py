import metrics, prepare_data
import numpy as np
from table import draw_plot
from sksurv.linear_model import CoxPHSurvivalAnalysis
import matplotlib.pyplot as plt

def surv_plot():
    X_test_sel = X_test[0:10, :]
    surv_funcs = cox_ph.predict_survival_function(X_test_sel)

    i = 0
    for fn in surv_funcs:
        plt.step(fn.x, fn(fn.x), where="post", label=str(i))
        i += 1
        plt.ylabel("Survival probability")

    plt.xlabel("Time in days")
    plt.legend()
    plt.xlim(0, 3000)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig('files/COX PH Surv.png')
    plt.show()

# Data
X_train, y_train, X_test, y_test = prepare_data.get_train_test()

# COX PH with Best Params.
random_state = 64
# Approved
# Old:: best_params = {'alpha': 10, 'n_iter': 50, 'ties': 'efron', 'tol': 1e-09}
best_params = {'alpha': 0, 'n_iter': 50, 'ties': 'efron', 'tol': 1e-09}

cox_ph = CoxPHSurvivalAnalysis(alpha=10, ties="efron", n_iter=50, tol=1e-9)

# 5 Years Analysis
times = np.arange(365, 1826, 30)
cox_ph.fit(X_train, y_train)

((train_plot, train_acc_auc), (test_plot, test_acc_auc)) = metrics.c_auc_score(cox_ph, X_train, y_train, X_test, y_test, times)
train_acc_bs, test_acc_bs = metrics.i_brier_score(cox_ph, X_train, y_train, X_test, y_test, times)
train_acc_icpw, test_acc_icpw = metrics.c_index_icpw(cox_ph, X_train, y_train, X_test, y_test)
train_acc_cindex, test_acc_cindex = metrics.c_index_censored(cox_ph, X_train, y_train, X_test, y_test)

# Plot of Metrics
draw_plot(['Train', train_acc_cindex, train_acc_icpw, train_acc_auc, train_acc_bs],
          ['Test', test_acc_cindex, test_acc_icpw, test_acc_auc, test_acc_bs],
          'COXPH model')

with open('files/coxph.npy', 'wb') as f:
    np.save(f, train_plot)
    np.save(f, test_plot)

surv_plot()
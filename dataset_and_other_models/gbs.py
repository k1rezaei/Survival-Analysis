import metrics, prepare_data
import numpy as np
from table import draw_plot
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import matplotlib.pyplot as plt
import pandas as pd

def surv_plot():
    X_test_sel = X_test[0:10, :]
    surv_funcs = gbs.predict_survival_function(X_test_sel)

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
    plt.savefig('files/GBS Surv.png')
    plt.show()



# Data
X_train, y_train, X_test, y_test = prepare_data.get_train_test()

# GBS with Best Params.
random_state = 64
# Approved
# Old:: best_params = {'learning_rate': 0.1, 'loss': 'coxph', 'min_samples_leaf': 8, 'min_samples_split': 2, 'n_estimators': 500}
# New
best_params = {'learning_rate': 0.05, 'loss': 'coxph', 'min_samples_leaf': 8, 'min_samples_split': 2, 'n_estimators': 1000}
gbs = GradientBoostingSurvivalAnalysis(loss="coxph",
                                       learning_rate=0.05,
                                       n_estimators=1000,
                                       min_samples_split=2,
                                       min_samples_leaf=8)

# 5 Years Analysis
times = np.arange(365, 1826, 30)
gbs.fit(X_train, y_train)

# Metrics
((train_plot, train_acc_auc), (test_plot, test_acc_auc)) = metrics.c_auc_score(gbs, X_train, y_train, X_test, y_test, times)
train_acc_bs, test_acc_bs = metrics.i_brier_score(gbs, X_train, y_train, X_test, y_test, times)
train_acc_icpw, test_acc_icpw = metrics.c_index_icpw(gbs, X_train, y_train, X_test, y_test)
train_acc_cindex, test_acc_cindex = metrics.c_index_censored(gbs, X_train, y_train, X_test, y_test)

# Plot of Metrics
draw_plot(['Train', train_acc_cindex, train_acc_icpw, train_acc_auc, train_acc_bs],
          ['Test', test_acc_cindex, test_acc_icpw, test_acc_auc, test_acc_bs],
          'GBS model')

features = ['Y (مرده؟)', 'AF', 'AQ', 'AY', 'BC', 'BE', 'BG', 'BH', 'BI', 'BL', 'BM', 'BP',
       'CY', 'DC', 'DE', 'DW', 'DX', 'DY', 'Age', 'metastasis',
       'secondprim']
importances = gbs.feature_importances_
importances_df = pd.DataFrame({'Features': features, 'Importances': importances})
importances_df.plot.bar(x='Features', y='Importances', color='teal')
print(importances)
plt.show()

with open('files/gbs.npy', 'wb') as f:
    np.save(f, train_plot)
    np.save(f, test_plot)

surv_plot()

# LaTeX Draw Plot
print('[', end='')
for x in gbs.feature_importances_:
    print(f'{round(x, 8)}, ', end='')

print()
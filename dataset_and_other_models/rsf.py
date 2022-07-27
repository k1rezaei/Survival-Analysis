import metrics, prepare_data
import numpy as np
from table import draw_plot
from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt


def surv_plot():
    X_test_sel = X_test[0:10, :]
    surv_plot = rsf.predict_survival_function(X_test_sel, return_array=True)

    for i, s in enumerate(surv_plot):
        plt.step(rsf.event_times_, s, where="post", label=str(i))

        # LaTeX Plot Drawing ...
        # print(f'%{i}')
        # print(f'\\addplot+[const plot, no marks, thick, c{i+1}]' +  'coordinates {', end='')
        # for j in range(len(rsf.event_times_)):
        #     print(f'({rsf.event_times_[j]}, {s[j]})', end=' ')
        # print('};')


    plt.ylabel("Survival probability")
    plt.xlabel("Time in days")
    plt.legend()
    plt.grid(True)
    plt.savefig('files/RSF Surv2.png', dpi=400)


# Data
X_train, y_train, X_test, y_test = prepare_data.get_train_test()

# RSF Model with Best Params
random_state = 64
# Approved
# Old:: best_params = {'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 500}
# New
best_params = {'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 50}
rsf = RandomSurvivalForest(n_estimators=50,
                           min_samples_split=3,
                           min_samples_leaf=3,
                           max_features="sqrt",
                           n_jobs=1,
                           random_state=random_state)

# 5 Years Analysis
times = np.arange(365, 1826, 30)
rsf.fit(X_train, y_train)

# Metrics
((train_plot, train_acc_auc), (test_plot, test_acc_auc)) = metrics.c_auc_score(rsf, X_train, y_train, X_test, y_test,
                                                                               times)
train_acc_bs, test_acc_bs = metrics.i_brier_score(rsf, X_train, y_train, X_test, y_test, times)
train_acc_icpw, test_acc_icpw = metrics.c_index_icpw(rsf, X_train, y_train, X_test, y_test)
train_acc_cindex, test_acc_cindex = metrics.c_index_censored(rsf, X_train, y_train, X_test, y_test)

# Plot of Metrics
draw_plot(['Train', train_acc_cindex, train_acc_icpw, train_acc_auc, train_acc_bs],
          ['Test', test_acc_cindex, test_acc_icpw, test_acc_auc, test_acc_bs],
          'RSF model')

# Save Plots
with open('files/rsf.npy', 'wb') as f:
    np.save(f, train_plot)
    np.save(f, test_plot)

surv_plot()

features = ['Y', 'AF', 'AQ', 'AY', 'BC', 'BE', 'BG', 'BH', 'BI', 'BL', 'BM', 'BP',
                'CY', 'DC', 'DE', 'DW', 'DX', 'DY', 'Age', 'metastasis',
                'secondprim']

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rsf, n_iter=15, random_state=random_state)
perm.fit(X_test, y_test)
eli5.show_weights(perm, feature_names=features)

print(perm.feature_importances_.tolist())
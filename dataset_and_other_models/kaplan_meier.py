from models import metrics, prepare_data
import numpy as np
import matplotlib.pyplot as plt

from sksurv.nonparametric import kaplan_meier_estimator

X_train, y_train, X_test, y_test = prepare_data.get_train_test()

event_indicator_train, event_time_train, event_indicator_test, event_time_test = \
        prepare_data.get_event_indicator_time(y_train, y_test)

x, y = kaplan_meier_estimator(
    np.concatenate([event_indicator_train, event_indicator_test]),
    np.concatenate([event_time_train, event_time_test]))
plt.step(x, y, where="post", color='r')

plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.01, 0.1))
plt.xlim(0, 3000)

plt.title('Kaplan-Meier Estimator')
plt.ylabel('Survival Function $S(t)$')
plt.xlabel('Time $t$ (days)')

plt.grid(color='black', linestyle='--', linewidth=0.1)
plt.savefig('kaplan-meier.png')
plt.show()

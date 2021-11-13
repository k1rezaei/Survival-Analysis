import matplotlib.pyplot as plt
import numpy as np


with open('files/rsf.npy', 'rb') as f:
    rsf_train = np.load(f)
    rsf_test = np.load(f)

with open('files/gbs.npy', 'rb') as f:
    gbs_train = np.load(f)
    gbs_test = np.load(f)

with open('files/cwgbs.npy', 'rb') as f:
    cwgbs_train = np.load(f)
    cwgbs_test = np.load(f)

with open('files/coxph.npy', 'rb') as f:
    coxph_train = np.load(f)
    coxph_test = np.load(f)

with open('files/coxnet.npy', 'rb') as f:
    coxnet_train = np.load(f)
    coxnet_test = np.load(f)

times = np.arange(365, 1826, 30)

train = [coxnet_train, coxph_train, gbs_train, rsf_train, cwgbs_train]
test = [coxnet_test, coxph_test, gbs_test, rsf_test, cwgbs_test]

labels = ['coxnet', 'coxph', 'gbs', 'rsf', 'cwgbs']

'''for i in range(5):
    plt.plot(times, train[i], marker="o", color='C{}'.format(i), label=labels[i])
    plt.xlabel("days ($t$) ")
    plt.ylabel("time-dependent $AUC(t)$")
    plt.axhline(np.mean(train[i]), color='C{}'.format(i), linestyle="--")
    plt.legend()

plt.title('AUC based on time for different models on Train Data')
plt.savefig('files/AUC_train.png', dpi=500)
plt.show()'''


for i in range(5):
    plt.plot(times, test[i], marker="o", color='C{}'.format(i), label=labels[i])
    plt.xlabel("days ($t$) ")
    plt.ylabel("time-dependent $AUC(t)$")
    plt.axhline(np.mean(test[i]), color='C{}'.format(i), linestyle="--")
    plt.legend()

plt.title('AUC based on time for different models on Test Data')
plt.savefig('files/AUC_test.png', dpi=500)
plt.show()





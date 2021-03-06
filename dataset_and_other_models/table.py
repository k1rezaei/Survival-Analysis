import matplotlib.pyplot as plt
import pandas as pd


def draw_plot(train_acc: list, test_acc: list, title: str):
    df = pd.DataFrame([train_acc, test_acc],
                      index=['Train', 'Test'],
                      columns=['data\\method', 'c-index censored', 'c-index icpw', 'auc', 'berier score']).round(3)

    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    plt.savefig(f'files/{title}', dpi=400)
    plt.show()

# draw_plot(['text', 1, 2, 3, 4], ['text2', 10, 20, 30, 40], 'ali')

def draw_plot2(train_acc: list, test_acc: list, title: str):
    df = pd.DataFrame([train_acc, test_acc],
                      index=['Train', 'Test'],
                      columns=['data\\method', 'c-index censored', 'c-index icpw']).round(3)

    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

    plt.savefig(f'files/{title}', dpi=400)
    plt.show()


def draw_plot3(train_acc: list, val_acc: list, test_acc: list, title: str):
    df = pd.DataFrame([train_acc, val_acc, test_acc],
                      index=['Train', 'Validation', 'Test'],
                      columns=['data\\method', 'c-index', 'IBS']).round(3)

    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

    plt.savefig(f'pycox/{title}', dpi=400)
    plt.show()


# draw_plot3(['Train', 0.9206144226194521, 0.049805243307903445],
#           ['Validation', 0.7795698924731183, 0.1506111627887327],
#           ['Test', 0.8068132815868909, 0.1161988341042458], 'PC Hazard]')



import json
import matplotlib.pyplot as plt
import numpy as np
from bidi.algorithm import get_display
from arabic_reshaper import reshape

all_features = ['Y', 'AF', 'AQ', 'AY', 'BC', 'BE', 'BG', 'BH', 'BI', 'BL', 'BM', 'BP',
                    'CY', 'DC', 'DE', 'DW', 'DX', 'DY', 'Age', 'metastasis',
                    'secondprim']

labels = ["کاهش وزن", "درگیری غدد لنفاوی", "محل درگیری اولیه", "سطح هیستولوژیک", "سایز تومور",
              "تعداد غدد لنفاوی درگیر",
              "جراحی شدن",
              "پرتو‌درمانی", "شیمی‌درمانی", "برداشت توده", "برداشت توده با برش گردن", 'وضعیت حاشیه',
              "پیگیری منظم بیمار", "رخداد عود", "زمان عود", "جراحی‌ عود",
              "پرتودرمانی عود", "شیمی‌درمانی  عود", "سن", 'متاستاز',
              'بدخیمی دوم']

eng_labels = ['Weight loss', 'Lymph node involvement', 'Site', 'Histologic grade', 'Tumor size',
                  'Number of involved lymph nodes', 'Surgery', 'Radiotherapy', 'Chemotherapy', 'Resection only',
                  'Resection with neck dissection', 'Margin status', 'Regular follow up', 'Recurrence',
                  'Recurrence time', 'Surgery after recurrence', 'Radiotherapy after recurrence',
                  'Chemotherapy after recurrence', 'Age', 'Metastasis', 'Second primary malignancy']

persian_labels = [get_display(reshape(label)) for label in labels]

mapper = {}
print(len(all_features), len(labels))

for i in range(len(labels)):
    # mapper[all_features[i]] = persian_labels[i]
    mapper[all_features[i]] = eng_labels[i]

def analyze(method_name: str, fs_method: str, settings: list):
    scores = []
    data = []


    for setting in settings:
        if setting['Method'] == method_name and setting['Feature Selection Method'] == fs_method:
            c_index_val = setting['Evaluation']['Val']['c-index']
            c_index_test = setting['Evaluation']['Test']['c-index']
            c_index = (c_index_val + c_index_test) / 2

            scores.append(c_index)
            data.append(setting)

    indices = np.argsort(scores)
    N = len(scores)

    plt_data = plt.hist(scores, bins=30)
    cnt_plt, x_plt = plt_data[0], plt_data[1]
    for i in range(len(cnt_plt)):
        print(f'({round(x_plt[i], 7)}, {cnt_plt[i]})')
    print("------")
    plt.title('distribution of c-index in different runs on different set of features.')
    plt.xlabel('c-index')

    if fs_method == 'semi_exhaustive_search':
        plt.ylabel('using 15 of 21 features')
    elif fs_method == 'exhaustive_search':
        plt.ylabel('using 19 of 21 features')

    plt.savefig(f'analyzer/new_{fs_method}_v2.png', dpi=400)
    plt.show()


    cnt_rem = {}
    cnt_in = {}

    if fs_method == 'semi_exhaustive_search':
        K = 250
    else:
        K = 120

    print(N)

    good_sets = 0

    for k in range(N-1, N-K, -1):
        i = indices[k]

        print(data[i])
        selected_features = data[i]['Selected Features']
        remove_features = list(set(all_features) - set(selected_features))

        if fs_method == 'exhaustive_search' and scores[i] < 0.81:
            break

        good_sets += 1

        print(f'Score: {scores[i]}')
        print(f'Features: {selected_features}')
        print(f'Removed Features: {remove_features}')

        for x in remove_features:
            cnt_rem[x] = cnt_rem.get(x, 0) + 1

        for x in selected_features:
            cnt_in[x] = cnt_in.get(x, 0) + 1

        print('------')

    for x in cnt_rem.keys():
        if fs_method == 'exhaustive_search':
            cnt_rem[x] = cnt_rem[x] * 1.0 / 21
        elif fs_method == 'semi_exhaustive_search':
            cnt_rem[x] = cnt_rem[x] * 1.0 / 318

    for x in cnt_in.keys():
        if fs_method == 'exhaustive_search':
            cnt_in[x] = cnt_in[x] * 1.0 / good_sets - (19/21)
        elif fs_method == 'semi_exhaustive_search':
            cnt_in[x] = cnt_in[x] * 1.0 / good_sets

    bar = []
    bar2 = []
    features = []
    for x in all_features:
        if x not in cnt_rem:
            cnt_rem[x] = 0
        if x not in cnt_in:
            cnt_in[x] = 0

        features.append(mapper[x])
        bar.append(cnt_rem[x])
        bar2.append(cnt_in[x])

    if fs_method == 'exhaustive_search':
        threshold = 0.81
    else:
        threshold = 0.8

    draw_plot_and_save(name=features, value=bar, save_path='anti_'+fs_method,
                       title=f'Probability that each feature does not appear in a subset which is able\n to train the model with accuracy greater than threshold={threshold}')
    draw_plot_and_save(name=features, value=bar2, save_path='ok_' + fs_method,
                       title=f'Probability that each feature appears in a subset which is able\n to train the model with accuracy greater than threshold={threshold}')



def draw_plot_and_save(name, value, save_path, title):
    plt.rcParams.update({'font.size': 21})
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.barh(name, value)

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)

    # Show top values
    ax.invert_yaxis()

    # Add Plot Title
    ax.set_title(get_display(reshape(title)), loc='left', size=16)

    print(list(zip(name, value)))
    for i in range(len(name)):
        print('\\rl{' + labels[i] + '}/' + str(round(value[i], 3)) + '/inrect,')

    for i in range(len(name)):
        print(eng_labels[i] + '/' + str(round(value[i], 3)) + '/inrect,')

    plt.tight_layout()
    plt.savefig(f'analyzer/new_feature_importance_{save_path}.png', dpi=400)
    plt.show()


if __name__ == '__main__':
    # Semi-Exhaustive Search
    # f = open('runs/log_hazard_new_experiments.txt')

    # Exhaustive Search
    f = open('runs/log_hazard_new_experiments_exhaustive.txt')

    settings = json.load(f)
    print(len(settings))

    # Semi Exhaustive
    # analyze(method_name='Logistic Hazard', fs_method='semi_exhaustive_search', settings=settings)

    # Exhaustive
    analyze(method_name='Logistic Hazard', fs_method='exhaustive_search', settings=settings)


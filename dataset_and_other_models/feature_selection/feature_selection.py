import pandas as pd
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from arabic_reshaper import reshape
# from Resources.models import prepare_data, metrics
import seaborn as sns
import numpy as np


def get_importance_plot(importance: list, title: str, filename: str):
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

    print(len(labels), len(eng_labels))
    persian_labels = [get_display(reshape(label)) for label in labels]

    features = ['Y', 'AF', 'AQ', 'AY', 'BC', 'BE', 'BG', 'BH', 'BI', 'BL', 'BM', 'BP',
                'CY', 'DC', 'DE', 'DW', 'DX', 'DY', 'Age', 'metastasis',
                'secondprim']

    # name = persian_labels
    name = eng_labels
    value = [rsf_importance, gbs_importance]
    plt.rcParams.update({'font.size': 21})

    # Figure Size
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    for i in range(2):
        # Horizontal Bar Plot
        ax[i].barh(name, value[i])

        ######### for diagram ...
        ans = ''
        print(i)
        for j in range(len(name)):
            ans += f'{name[j]}' + '/' + f'{round(value[i][j], 3)}' + '/inrect,\n'
        print(ans)
        print("------------")
        #########

        # Remove axes splines
        for s in ['top', 'bottom', 'left', 'right']:
            ax[i].spines[s].set_visible(False)

        # Remove x, y Ticks
        ax[i].xaxis.set_ticks_position('none')
        ax[i].yaxis.set_ticks_position('none')

        # Add padding between axes and labels
        ax[i].xaxis.set_tick_params(pad=5)
        ax[i].yaxis.set_tick_params(pad=10)

        # Add x, y gridlines
        ax[i].grid(b=True, color='grey',
                   linestyle='-.', linewidth=0.5,
                   alpha=0.2)

        # Show top values
        ax[i].invert_yaxis()
    ax[0].set_title('RSF')
    ax[1].set_title('GBS')
    plt.tight_layout()

    # Add annotation to bars
    # for i in ax.patches:
    #     plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
    #              str(round((i.get_width()), 2)),
    #              fontsize=10, fontweight='bold',
    #              color='grey')

    # Add Plot Title
    # ax.set_title(get_display(reshape(title)),
    #              loc='center', )

    # Show Plot
    # plt.savefig(f'feature_selection/{filename}')
    plt.savefig(f'{filename}', dpi=400)
    plt.show()


def corr_plot():
    X = pd.read_csv('./dataset/train_x.csv')
    cor = X.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(cor, annot=True)
    plt.savefig('corr.png')
    plt.show()


if __name__ == '__main__':
    features = ['Y', 'AF', 'AQ', 'AY', 'BC', 'BE', 'BG', 'BH', 'BI', 'BL', 'BM', 'BP',
                'CY', 'DC', 'DE', 'DW', 'DX', 'DY', 'Age', 'metastasis',
                'secondprim']

    # gbs_importance = [0.01586118, 0.02269904, 0.02534627, 0.02335027, 0.13774357, 0.08562037, 0.06410807, 0.16028378,
    #                   0.02433251, 0.00305983, 0.01763342, 0.0059101, 0.0232246, 0.0988998, 0.03733395,
    #                   0.0273809, 0.0066915, 0.00649037, 0.16351086, 0.01473074, 0.]

    gbs_importance = [0.01757924, 0.01760302, 0.05340613, 0.03376705, 0.14453245, 0.0658953, 0.09726644, 0.10325038,
                      0.02943433, 0.00097165, 0.01385057, 0.02341436, 0.02109995, 0.04018313, 0.14207674, 0.03443186,
                      0.01065766, 0.01478682, 0.13483664, 0.0009563, 0.0]

    # rsf_importance = [-0.001582506114228203,
    #                   -0.004373471442957884,
    #                   0.008919579916558739,
    #                   0.005236656596173169,
    #                   0.020975399223133333,
    #                   0.0075097108329736555,
    #                   0.048367141418500896,
    #                   0.017493885771831367,
    #                   0.023622500359660476,
    #                   -0.0004891382534887523,
    #                   0.004776291181124977,
    #                   0.005639476334340351,
    #                   0.01153790821464534,
    #                   0.01035822183858434,
    #                   0.014530283412458603,
    #                   0.005179110919292159,
    #                   -8.631851532157065e-05,
    #                   -0.002273054236800494,
    #                   0.026758739749676273,
    #                   -0.0014098690835851284,
    #                   0.0]

    rsf_importance = [0.0010111455819832675, -0.0036768930253934756, 0.014776513845800343, -0.00022980581408703997,
                      0.015328047799609383, 0.016591979777088418, 0.03129955187866258, 0.009789727680110357,
                      0.025922095829024525, -0.00025278639549577433, 0.004274388142020051, 0.0025278639549580983,
                      0.006066873491899412, 0.023440193036883872, 0.05915201654601866, 0.0016086406986097165,
                      0.0026427668620016367, 0.007123980236700037, 0.01983224175571648, 0.0036768930253935944, 0.0]

    rsf = np.array(rsf_importance)
    rsf -= np.min(rsf)
    rsf_importance = rsf.tolist()

    get_importance_plot(rsf_importance, gbs_importance, 'importance_v2.png')

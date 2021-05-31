import sys
import matplotlib.pyplot as plt

sys.path.append('../../kernel_logrank/utils')
sys.path.append('../data')
sys.path.append('../../kernel_logrank/tests')
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['font.size'] = 25
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"

# plt.rcParams['text.latex.preamble'] = "\\usepackage{subdepth}, \\usepackage{type1cm}"
plt.rcParams['text.latex.preamble'] = "\\usepackage{type1cm}"

data = 'colon'

if data == 'biofeedback':
    data = pd.read_csv('../data/biofeedback.txt', sep='\t', lineterminator='\n')
    mask = (data.bfb == 1)

    ax = plt.subplot()

    kmf_no_treatment = KaplanMeierFitter()
    ax = kmf_no_treatment.fit(data.loc[~mask, 'thdur'], data.loc[~mask, 'success'],
                              label='No biofeedback', alpha=1).plot_survival_function(ax=ax, linestyle='-')

    kmf_treatment = KaplanMeierFitter()
    ax = kmf_treatment.fit(data.loc[mask, 'thdur'], data.loc[mask, 'success'],
                           label='Biofeedback', alpha=1).plot_survival_function(ax=ax, linestyle='--')
    plt.xlabel('Time')
    plt.ylim(0, 1.02)

    plt.ylabel('Survival')
    plt.tight_layout()
    plt.savefig('biofeedback_km_curves.pdf')
    plt.show()

#
linestyles = [':', '-.', '--', '-']
#
#
# if data == 'loans':
#     data = pd.read_csv('../data/loan_data')
#     data['loan_band'] = pd.cut(data.LoanOriginalAmount2, [0, 0.2, 0.3, 0.8, 2.6])
#     plt.tight_layout()
#     plt.show()
#
#     loan_bands = data['loan_band'].unique()
#     band_names = [4, 3, 2, 1]
#     ax = plt.subplot()
#
#     for i in range(4):
#         mask = data.loan_band == loan_bands[i]
#         band_name = band_names[i]
#         data[mask]
#         naf = NelsonAalenFitter()
#         times = np.linspace(0,1500,50)
#
#         fitted = naf.fit(data.loc[mask, 'time'], data.loc[mask, 'status'],
#                                       label='cum_hazard')
#         cum_hazard_df = fitted.cumulative_hazard_
#
#         cum_hazard = cum_hazard_df['cum_hazard'].to_numpy()
#         times = cum_hazard_df.index.to_numpy()
#         ax = plt.plot(times, np.log(cum_hazard), label='G' + str(band_name), linestyle=linestyles[i])
#
#     plt.legend()
#     plt.xlabel('Time')
#     plt.ylabel('Log cumulative hazard')
#     plt.tight_layout()
#     plt.show()

if data == 'colon':
    data = pd.read_csv('../data/colon')
    data = data[data.etype == 2]
    data['age_band'] = pd.qcut(data.age, 4)
    print(data.head())
    age_bands = data.age_band.unique().sort_values()
    print('age bands', age_bands)
    ax = plt.subplot()

    for i in range(4):
        mask = data.age_band == age_bands[i]
        print('num individuals in age band', age_bands[i], 'equals', np.sum(mask))
        naf = NelsonAalenFitter()

        fitted = naf.fit(data.loc[mask, 'time'], data.loc[mask, 'status'],
                         label='cum_hazard')
        cum_hazard_df = fitted.cumulative_hazard_

        cum_hazard = cum_hazard_df['cum_hazard'].to_numpy()
        times = cum_hazard_df.index.to_numpy()
        ax = plt.plot(times, cum_hazard, label='Q' + str(i+1), linestyle=linestyles[i])
        print(f'i plus 1 is {i+1}, and age band {age_bands[i]}')
    plt.legend()
    plt.xlabel('Time (in days)')
    plt.ylabel('Cumulative hazard')
    plt.tight_layout()
    plt.savefig('cumulative_hazard_colon.pdf')
    plt.show()

#     #
#     # loan_bands = data['loan_band'].unique()
#     # band_names = [4, 3, 2, 1]
#     # ax = plt.subplot()
#     #
#     # for i in range(4):
#     #     mask = data.loan_band == loan_bands[i]
#     #     band_name = band_names[i]
#     #     data[mask]
#     #     naf = NelsonAalenFitter()
#     #     times = np.linspace(0,1500,50)
#     #
#     #     fitted = naf.fit(data.loc[mask, 'time'], data.loc[mask, 'status'],
#     #                                   label='cum_hazard')
#     #     cum_hazard_df = fitted.cumulative_hazard_
#     #
#     #     cum_hazard = cum_hazard_df['cum_hazard'].to_numpy()
#     #     times = cum_hazard_df.index.to_numpy()
#     #     ax = plt.plot(times, np.log(cum_hazard), label='G' + str(band_name), linestyle=linestyles[i])
#     #
#     # plt.legend()
#     # plt.xlabel('Time')
#     # plt.ylabel('Log cumulative hazard')
#     # plt.tight_layout()
#     # plt.show()


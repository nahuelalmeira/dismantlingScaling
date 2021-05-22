import os
import sys
import pathlib
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
WINDOW_WIDTH = 101
POLYORDER = 2

from collections import defaultdict

sys.path.append('..')
#from mpl_settings_v3 import *

CWD = pathlib.Path(__file__).parent
PACKAGE_DIR = CWD.parent.parent
NETWORKS_DIR = PACKAGE_DIR / 'networks'
DATA_DIR = PACKAGE_DIR / 'data'

LINEAR_NETWORKS = ['Lattice', 'PLattice', 'Ld3']
NON_PARAMETRIC_NETWORKS = [
    'Lattice', 'PLattice', 'Ld3', 'DT', 'PDT', 'GG', 'RN'
]

def get_base_network_name(net_type, size, param):
    base_net_name = get_base_net_name(net_type, size, param)
    size_char = 'L' if net_type in LINEAR_NETWORKS else 'N'
    base_net_name_size = base_net_name + '_{}{}'.format(size_char, int(size))
    return base_net_name, base_net_name_size

def get_base_net_name(net_type, size, param):
    if net_type == 'ER':
        base_net_name = 'ER_k{:.2f}'.format(float(param))
    elif net_type == 'RR':
        base_net_name = 'RR_k{:02d}'.format(float(param))
    elif net_type == 'BA':
        base_net_name = 'BA_m{:02d}'.format(int(param))
    elif net_type == 'MR':
        if 'k' in param:
            base_net_name = 'MR_k{:.2f}'.format(float(param[1:]))
        elif 'rMST' == param:
            base_net_name = 'MR_rMST'
        else:
            base_net_name = 'MR_r{:.6f}'.format(float(param[1:]))
    elif net_type in NON_PARAMETRIC_NETWORKS:
        base_net_name = f'{net_type}_param'
    else:
        logging.error(f'{net_type} not supported')
        base_net_name = ''
    return base_net_name

def load_dataframes(
    net_type, 
    param, 
    attack, 
    sizes, 
    min_seeds=None, 
    nseeds=None,
    spline=False,
    method='cpp'
):
    dfs = {}
    for size in sizes:
        try:
            df = load_dataframe(
                net_type, size, param, attack, 
                min_nseeds=min_seeds, spline=spline,
                nseeds=nseeds,
                method=method
            )
            df.attrs['size'] = size
            dfs[size] = df
        except FileNotFoundError:
            continue
    return dfs


def load_dataframe(
    net_type, 
    size, 
    param, 
    attack, 
    nseeds=None, 
    min_nseeds=1,
    spline=False,
    method='cpp'
):
    dir_name = NETWORKS_DIR / net_type
    
    base_net_name, base_net_name_size = get_base_network_name(
        net_type, size, param
    )
    net_dir_name = dir_name / base_net_name / base_net_name_size
    if nseeds:
        file_name = attack + '_nSeeds{:d}_{}.csv'.format(nseeds, method)
        full_file_name = net_dir_name / file_name
    else:
        full_file_name, nseeds = find_dataframe_with_max_nseeds(
            attack, net_dir_name, method=method, min_nseeds=min_nseeds
        )

    df = pd.read_csv(full_file_name, index_col=0)
    df.attrs['nseeds'] = nseeds

    if spline:
        for col in ['Sgcc', 'Nsec', 'meanS']:
            df[col] = savgol_filter(df[col], WINDOW_WIDTH, POLYORDER)

    return df

def find_dataframe_with_max_nseeds(
    attack, 
    net_dir_name, 
    method='cpp', 
    min_nseeds=1
):

    paths = [
        path for path in net_dir_name.glob(f'{attack}_nSeeds*_{method}.csv')
    ]
    nseeds_values = [
        int(str(path).split('nSeeds')[1].split('_')[0]) for path in paths
    ]
    if not nseeds_values or np.max(nseeds_values) < min_nseeds:
        raise FileNotFoundError

    nseeds = np.max(nseeds_values)
    idx = np.argmax(nseeds_values)
    full_file_name = paths[idx]
    
    return full_file_name, nseeds


def get_peaks(dfs, metric='meanS', spline=False):

    fc_values  = []
    max_values = []
    Ngcc_values = []

    for size, df in dfs.items():
        df = dfs[size]
        
        metric_values = df[metric].values
        sgcc_values = df['Sgcc'].values
        if spline:
            metric_values = savgol_filter(metric_values, 101, 2)
            sgcc_values = savgol_filter(sgcc_values, 101, 2)
        
        max_idx = metric_values.argmax()
        max_value = metric_values[max_idx]
        Ngcc_value = size*sgcc_values[max_idx]
        fc_values.append(max_idx/size)
        max_values.append(max_value)
        Ngcc_values.append(Ngcc_value)

    data =  {
        'net_size': dfs.keys(), 
        'peak_position': fc_values, 
        'peak_value': max_values,
        'Ngcc_value': Ngcc_values
    }
    peak_df = pd.DataFrame(data)
    peak_df.attrs['metric'] = metric
    return peak_df


def get_critical_measures(dfs, metric, fc=0):
    sizes = dfs.keys()
    crit_df = get_peaks(dfs, metric)
    crit_df['thresh_value'] = [
        dfs[size][metric][int(fc*size)] for size in sizes
    ]
    crit_df.attrs['fc'] = fc

    return crit_df


def compute_fc_v2(
    dfs, min_f, max_f, method='beta', only_next=False, verbose=False
):

    N_values = sorted(list(dfs.keys()))

    N1_over_N2 = {}
    for N in N_values:
        if method == 'beta':
            N1_over_N2[N] = ((N*dfs[N]['Sgcc'])/dfs[N]['Nsec']).values
        elif method == 'binder':
            N1_over_N2[N] = dfs[N]['meanS']/(N*(dfs[N]['Sgcc']**2)).values

    max_N = N_values[-1]
    mask = np.arange(int(min_f*max_N), int(max_f*max_N))
    n_values = len(mask)
    x = dfs[max_N]['f'][mask].values
    inter_values = []
    s = np.zeros(n_values)
    for i, Na in enumerate(N_values):
        for j, Nb in enumerate(N_values):
            if Nb <= Na:
                continue
            if only_next and j != i+1:
                continue
            mask = np.arange(int(min_f*Na), int(max_f*Na))
            xp = dfs[Na]['f'][mask].values
            fp = N1_over_N2[Na][mask]
            Na_values = np.interp(x, xp, fp)

            mask = np.arange(int(min_f*Nb), int(max_f*Nb))
            xp = dfs[Nb]['f'][mask].values
            fp = N1_over_N2[Nb][mask]
            Nb_values = np.interp(x, xp, fp)
            s += np.fabs(1 - Na_values/Nb_values)
            inter = np.argmin(s)/max_N
            if verbose:
                print(Na, Nb, inter+min_f, sep='\t')
            inter_values.append(inter)
    mean_inter = np.mean(inter_values)
    std_inter = np.std(inter_values)
    fc = min_f + mean_inter
    return fc, std_inter


def linear_regression(sizes, values, scale='loglog'):

    if scale == 'loglog':
        X = np.log(sizes)
        Y = np.log(values)
    elif scale == 'logy':
        X = np.array(sizes)
        Y = np.log(values)
    elif scale == 'logx':
        X = np.log(sizes)
        Y = np.array(values)
    elif scale == 'linear':
        X = np.array(sizes)
        Y = np.array(values)
    else:
        raise ValueError('ERROR: scale', scale, 'not supported')

    coeffs, cov = np.polyfit(X, Y, 1, cov=True)
    errors = np.sqrt(np.diag(cov))
    y_error = errors[0]

    slope, intercept = coeffs[0], coeffs[1]
    Y_pred = intercept + X*slope

    if scale in ['loglog', 'logy']:
        Y_pred = np.exp(Y_pred)

    return Y_pred, slope, y_error

def print_data(metric, slope, y_err):
    str_slope = '{:.3f}+-{:.3f}'.format(slope, y_err)
    if metric == 'Nsec':
        str_exp = r'\beta/\nu = {:.3f}+-{:.3f}'.format(2-slope, y_err)
    elif metric == 'meanS':
        str_exp = r'\gamma/\nu = {:.3f}+-{:.3f}'.format(slope, y_err)
    print(f'{metric}')
    print('slope:', str_slope)
    print(str_exp)
    print('------------------')


attackStats = {
    'Ran': {
        'fmin': 0.47,
        'fmax': 0.54,
        'ymin': 1,
        'ymax': 8
    },
    'Deg': {
        'fmin': 0.27,
        'fmax': 0.34,
        'ymin': 1,
        'ymax': 8
    },
    'DegU': {
        'fmin': 0.35,
        'fmax': 0.4,
        'ymin': 1,
        'ymax': 8
    },
    'Btw': {
        'fmin': 0.,
        'fmax': 0.1,
        'ymin': 1,
        'ymax': 1e6
    },
    'BtwU': {
        'fmin': 0.0,
        'fmax': 0.1,
        'ymin': 1,
        'ymax': 1e9
    },
    'Btw_cutoff2': {
        'fmin': 0.27,
        'fmax': 0.34,
        'ymin': 1,
        'ymax': 8
    },
    'Btw_cutoff3': {
        'fmin': 0.24,
        'fmax': 0.31,
        'ymin': 1,
        'ymax': 8
    },
    'Btw_cutoff4': {
        'fmin': 0.22,
        'fmax': 0.32,
        'ymin': 1,
        'ymax': 8
    },
    'Btw_cutoff6': {
        'fmin': 0.15,
        'fmax': 0.3,
        'ymin': 1,
        'ymax': 8
    },
    'Btw_cutoff8': {
        'fmin': 0.17,
        'fmax': 0.3,
        'ymin': 1,
        'ymax': 8
    },
    'BtwU_cutoff2': {
        'fmin': 0.3,
        'fmax': 0.38,
        'ymin': 1,
        'ymax': 8
    },
    'BtwU_cutoff3': {
        'fmin': 0.24,
        'fmax': 0.34,
        'ymin': 1,
        'ymax': 8
    },
    'BtwU_cutoff4': {
        'fmin': 0.2,
        'fmax': 0.3,
        'ymin': 1,
        'ymax': 8
    },
    'BtwU_cutoff6': {
        'fmin': 0.15,
        'fmax': 0.3,
        'ymin': 1,
        'ymax': 8
    },
    'BtwU_cutoff8': {
        'fmin': 0.15,
        'fmax': 0.24,
        'ymin': 1,
        'ymax': 8
    }
}

def plot_fc():
    ncols = 4
    nrows =  len(attacks) // ncols + 1
    fig, axes = plt.subplots(
        figsize=(8*ncols, 6*nrows), ncols=ncols, nrows=nrows
    )
    axes = axes.flatten()
    for j, attack in enumerate(attacks):

        dfs, fc, fc_err, fmin, fmax = fc_data[attack]

        ax = axes[j]

        if attack in ['Btw', 'BtwU']:
            ax.set_yscale('log')

        ax.set_xlabel('f')
        ax.set_ylabel('S_1/S_2')
        ymin = attackStats[attack]['ymin']
        ymax = attackStats[attack]['ymax']
        ax.text(0.05, 0.1, attack, transform=ax.transAxes)
        ax.set_xlim(fmin, fmax)
        ax.set_ylim(ymin, ymax)
        for i, (size, df) in enumerate(dfs.items()):
            ax.plot(
                df.f, df.S1_over_S2, '-', 
                label=r'${{{}}}$'.format(size), #color=colors[i]
            )
            ax.legend()
        ax.axvline(fc, linestyle='--', color='k', linewidth=2)
        ax.fill_betweenx(
            [ymin, ymax], fc-fc_err, fc+fc_err, color='k', alpha=0.3
        )
    #fig.tight_layout()
    plt.show()

def plot_peaks(attacks_):
    for attack in attacks_:
        fig, axes = plt.subplots(figsize=(12,6), ncols=3)
        fig.suptitle(attack)

        for ax in axes:
            ax.set_xscale('log')
            ax.set_yscale('log')

        for i, metric in enumerate(metrics + ['Ngcc']):

            X, Y, Y_pred, slope, y_err = peak_data[attack][metric]
            str_slope = '{:.3f}+-{:.3f}'.format(slope, y_err)

            ax = axes[i]
            ax.set_xlabel('L')
            ax.set_ylabel(metric)
            ax.plot(X, Y, 'o')
            ax.plot(X, Y_pred, '--k', label=str_slope)

            if metric == 'Nsec':
                ax.text(
                    0.1, 0.7,
                    r'$\beta/\nu$ = ' + \
                    r'${{{:.3f}}}\pm{{{:.3f}}}$'.format(2-slope, y_err),
                    transform=ax.transAxes
                )
            elif metric == 'Ngcc':
                ax.text(
                    0.1, 0.7,
                    r'$\beta/\nu$ = ' + \
                    r'${{{:.3f}}}\pm{{{:.3f}}}$'.format(2-slope, y_err),
                    transform=ax.transAxes
                )
            elif metric == 'meanS':
                ax.text(
                    0.1, 0.7,
                    r'$\gamma/\nu$ = ' + \
                    r'${{{:.3f}}}\pm{{{:.3f}}}$'.format(slope, y_err),
                    transform=ax.transAxes
                )                
            ax.legend()

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    net_type = 'DT'
    param = 'param'
    min_seeds = 2000
    metrics = ['meanS', 'Nsec']

    attacks = [
        'Ran', 
        'Deg',
        'DegU',
        'Btw_cutoff2',
        'Btw_cutoff3',
        'Btw_cutoff4',
        'Btw_cutoff6',
        'Btw_cutoff8',
        'BtwU_cutoff2',
        'BtwU_cutoff3',
        'BtwU_cutoff4',
        'BtwU_cutoff6',
        'BtwU_cutoff8',
        'Btw',
        'BtwU',
    ]

    sizes = [
        512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536, 131072
    ]

    ## Compute data 

    fc_data = {}
    peak_data = {}
    output_data = defaultdict(list)
    output_data['attack'] = attacks

    for attack in attacks:
        print(attack)
        peak_data[attack] = {}

        dfs = load_dataframes(
            net_type, param, attack, sizes, min_seeds=min_seeds, spline=True
        )
        ## Keep only the 5 largest values
        largest_sizes = list(sorted(dfs.keys()))[-5:]
        #largest_sizes = list(sorted(dfs.keys()))
        dfs = {size: df for size, df in dfs.items() if size in largest_sizes}

        print('sizes:', largest_sizes)
        for size, df in dfs.items():
            print('---', size, df.attrs['nseeds'])

        ## Compute fc
        fmin = attackStats[attack]['fmin']
        fmax = attackStats[attack]['fmax']
        fc, fc_err = compute_fc_v2(dfs, fmin, fmax)
        for size, df in dfs.items():
            df['S1_over_S2'] = size * df.Sgcc / df.Nsec
        fc_data[attack] = [dfs, fc, fc_err, fmin, fmax]
        output_data['fc'].append(fc)
        output_data['fc_err'].append(fc_err)
        print('fc = {:.5f}+-{:.5f}'.format(fc, fc_err))


        ## Compute exponents
        for metric in metrics:
            crit_df = get_critical_measures(dfs, metric)
            L_values = np.sqrt(crit_df.net_size)
            X = L_values
            Y = crit_df.peak_value
            Y_pred, slope, y_err = linear_regression(X, Y)
            peak_data[attack][metric] = [X, Y, Y_pred, slope, y_err]

            if metric == 'meanS':
                gamma_over_nu = slope
                output_data['gamma_over_nu'].append(gamma_over_nu)
                output_data['gamma_over_nu_err'].append(y_err)

                Y2 = crit_df.Ngcc_value
                Y_pred2, slope2, y_err2 = linear_regression(X, Y2)
                peak_data[attack]['Ngcc'] = [X, Y2, Y_pred2, slope2, y_err2]
                beta_over_nu_2 = 2-slope2
                output_data['beta_over_nu_2'].append(beta_over_nu_2)
                output_data['beta_over_nu_2_err'].append(y_err2)
            elif metric == 'Nsec':
                beta_over_nu = 2-slope
                output_data['beta_over_nu'].append(beta_over_nu)
                output_data['beta_over_nu_err'].append(y_err)

            print_data(metric, slope, y_err)

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(DATA_DIR / 'data.csv', sep=';', index=False)

    ## Plot
    attacks_ = attacks
    attacks_ = ['Btw_cutoff8', 'Btw']
    plot_peaks(attacks_)
    plot_fc()




"""
plt.xscale('log')
plt.yscale('log')
plt.plot(crit_df.net_size, crit_df.peak_position, '-o')
plt.show()

plt.xscale('log')
plt.yscale('log')
plt.plot(crit_df.net_size, crit_df.peak_value, '-o')
plt.show()

plt.xscale('log')
plt.yscale('log')
plt.plot(crit_df.net_size, crit_df.thresh_value, '-o')
plt.show()
"""
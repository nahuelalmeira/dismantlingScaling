import os
import pandas as pd
import numpy as np
from auxiliary import get_base_network_name
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cycler import cycler

fig_dir = os.path.join('..', 'draft', 'figs')
png_dir = os.path.join(fig_dir, 'png')
pdf_dir = os.path.join(fig_dir, 'pdf')

attack_dict = {
    'Ran': r'$\mathrm{Rnd}$', 'Deg': r'$\mathrm{ID}$', 'DegU': r'$\mathrm{RD}$',
    'Btw': r'$\mathrm{IB}$', 'BtwU': r'$\mathrm{RB}$',
    'Eigenvector': r'$\mathrm{IE}$', 'EigenvectorU': r'$\mathrm{RE}$',
    'CI': r'$\mathrm{ICI}$', 'CIU': r'$\mathrm{RCI}$', 'CIU2': r'$\mathrm{RCI2}$',
}

for i in range(2, 257):
    attack_dict['BtwU_cutoff{}'.format(i)] = r'$\mathrm{RB}$' + r'${{{}}}$'.format(i)
    attack_dict['Btw_cutoff{}'.format(i)] = r'$\mathrm{IB}$' + r'${{{}}}$'.format(i)

measures_dict = {
    'Nsec': r'$N_2$',
    'meanS': r'$\langle s \rangle$'
}

letters = [
    r'$\mathrm{(a)}$', r'$\mathrm{(b)}$', r'$\mathrm{(c)}$', 
    r'$\mathrm{(d)}$', r'$\mathrm{(e)}$', r'$\mathrm{(f)}$', 
    r'$\mathrm{(g)}$', r'$\mathrm{(h)}$', r'$\mathrm{(i)}$'
]

attack_colors = {
    'B': '#27647b', 'D': '#ca3542', 'R': '#57575f', 'E': '#50C878',
    'C': '#80471C', # Brown peanut
    'B3': '#6F2DA8', # Purple
    'B4': '#F05E23' # Orange salamander
}


def load_delta(
    net_type, 
    size, 
    param, 
    attack, 
    nseeds=None, 
    min_nseeds=None, 
    return_nseeds=None
):
    dir_name = os.path.join('../networks', net_type)   
    base_net_name, base_net_name_size = get_base_network_name(
        net_type, size, param
    )
    net_dir_name = os.path.join(dir_name, base_net_name,
            base_net_name_size
    )
    if nseeds:
        delta_file_name = os.path.join(net_dir_name,
            f'Delta_values_{attack}_nSeeds{nseeds}.txt'
        )
        if not os.path.isfile(delta_file_name):
            raise FileNotFoundError('File does not exist')
    else:
        files = [file for file in os.listdir(net_dir_name) if 'Delta' in file]
        files = [file for file in files if attack + '_nSeeds' in file]

        if not files:
            pattern = f'Delta_values_{attack}_nSeeds*.txt'
            raise FileNotFoundError(f'No files matching pattern {pattern}')

        nseeds_values = [
            int(file.split('nSeeds')[1].split('.')[0]) for file in files
        ]

        if min_nseeds:
            if np.max(nseeds_values) < min_nseeds:
                raise FileNotFoundError('No file exist with seeds enough')

        nseeds = np.max(nseeds_values)
        idx = np.argmax(nseeds_values)
        file_name = files[idx]
        delta_file_name = os.path.join(net_dir_name, file_name)

    delta_values = np.loadtxt(delta_file_name)

    if return_nseeds:
        return delta_values, nseeds

    return delta_values

def load_deltas(
    net_type, 
    size, 
    param, 
    attack, 
    nseeds=None,
    n_deltas=10
):
    dir_name = os.path.join('../networks', net_type)   
    base_net_name, base_net_name_size = get_base_network_name(
        net_type, size, param
    )
    net_dir_name = os.path.join(dir_name, base_net_name,
            base_net_name_size
    )
    delta_file_name = os.path.join(net_dir_name,
        f'{n_deltas}_delta_values_{attack}_nSeeds{nseeds}.txt'
    )
    if not os.path.isfile(delta_file_name):
        raise FileNotFoundError('File does not exist')

    delta_values = np.loadtxt(delta_file_name)

    return delta_values

def average_delta(
    net_type, 
    param, 
    attack, 
    N_values, 
    nseeds=None, 
    min_nseeds=None
):
    mean_pos_values = []
    std_pos_values = []
    mean_delta_values = []
    std_delta_values = []
    for N in N_values:
        delta_values = load_delta(
            net_type, N, param, attack, nseeds=nseeds, min_nseeds=min_nseeds
        )
        pos, delta = delta_values.mean(axis=0)
        std_pos, std_delta = delta_values.std(axis=0)
        mean_pos_values.append(pos)
        std_pos_values.append(std_pos)
        mean_delta_values.append(delta)
        std_delta_values.append(std_delta)
    return mean_pos_values, std_pos_values, mean_delta_values, std_delta_values

def load_dataframe(
    net_type, 
    size, 
    param, 
    attack, 
    nseeds=None, 
    min_nseeds=None
):
    dir_name = os.path.join('../networks', net_type)   
    base_net_name, base_net_name_size = get_base_network_name(
        net_type, size, param
    )
    net_dir_name = os.path.join(dir_name, base_net_name,
            base_net_name_size
    )
    if nseeds:
        full_file_name = os.path.join(net_dir_name,
            attack + '_nSeeds{:d}_cpp.csv'.format(nseeds)
        )
    else:
        files = [file for file in os.listdir(net_dir_name) if 'cpp' in file]
        files = [file for file in files if attack + '_nSeeds' in file]
        nseeds_values = [
            int(file.split('nSeeds')[1].split('_')[0]) for file in files
        ]
        if not files:
            raise FileNotFoundError
        if min_nseeds:
            if np.max(nseeds_values) < min_nseeds:
                #print(np.max(nseeds_values), min_nseeds)
                raise FileNotFoundError

        nseeds = np.max(nseeds_values)
        idx = np.argmax(nseeds_values)
        file_name = files[idx]
        full_file_name = os.path.join(net_dir_name, file_name)

    df = pd.read_csv(full_file_name, index_col=0)
    df.attrs['nseeds'] = nseeds
    return df


def getPeaks(dfs, measure):
    N_values = sorted(dfs.keys())

    fc_values  = []
    max_values = []

    for N in N_values:
        df = dfs[N]
        max_idx = df[measure].idxmax()
        max_value = df[measure][max_idx]

        fc_values.append(max_idx/N)
        max_values.append(max_value)

    return fc_values, max_values

def get_critical_measures(dfs, measure, fc):

    N_values = sorted(dfs.keys())

    if fc == 'peak':
        fc_values, _ = getPeaks(dfs, measure)
    elif isinstance(fc, float):
        fc_values = [fc] * len(N_values)
    else:
        print('ERROR')

    crit_values = []
    for i, N in enumerate(N_values):
        df = dfs[N]
        fc = fc_values[i]

        if measure == 'Sgcc':
            crit_values.append(N*df[measure][int(fc*N)])
        else:
            crit_values.append(df[measure][int(fc*N)])

    return np.array(crit_values)


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

def compute_crossings(dfs, min_f, max_f, method='beta', only_next=False):

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
            inter_values.append([Na, Nb, inter+min_f])
            print(Na, Nb, inter+min_f, sep='\t')

    return inter_values


def get_rc_values(
    sizes, 
    l_values=None, 
    net_type='DT', 
    param='param', 
    nseeds=None, 
    min_nseeds=None,
    verbose=False, 
    base_attack='BtwU'
):

    if l_values is None:
        l_values = np.arange(2, 100)

    attacks = [base_attack] + [base_attack + f'_cutoff{l}' for l in l_values]
    rc_values = {}
    rc_values_std = {}
    for size in sizes:
        print(size)
        rc_values[size] = []
        rc_values_std[size] = []
        for attack in attacks:
            try:
                delta_values = load_delta(
                    net_type, size, param, attack, 
                    nseeds=nseeds, min_nseeds=min_nseeds
                )
                rc = delta_values[:,0].mean(axis=0)
                rc_std = delta_values[:,0].std(axis=0) / np.sqrt(delta_values.shape[0]-1)
                rc_values[size].append(rc)
                rc_values_std[size].append(rc_std)
            except FileNotFoundError:
                rc_values[size].append(np.NaN)
                rc_values_std[size].append(np.NaN)
            except IndexError:
                if verbose:
                    print(attack)
                rc_values[size].append(np.NaN)
                rc_values_std[size].append(np.NaN)
        rc_values[size] = np.array(rc_values[size])
        rc_values_std[size] = np.array(rc_values_std[size])

    return rc_values, rc_values_std

def get_l_cutoff(
    sizes, 
    threshold=0.01, 
    rc_values=None, 
    net_type='DT', 
    param='param', 
    base_attack='BtwU',
    nseeds=1000
):

    if not rc_values:
        rc_values, rc_values_std = get_rc_values(
            sizes, net_type=net_type,
             param=param, base_attack=base_attack, nseeds=nseeds
        )

    l_cutoff = {}
    for size in sizes:
        rc = rc_values[size][0]
        for i in range(1, len(rc_values[size])):
            l = i + 1
            rc_l = rc_values[size][i]
            if not np.isnan(rc_l):
                diff = (rc_l - rc) / rc
                #diff[diff<0] = threshold/100
                if diff < threshold:
                    l_cutoff[size] = l
                    break
    return l_cutoff

def get_histo(comp_sizes, nbins=None, log=True, density=False):

    if nbins is None:
        nbins = 20

    mask = comp_sizes > 0
    comp_sizes = comp_sizes[mask]
    min_s = np.min(comp_sizes)
    max_s = np.max(comp_sizes)
    if log:
        bins = np.logspace(np.log10(min_s), np.log10(max_s), nbins)
    else:
        bins = np.linspace(min_s, max_s+1, nbins)
    freq, bin_edges = np.histogram(comp_sizes, bins=bins, density=density)
    freq = freq.astype('float')

    if density == False:
        freq_norm = freq / np.diff(bin_edges)
    else:
        freq_norm = freq

    freq_norm[freq_norm==0] = np.NaN
    mask = ~np.isnan(freq_norm)

    X = bins[:-1]
    X = X[mask]
    Y = freq_norm[mask]

    return X, Y
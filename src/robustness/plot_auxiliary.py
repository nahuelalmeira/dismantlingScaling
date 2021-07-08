import os
import logging
import pandas as pd
import numpy as np

from robustness import NETWORKS_DIR, ROOT_DIR
from robustness.auxiliary import get_base_network_name
from robustness.utils import ROOT_DIR

logger = logging.getLogger(__name__)

fig_dir = ROOT_DIR / 'draft' / 'figs'
png_dir = fig_dir / 'png'
pdf_dir = fig_dir / 'pdf'

attack_dict = {
    'Ran': r'$\mathrm{Rnd}$', 
    #'Deg': r'$\mathrm{ID}$', 'DegU': r'$\mathrm{RD}$',
    'Deg': r'$\mathrm{Deg}$', 'DegU': r'$\mathrm{RD}$',
    'Btw': r'$\mathrm{B}$', 'BtwU': r'$\mathrm{RB}$',
    'Eigenvector': r'$\mathrm{IE}$', 'EigenvectorU': r'$\mathrm{RE}$',
    'CI': r'$\mathrm{ICI}$', 'CIU': r'$\mathrm{RCI}$', 
    'CI2': r'$\mathrm{ICI2}$', 'CIU2': r'$\mathrm{RCI2}$',
}


for i in range(2, 257):
    attack_dict[f'BtwU_cutoff{i}'] = r'$\mathrm{RB}$' + r'${{{}}}$'.format(i)
    attack_dict[f'Btw_cutoff{i}'] = r'$\mathrm{B}$' + r'${{{}}}$'.format(i)

measures_dict = {
    'meanS': r'$\langle s \rangle$', 
    'Nsec': r'$S_2 N$', 'varSgcc': r'$\chi$',
    'Sgcc': r'$N_1$', 
    'num_over_denom': r'$\langle s \rangle_{\mathrm{ens}}$', 
    'num': r'$M_2$'
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
    dir_name = NETWORKS_DIR /  net_type
    base_net_name, base_net_name_size = get_base_network_name(
        net_type, size, param
    )
    net_dir_name = dir_name / base_net_name / base_net_name_size

    if nseeds:
        delta_file_name = (
            net_dir_name / f'Delta_values_{attack}_nSeeds{nseeds}.txt'
        )
        if not delta_file_name.is_file():
            raise FileNotFoundError(
                f'File "{delta_file_name}" does not exist'
            )
    else:
        pattern = f'Delta_values_{attack}_nSeeds*'
        files = list(net_dir_name.glob(pattern))
        if not files:
            raise FileNotFoundError(f'No files matching pattern {pattern}')

        nseeds_values = [
            int(str(file).split('nSeeds')[1].split('.')[0]) for file in files
        ]

        if min_nseeds:
            if max(nseeds_values) < min_nseeds:
                raise FileNotFoundError('No file exist with enough seeds')

        nseeds = max(nseeds_values)
        idx = np.argmax(nseeds_values)
        file_name = files[idx]
        delta_file_name = net_dir_name / file_name

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
    dir_name = NETWORKS_DIR / net_type
    base_net_name, base_net_name_size = get_base_network_name(
        net_type, size, param
    )
    net_dir_name = dir_name / base_net_name / base_net_name_size
    if nseeds:
        full_file_name = net_dir_name / (attack + f'_nSeeds{nseeds}_cpp.csv')
    else:
        pattern = f'{attack}_nSeeds*_cpp.csv'
        files = list(net_dir_name.glob(pattern))
        nseeds_values = [
            int(str(file).split('nSeeds')[1].split('_')[0]) for file in files
        ]
        if not files:
            raise FileNotFoundError
        if min_nseeds:
            if max(nseeds_values) < min_nseeds:
                raise FileNotFoundError

        nseeds = max(nseeds_values)
        idx = np.argmax(nseeds_values)
        file_name = files[idx]
        full_file_name = net_dir_name / file_name

    df = pd.read_csv(full_file_name, index_col=0)
    if 'num' in df.columns:
        df['num_over_denom'] = df['num'] / df['denom']
    df.attrs['nseeds'] = nseeds
    return df


def getPeaks(dfs, measure):
    sizes = sorted(dfs.keys())

    fc_values  = []
    max_values = []

    for size in sizes:
        df = dfs[size]
        N = df.shape[0]
        max_idx = df[measure].idxmax()
        max_value = df[measure][max_idx]

        fc_values.append(max_idx/N)
        max_values.append(max_value)

    return fc_values, max_values

def get_critical_measures(dfs, measure, fc, include_gcc=False):

    sizes = sorted(dfs.keys())

    if fc == 'peak':
        if measure == 'Sgcc':
            peak_measure = 'Nsec'
        else:
            peak_measure = measure
        fc_values, _ = getPeaks(dfs, peak_measure)
    elif isinstance(fc, float):
        fc_values = [fc] * len(sizes)
    else:
        print('ERROR')
    crit_values = []
    for fc, size in zip(fc_values, sizes):
        df = dfs[size]
        N = df.shape[0]

        idx = int(fc*N)
        if measure == 'Sgcc':
            crit_value = N*df[measure][idx]
        elif measure == 'num':
            crit_value = df[measure][idx]/N
            if include_gcc:
                crit_value += df.Sgcc.values[idx]
        else:
            crit_value = df[measure][idx]
        crit_values.append(crit_value)
    return np.array(crit_values)


def compute_fc_v2(
    dfs, 
    min_f, 
    max_f, 
    method='beta', 
    only_next=False, 
    verbose=False, 
    t=1.96
):
    sizes = sorted(list(dfs.keys()))
    
    N1_over_N2 = {}
    max_size = 0
    for size in sizes:
        df = dfs[size]
        N = df.shape[0]
        if N > max_size:
            Nmax = N
        if method == 'beta':
            N1_over_N2[size] = ((N*df.Sgcc)/df.Nsec).values
        elif method == 'binder':
            N1_over_N2[size] = df.meanS/(N*(df.Sgcc**2)).values
    max_size = sizes[-1]
    mask = np.arange(int(min_f*Nmax), int(max_f*Nmax))
    n_values = len(mask)
    x = dfs[max_size].f[mask].values
    inter_values = []
    s = np.zeros(n_values)
    for i, size_a in enumerate(sizes):
        for j, size_b in enumerate(sizes):
            if size_b <= size_a:
                continue
            if only_next and j != i+1:
                continue
                
            dim_factor_a = int(dfs[size_a].shape[0] / size_a)
            dim_factor_b = int(dfs[size_b].shape[0] / size_b)
                
            mask = np.arange(int(min_f*size_a*dim_factor_a), int(max_f*size_a*dim_factor_a))
            xp = dfs[size_a].f[mask].values
            fp = N1_over_N2[size_a][mask]
            size_a_values = np.interp(x, xp, fp)

            mask = np.arange(int(min_f*size_b*dim_factor_b), int(max_f*size_b*dim_factor_b))
            xp = dfs[size_b].f[mask].values
            fp = N1_over_N2[size_b][mask]
            size_b_values = np.interp(x, xp, fp)
            s += np.fabs(1 - size_a_values/size_b_values)
            inter = np.argmin(s)/Nmax
            if verbose:
                print(size_a, size_b, inter+min_f, sep='\t')
            inter_values.append(inter)
    mean_inter = np.mean(inter_values)
    std_inter = np.std(inter_values) * t
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
            logger.info(Na, Nb, inter+min_f, sep='\t')

    return inter_values


def get_simulation_paths(net_dir_name, base_delta_file_name):
    pattern = base_delta_file_name.format('*', '*')
    patt_gen = net_dir_name.glob(pattern)
    names = []
    for p in patt_gen:
        attack = '_'.join(p.stem.split('_')[2:-1])
        nseeds = int(p.stem.split('nSeeds')[1])
        names.append((attack, nseeds))
    names_df = pd.DataFrame(names, columns=['attack', 'nseeds'])
    return names_df

def filter_delta_names(
    names_df, 
    base_attack, 
    nseeds=None, 
    min_nseeds=None
):
    filtered_names_df = names_df.loc[
        (names_df.attack==base_attack) |
        (names_df.attack.str.startswith(f'{base_attack}_'))
    ]
    if nseeds:
        filtered_names_df = (
            filtered_names_df
            .loc[filtered_names_df.nseeds==nseeds]
            .set_index('attack')
            .nseeds
        )
    else:
        if not min_nseeds:
            min_nseeds = 0
        filtered_names_df = filtered_names_df.groupby(by='attack').max()
        filtered_names_df = (
            filtered_names_df
            .loc[filtered_names_df.nseeds>=min_nseeds]
            .nseeds
        )
    return filtered_names_df

def get_delta_data(
    net_type,
    param,
    size,
    base_attack,
    nseeds=None,
    min_nseeds=None
    
):
    dir_name = NETWORKS_DIR /  net_type
    base_net_name, base_net_name_size = get_base_network_name(
        net_type, size, param
    )
    net_dir_name = dir_name / base_net_name / base_net_name_size
    
    base_delta_file_name = 'Delta_values_{}_nSeeds{}'
    
    names_df = get_simulation_paths(net_dir_name, base_delta_file_name)
    filtered_names_df = filter_delta_names(
        names_df, base_attack, nseeds=nseeds, min_nseeds=min_nseeds
    )   
    delta_data = {}
    for attack, nseeds in filtered_names_df.iteritems():
        delta_file_name = (
            net_dir_name / 
            (base_delta_file_name.format(attack, nseeds) + '.txt')
        )
        delta_data[attack] = np.loadtxt(delta_file_name)
    return delta_data

def get_rc_values(
    net_type, 
    param, 
    base_attack, 
    sizes, 
    nseeds=None, 
    min_nseeds=None
):
    rc_values = {}
    rc_values_std = {}
    for size in sizes:
        delta_data_size = get_delta_data(
            net_type, param, size, base_attack,
            nseeds=nseeds, min_nseeds=min_nseeds
        )
        rc_values[size] = {
            attack: delta_data[:,0].mean() 
            for attack, delta_data in delta_data_size.items()
        }
        rc_values_std[size] = {
            attack: delta_data[:,0].std() 
            for attack, delta_data in delta_data_size.items()
        }
    return rc_values, rc_values_std

def get_l_from_attack_name(attack):
    try:
        l = int(attack.split('cutoff')[1])
    except IndexError:
        l = np.nan
    return l

def get_l_cutoff(
    net_type,
    param,
    base_attack,
    sizes, 
    threshold=0.01,
    nseeds=None,
    min_nseeds=None
):
    l_cutoffs = {}
    for size in sizes:
        delta_data = get_delta_data(
            net_type, param, size, base_attack, 
            min_nseeds=min_nseeds, nseeds=nseeds
        )
        lmax = int(np.nanmax([
            get_l_from_attack_name(attack) for attack in delta_data.keys()
        ])) + 1

        aux = np.array([np.nan]*lmax)
        base_data = delta_data[base_attack]
        for attack, data in delta_data.items():
            if attack == base_attack:
                continue

            l = get_l_from_attack_name(attack)
            nseeds = data.shape[0]
            rc_values_cutoff = data[:,0]
            rc_values_base = base_data[:,0][:nseeds]
            rc = rc_values_base.mean()
            diff = np.fabs((rc_values_cutoff.mean() - rc) / rc)
            aux[l] = diff   
            try:
                l_cutoff = np.where(aux<threshold)[0][0]
            except IndexError:
                l_cutoff = np.nan
            l_cutoffs[size] = l_cutoff

    return l_cutoffs

def _old_get_l_cutoff_2(
    net_type,
    param,
    base_attack,
    sizes, 
    threshold=0.01, 
    rc_values=None, 
    nseeds=None,
    min_nseeds=None
):

    if not rc_values:
        rc_values, rc_values_std = get_rc_values(
            net_type, param, base_attack, sizes, 
            nseeds=nseeds, min_nseeds=min_nseeds
        )

    l_cutoff = {}
    for size, rc_values_size in rc_values.items():
        rc_base = rc_values[size][base_attack]
        aux = []
        for attack, rc_l in rc_values_size.items():
            if attack == base_attack:
                continue
            l = int(attack.split('cutoff')[1])
            diff = (rc_l - rc_base) / rc_base
            aux.append((l, diff))
            
        aux = list(sorted(aux, key=lambda x: x[0]))
        for l, diff in aux:
            if diff < threshold:
                l_cutoff[size] = l
                break
    return l_cutoff

def _old_get_rc_values(
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

    attacks = [base_attack] + [f'{base_attack}_cutoff{l}' for l in l_values]
    rc_values = {}
    rc_values_std = {}
    for size in sizes:
        logger.info(size)
        rc_values[size] = []
        rc_values_std[size] = []
        for attack in attacks:
            try:
                delta_values = load_delta(
                    net_type, size, param, attack, 
                    nseeds=nseeds, min_nseeds=min_nseeds
                )
                rc = delta_values[:,0].mean(axis=0)
                rc_std = (
                    delta_values[:,0].std(axis=0) / 
                    np.sqrt(delta_values.shape[0])
                )
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

def _old_get_l_cutoff(
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

def load_comp_sizes_data(
    net_type, 
    param, 
    size, 
    attack, 
    f_value, 
    dropLargest=1
):
    dir_name = NETWORKS_DIR / net_type
    base_net_name, base_net_name_size = get_base_network_name(
        net_type, size, param
    )
    base_net_dir = dir_name / base_net_name / base_net_name_size
    base_file_name = f'comp_sizes_{attack}_f{f_value}_drop{dropLargest}'
    full_comp_sizes_file = base_net_dir / f'{base_file_name}.txt'
    full_seeds_file = base_net_dir / f'{base_file_name}_seeds.txt'
    comp_sizes = np.loadtxt(full_comp_sizes_file, dtype=int)   
    try: 
        nseeds = len(np.loadtxt(full_seeds_file, dtype=int))
    except Exception:
        logger.warning('Seeds file does not exist')
        nseeds = 1
    return comp_sizes, nseeds
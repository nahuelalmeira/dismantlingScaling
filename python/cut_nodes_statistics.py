import os
import sys
import errno
import tarfile
import igraph as ig
import numpy as np
import pandas as pd
from auxiliary import get_base_network_name, supported_attacks, read_data_file
from auxiliary import load_delta_data

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])

if net_type in ['ER', 'RR', 'BA', 'MR', 'DT']:
    N = int(size)
elif net_type == 'Lattice':
    L = int(size)
    N = L*L
elif net_type == 'Ld3':
    L = int(size)
    N = L*L*L

overwrite = False
if 'overwrite' in sys.argv:
    overwrite = True

verbose = False
if 'verbose' in sys.argv:
    verbose = True

attacks = []
for attack in supported_attacks:
    if attack in sys.argv:
        attacks.append(attack)

print('------- Params -------')
print('net_type =', net_type)
print('param    =', param)
print('min_seed =', min_seed)
print('max_seed =', max_seed)
print('----------------------', end='\n\n')

dir_name = os.path.join('../networks', net_type)
base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
base_network_dir_name = os.path.join(dir_name, base_net_name, base_net_name_size)


for attack in attacks:
    print(attack)
    n_seeds = max_seed - min_seed
    output_file_name = os.path.join(
        base_network_dir_name, 'cut_nodes_stats_' + attack + '_nSeeds{:d}.txt'.format(n_seeds)
    )
    if not overwrite:
        if os.path.isfile(output_file_name):
            continue

    data = []

    valid_its = 0
    for seed in range(min_seed, max_seed):
        if verbose:
            print(seed)
        g, max_pos, delta_max = load_delta_data(net_type, size, param, attack, seed)

        attack_order = g['attack_order']
        to_delete = set(g.vs['oi']).difference(set(attack_order[:max_pos+1]))
        g.delete_vertices(to_delete)

        components = g.components(mode='WEAK')
        gcc = components.giant()

        comp_sizes = sorted([len(c) for c in components], reverse=True)

        if len(comp_sizes) == 1:
            comp_sizes.append(0)

        N_gcc = comp_sizes[0]
        N_sec = comp_sizes[1]
        comp_sizes.remove(N_gcc)
        comp_sizes = np.array(comp_sizes)
        if np.sum(comp_sizes) == 0:
            meanS = np.NaN
        else:
            meanS = np.sum(comp_sizes**2) / np.sum(comp_sizes)

        data.append([N_gcc, N_sec, meanS])
        valid_its += 1

    np.savetxt(output_file_name, data)

    print('Correct seeds = ', valid_its)
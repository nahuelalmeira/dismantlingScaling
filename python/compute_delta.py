import os
import sys
import tarfile
import igraph as ig
import numpy as np
import pandas as pd
from auxiliary import get_base_network_name, supported_attacks, read_data_file, get_number_of_nodes

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])

N = get_number_of_nodes(net_type, size)

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
        base_network_dir_name, 'Delta_values_' + attack + '_nSeeds{:d}.txt'.format(n_seeds)
    )
    if not overwrite:
        if os.path.isfile(output_file_name):
            continue

    delta_max_values = []

    valid_its = 0
    for seed in range(min_seed, max_seed):

        network = base_net_name_size + '_{:05d}'.format(seed)
        attack_dir_name = os.path.join(dir_name, base_net_name, base_net_name_size, network, attack)

        ## Read data
        try:
            aux = read_data_file(attack_dir_name, 'comp_data', reader='numpy')
        except FileNotFoundError:
            continue

        if verbose:
            print(seed)

        len_aux = aux.shape[0]
        len_aux = aux.shape[0]
        if len_aux > N:
            print('ERROR: Seed {}. Len of array is greater than network size'.format(seed))
            continue
        if len_aux < 0.9*N:
            print('ERROR: Seed {}. Len of array is too short'.format(seed))
            continue

        valid_its += 1

        Ngcc_values = aux[:,0][::-1]
        delta_values = np.abs(np.diff(Ngcc_values))
        max_pos = np.argmax(delta_values)
        delta_max = delta_values[max_pos]
        Sgcc_c = Ngcc_values[max_pos] / N

        delta_max_values.append([max_pos/N, delta_max/N, Sgcc_c])

    np.savetxt(output_file_name, delta_max_values)

    print('Correct seeds = ', valid_its)

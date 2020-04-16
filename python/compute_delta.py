import os
import sys
import tarfile
import igraph as ig
import numpy as np
import pandas as pd
from auxiliary import get_base_network_name, supported_attacks

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])

if net_type in ['ER', 'RR', 'BA', 'MR', 'DT']:
    N = int(size)

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

for attack in attacks:
    print(attack)
    n_seeds = max_seed - min_seed
    csv_file_name = os.path.join(dir_name, base_net_name, base_net_name_size, 
                                 '{}_nSeeds{:d}_cpp.csv'.format(attack, n_seeds))
    if not overwrite:
        if os.path.isfile(csv_file_name):
            continue

    delta_max_values = []

    valid_its = 0
    for seed in range(min_seed, max_seed):

        network = base_net_name_size + '_{:05d}'.format(seed)
        base_network_dir_name = os.path.join(dir_name, base_net_name, base_net_name_size)
        attack_dir_name = os.path.join(dir_name, base_net_name, base_net_name_size, network, attack)

        ## Extract network file
        tar_input_name = 'comp_data.tar.gz'
        full_tar_input_name = os.path.join(attack_dir_name, tar_input_name)
        if not os.path.exists(full_tar_input_name):
            continue
        tar = tarfile.open(full_tar_input_name, 'r:gz')
        tar.extractall(attack_dir_name)
        tar.close()

        full_file_name  = os.path.join(attack_dir_name, 'comp_data.txt')
        if not os.path.isfile(full_file_name):
            continue

        if verbose:
            print(seed)

        ## Read data
        aux = np.loadtxt(full_file_name)

        ## Remove network file
        os.remove(full_file_name)

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

        delta_max_values.append([max_pos/N, delta_max])

    n_seeds = max_seed - min_seed
    output_file_name = os.path.join(
        base_network_dir_name, 'Delta_values_' + attack + '_nSeeds{:d}.txt'.format(n_seeds)
    )
    np.savetxt(output_file_name, delta_max_values)

    print('Correct seeds = ', valid_its)

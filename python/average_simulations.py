import os
import sys
import tarfile
import igraph as ig
import numpy as np
import pandas as pd
from auxiliary import get_base_network_name, supported_attacks, read_data_file
from auxiliary import get_number_of_nodes

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

for attack in attacks:
    print(attack)

    n_seeds = max_seed - min_seed
    csv_file_name = os.path.join(dir_name, base_net_name, base_net_name_size,
                                 '{}_nSeeds{:d}_cpp.csv'.format(attack, n_seeds))
    if not overwrite:
        if os.path.isfile(csv_file_name):
            continue

    Ngcc_values     = np.zeros(N)
    Ngcc_sqr_values = np.zeros(N)
    Nsec_values     = np.zeros(N)
    meanS_values    = np.zeros(N)
    chiDelta_values = np.zeros(N)

    valid_its = 0
    for seed in range(min_seed, max_seed):

        network = base_net_name_size + '_{:05d}'.format(seed)
        attack_dir_name = os.path.join(
            dir_name, base_net_name, base_net_name_size, network, attack
            )

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

        Ngcc_values_it = np.append(aux[:,0][::-1], np.repeat(1, (N-len_aux)))
        Ngcc_values += Ngcc_values_it
        Ngcc_sqr_values += Ngcc_values_it**2

        chiDelta_values_it = np.append(np.diff(Ngcc_values_it), 0)
        chiDelta_values += chiDelta_values_it

        Nsec_values_it = np.append(aux[:,1][::-1], np.repeat(1, (N-len_aux)))
        Nsec_values += Nsec_values_it

        meanS_values_it = np.append(aux[:,2][::-1], np.repeat(1, (N-len_aux)))
        meanS_values += meanS_values_it

    varSgcc_values = (Ngcc_sqr_values/valid_its - (Ngcc_values/valid_its)**2) / (N)
    Ngcc_values = Ngcc_values / valid_its
    Sgcc_values = Ngcc_values / N
    Nsec_values = Nsec_values / valid_its
    meanS_values = meanS_values / valid_its
    chiDelta_values = chiDelta_values / valid_its

    d = {
        'f': np.arange(N)/N,
        'Sgcc': Sgcc_values,
        'varSgcc': varSgcc_values,
        'Nsec': Nsec_values,
        'meanS': meanS_values,
        'chiDelta': chiDelta_values
    }
    df = pd.DataFrame(data=d)
    df.to_csv(csv_file_name)

    print('Correct seeds = ', valid_its)

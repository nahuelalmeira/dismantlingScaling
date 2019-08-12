import os
import sys
import igraph as ig
import numpy as np
import pandas as pd
from auxiliary import get_base_network_name

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])

if net_type == 'ER':
    N = int(size)

if 'overwrite' in sys.argv:
    overwrite = True
else:
    overwrite = False

attacks = []
if 'BtwU' in sys.argv:
    attacks.append('BtwU')
if 'DegU' in sys.argv:
    attacks.append('DegU')
if 'Btw' in sys.argv:
    attacks.append('Btw')
if 'Deg' in sys.argv:
    attacks.append('Deg')
if 'Ran' in sys.argv:
    attacks.append('Ran')

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
        attack_dir_name = os.path.join(dir_name, base_net_name, base_net_name_size, network, attack)
        
        full_file_name  = os.path.join(attack_dir_name, 'comp_data.txt')
        if not os.path.isfile(full_file_name):
            continue
        print(seed)

        aux = np.loadtxt(full_file_name)
        len_aux = aux.shape[0]
        len_aux = aux.shape[0]
        if len_aux > N:
            print('ERROR: Len of array is greater than network size')
            continue
        if len_aux < 0.9*N:
            print('ERROR: Len of array is too short')
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

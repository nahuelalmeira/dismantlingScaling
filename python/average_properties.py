import os
import sys
import tarfile
import igraph as ig
import numpy as np
import pandas as pd

from auxiliary import (
    get_base_network_name, supported_attacks, 
    simple_props, get_property_file_name,
    get_number_of_nodes
)

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

python_file_dir_name = os.path.dirname(__file__)
dir_name = os.path.join(python_file_dir_name, '../networks', net_type)
base_net_name, base_net_name_size = get_base_network_name(
    net_type, size, param
)

for attack in attacks:
    print(attack)

    n_seeds = max_seed - min_seed
    csv_file_name = os.path.join(
        dir_name, base_net_name, base_net_name_size,
        'properties_{}_nSeeds{:d}_cpp.csv'.format(attack, n_seeds)
    )
    if not overwrite:
        if os.path.isfile(csv_file_name):
            continue

    prop_values = {}
    prop_values['f'] = np.arange(N)/N
    for prop in simple_props:
        prop_values[prop] = np.zeros(N)

    valid_its = 0
    for seed in range(min_seed, max_seed):

        network = base_net_name_size + '_{:05d}'.format(seed)
        attack_dir_name = os.path.join(
            dir_name, base_net_name, base_net_name_size, network, attack
        )

        if verbose:
            print(seed)

        ## Read data
        for prop in simple_props:
            prop_file_name = get_property_file_name(prop, attack_dir_name)
            prop_values[prop] += np.loadtxt(prop_file_name)

        valid_its += 1

    for prop in simple_props:
        prop_values[prop] = prop_values[prop] / n_seeds

    print(simple_props)
    df = pd.DataFrame(data=prop_values, columns=['f'] + simple_props)
    df.to_csv(csv_file_name)

    print('Correct seeds = ', valid_its)

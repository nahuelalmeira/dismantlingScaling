import os
import sys
import tarfile
import numpy as np
import igraph as ig


from dismantling import get_index_list
from auxiliary import get_base_network_name, supported_attacks, get_edge_weights, read_data_file

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])

overwrite = False
if 'overwrite' in sys.argv:
    overwrite = True

package = 'igraph'
if 'networkit' in sys.argv:
    package = 'networkit'

save_centrality = False
if 'saveCentrality' in sys.argv:
    save_centrality = True

python_file_dir_name = os.path.dirname(__file__)
dir_name = os.path.join(python_file_dir_name, '../networks', net_type)
seeds = range(min_seed, max_seed)

attacks = []
for attack in supported_attacks:
    if attack in sys.argv:
        attacks.append(attack)

base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
base_net_dir = os.path.join(dir_name, base_net_name, base_net_name_size)

for attack in attacks:
    print(attack)
    for seed in seeds:
        net_name = base_net_name_size + '_{:05d}'.format(seed)
        net_dir = os.path.join(base_net_dir, net_name)

        print(net_name)

        output_dir = os.path.join(net_dir, attack)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_name = 'oi_list.txt'
        full_output_name = os.path.join(output_dir, output_name)

        full_c_output_name = None
        if save_centrality:
            c_output_name = 'initial_centrality.txt'
            full_c_output_name = os.path.join(output_dir, c_output_name)

        if os.path.isfile(full_output_name) and overwrite:
            os.remove(full_output_name)

        g = read_data_file(net_dir, net_name, reader=package)

        if 'BtwWU' in attack:
            g.es['weight'] = get_edge_weights(g, net_type, size, param, seed)

        ## Perform the attack
        get_index_list(
            g, attack, full_output_name,
            save_centrality=save_centrality,
            out_centrality=full_c_output_name,
            random_state=seed
        )
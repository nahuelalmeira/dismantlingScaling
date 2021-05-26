import os
import sys
import logging
import numpy as np

from robustness.auxiliary import (
    get_base_network_name, 
    supported_attacks,
    load_delta_data, 
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

logger = logging.getLogger(__name__)
#logger.setLevel(getattr(logging, logging_level))

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

base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
base_network_dir_name = os.path.join(dir_name, base_net_name, base_net_name_size)


for attack in attacks:
    logger.info(attack)
    n_seeds = max_seed - min_seed
    output_file_name = os.path.join(
        base_network_dir_name, f'cut_nodes_stats_{attack}_nSeeds{n_seeds}.txt'
    )
    if not overwrite:
        if os.path.isfile(output_file_name):
            continue

    data = []

    valid_its = 0
    for seed in range(min_seed, max_seed):
        logger.debug(seed)
        g, max_pos, delta_max = load_delta_data(
            net_type, size, param, attack, seed
        )

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

    logger.info('Correct seeds = ', valid_its)
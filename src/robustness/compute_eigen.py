import os
import sys
import logging
import numpy as np

from robustness.auxiliary import (
    get_base_network_name, 
    supported_attacks, 
    read_data_file, 
    get_number_of_nodes,
    get_position
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

logger = logging.getLogger(__name__)
#logger.setLevel(getattr(logging, logging_level))

dir_name = os.path.join('../networks', net_type)
base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
base_network_dir_name = os.path.join(dir_name, base_net_name, base_net_name_size)

for attack in attacks:
    logger.info(attack)
    n_seeds = max_seed - min_seed
    output_file_name = os.path.join(
        base_network_dir_name, 'eigen_' + attack + '_nSeeds{:d}.txt'.format(n_seeds)
    )
    if not overwrite:
        if os.path.isfile(output_file_name):
            continue

    eig_values = []
    valid_its = 0
    for seed in range(min_seed, max_seed):

        logger.debug(seed)

        dir_name = os.path.join('../networks', net_type)
        base_net_name, base_net_name_size = get_base_network_name(
            net_type, size, param
        )
        net_dir_name = os.path.join(
            dir_name, base_net_name,
            base_net_name_size,
            base_net_name_size + '_{:05}'.format(seed)
        )
        attack_dir_name = os.path.join(net_dir_name, attack)
        oi_file_name = os.path.join(attack_dir_name, 'oi_list.txt')
        index_list = np.loadtxt(oi_file_name, dtype='int')

        position = get_position(net_type, size, net_dir=net_dir_name)

        tar_input_name = 'comp_data.tar.gz'
        full_tar_input_name = os.path.join(attack_dir_name, tar_input_name)
        if not os.path.isfile(full_tar_input_name):
            continue

        aux = read_data_file(attack_dir_name, 'comp_data', reader='numpy')


        Sgcc_values = aux[:,0][::-1] / N
        delta_values = np.abs(np.diff(Sgcc_values))
        max_pos = np.argmax(delta_values)
        delta_max = delta_values[max_pos]

        data = position[index_list[:max_pos+1]]
        cov_mat = np.cov(data.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov_mat)

        order = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[order]
        eig_vecs = eig_vecs[order]


        valid_its += 1

        eig_values.append(
            np.array([
                eig_vals[0], eig_vecs[0,0], eig_vecs[0,1], eig_vals[1], 
                eig_vecs[1,0], eig_vecs[1,1]
            ])
        )

    np.savetxt(output_file_name, eig_values)

    print('Correct seeds = ', valid_its)

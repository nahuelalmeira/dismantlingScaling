import os
import sys
import tarfile
import numpy as np
import igraph as ig

from dismantling import get_index_list
from auxiliary import get_base_network_name

supported_attacks = ['Ran', 'Deg', 'DegU', 'Btw', 'BtwU']

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])

if 'overwrite' in sys.argv:
    overwrite = True
else:
    overwrite = False

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

        ## Extract network file
        tar_input_name = net_name + '.tar.gz'
        full_tar_input_name = os.path.join(net_dir, tar_input_name)
        if not os.path.exists(full_tar_input_name):
            continue
        tar = tarfile.open(full_tar_input_name, 'r:gz')
        tar.extractall(net_dir)
        tar.close()

        input_name = net_name + '.txt'
        full_input_name = os.path.join(net_dir, input_name)

        output_dir = os.path.join(net_dir, attack)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_name = 'oi_list.txt'
        full_output_name = os.path.join(output_dir, output_name)

        if os.path.isfile(full_output_name) and overwrite:
            os.remove(full_output_name)

        ## Read network file
        g = ig.Graph().Read_Edgelist(full_input_name, directed=False)

        ## Remove network file
        os.remove(full_input_name)

        ## Perform the attack
        get_index_list(g, attack, full_output_name)
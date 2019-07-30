import os
import sys
import numpy as np
import igraph as ig

from dismantling import get_index_list
from auxiliary import get_base_network_name

supported_attacks = ['Ran', 'Deg', 'Btw', 'DegU', 'BtwU']

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])

if 'overwrite' in sys.argv:
    overwrite = True
else:
    overwrite = False

dir_name = os.path.join('../networks', net_type)
seeds = range(min_seed, max_seed)

attacks = []
for attack in supported_attacks:
    if attack in sys.argv:
        attacks.append(attack)

base_net_name = get_base_network_name(net_type, size, param)
base_net_dir = os.path.join(dir_name, base_net_name)

for attack in attacks:
    print(attack)
    for seed in seeds:
        net_name = base_net_name + '_{:05d}'.format(seed)
        net_dir = os.path.join(base_net_dir, net_name)
        #input_name = net_name + '_gcc.txt'
        input_name = net_name + '.txt'
        full_input_name = os.path.join(net_dir, input_name)

        output_dir = os.path.join(net_dir, attack)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_name = 'oi_list.txt'
        full_output_name = os.path.join(output_dir, output_name)

        print(input_name)
        if os.path.isfile(full_output_name) and overwrite:
            os.remove(full_output_name)


        g = ig.Graph().Read_Edgelist(full_input_name, directed=False)    
        get_index_list(g, attack, full_output_name)
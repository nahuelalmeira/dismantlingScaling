import os
import sys
import pickle
import pathlib
import numpy as np
import igraph as ig

from auxiliary import get_base_network_name

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])

if 'overwrite' in sys.argv:
    overwrite = True
else:
    overwrite = False

seeds = range(min_seed, max_seed)
dir_name = os.path.join('../networks', net_type)
base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)


for seed in seeds:
    
    output_name = base_net_name_size + '_{:05d}.txt'.format(seed) 
    net_dir_name = os.path.join(dir_name, base_net_name, base_net_name_size, output_name[:-4])
    pathlib.Path(net_dir_name).mkdir(parents=True, exist_ok=True)
    full_name = os.path.join(net_dir_name, output_name)

    if not overwrite:
        if os.path.isfile(full_name):
            continue

    print(output_name)
    if net_type == 'ER':
        N = int(size)
        k = float(param)
        p = k/N
        G = ig.Graph().Erdos_Renyi(N, p)
    elif net_type == 'RR':
        N = int(size)
        k = int(param)
        G = ig.Graph().K_Regular(N, k)
    elif net_type == 'BA':
        N = int(size)
        m = int(param)
        G = ig.Graph().Barabasi(N, m)

    G.write_edgelist(full_name)
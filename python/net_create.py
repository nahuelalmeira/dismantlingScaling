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

base_net_name = get_base_network_name(net_type, size, param)

for seed in seeds:
   
    output_name = base_net_name + '_{:05d}.txt'.format(seed) 
    net_dir_name = os.path.join(dir_name, base_net_name, output_name[:-4])
    pathlib.Path(net_dir_name).mkdir(parents=True, exist_ok=True)
    full_name = os.path.join(net_dir_name, output_name)

    if not overwrite:
        if os.path.isfile(full_name):
            continue

    print(output_name)
    if net_type == 'ER':
        N = int(size)
        p = float(param)
        G = ig.Graph().Erdos_Renyi(N, p)
    elif net_type == 'BA':
        N = int(size)
        m = int(param)
        G = ig.Graph().Barabasi(N, m)  
    elif net_type == 'Lattice':
        L = int(size)
        f = float(param)
        G = ig.Graph().Lattice([L, L], nei=1, circular=False)
        M = G.ecount()
        edges_to_remove = np.random.choice(G.es(), int(f*M), replace=False)
        indices_to_remove = [e.index for e in edges_to_remove]
        G.delete_edges(indices_to_remove)

    G.write_edgelist(full_name) 
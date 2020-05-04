import os
import sys
import pathlib
import tarfile
import numpy as np
import igraph as ig

from auxiliary import get_base_network_name

from planar import create_proximity_graph

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])
compress = True
if 'compress_false' in sys.argv:
    compress = False

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

    tar_file_name = base_net_name_size + '_{:05d}.tar.gz'.format(seed)
    full_tar_file_name = os.path.join(net_dir_name, tar_file_name)

    if net_type == 'DT':
        position_file_name = 'position.txt'
        full_position_file_name = os.path.join(net_dir_name, position_file_name)

    if not overwrite:
        if os.path.isfile(full_name) or os.path.isfile(full_tar_file_name):
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
    elif net_type == 'Lattice':
        L = int(size)
        G = ig.Graph().Lattice(dim=[L, L], circular=False)
    elif net_type == 'Ld3':
        L = int(size)
        G = ig.Graph().Lattice(dim=[L, L, L], circular=False)
    elif net_type == 'MR':
        N = int(size)
        if param == 'rMST':
            G = create_proximity_graph(net_type, N=N, random_seed=seed)
        else:
            r = float(param)
            G = create_proximity_graph(net_type, N=N, r=r, random_seed=seed)
    elif net_type in ['DT', 'GG', 'RN']:
        N = int(size)
        G = create_proximity_graph(net_type, N=N, random_seed=seed)

    G.write_edgelist(full_name)
    if net_type == 'DT':
        points = G.vs['position']
        np.savetxt(full_position_file_name, points)

    if compress:

        ## Compress network file
        tar = tarfile.open(full_tar_file_name, 'w:gz')
        tar.add(full_name, arcname=output_name)
        tar.close()

        ## Remove network file
        os.remove(full_name)

        if net_type == 'DT':
            ## Compress network file
            tar = tarfile.open(os.path.join(net_dir_name, 'position.tar.gz'), 'w:gz')
            tar.add(full_position_file_name, arcname='position.txt')
            tar.close()

            ## Remove network file
            os.remove(full_position_file_name)
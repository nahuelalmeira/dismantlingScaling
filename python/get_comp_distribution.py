import igraph as ig
import numpy as np
import os
import sys
import tarfile
from auxiliary import get_base_network_name

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
str_f = sys.argv[4]
attack = sys.argv[5]
max_seed = int(sys.argv[6])

if net_type in ['ER', 'RR', 'BA', 'DT', 'GG', 'RN']:
    N = int(size)

overwrite = False
if 'overwrite' in sys.argv:
    overwrite = True

verbose = False
if 'verbose' in sys.argv:
    verbose = True

dir_name = os.path.join('../networks', net_type)
base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
base_net_dir = os.path.join(dir_name, base_net_name, base_net_name_size)

seeds = range(max_seed)

seed_file = os.path.join(dir_name, base_net_name, base_net_name_size, 
                         'comp_sizes_{}_f{}_seeds.txt'.format(attack, str_f))

if os.path.isfile(seed_file):
    if overwrite:
        os.remove(seed_file)
        past_seeds = []
    else:
        print('Past seeds will be considered')
        past_seeds = np.loadtxt(seed_file, dtype=int)
else:
    past_seeds = np.array([])

components_file = os.path.join(dir_name, base_net_name, base_net_name_size, 'comp_sizes_{}_f{}.txt'.format(attack, str_f))
if os.path.isfile(components_file):
    if overwrite:
        os.remove(components_file)

new_seeds = []
for seed in seeds:

    if seed in past_seeds:
        continue

    net_name = base_net_name_size + '_{:05d}'.format(seed)
    print(net_name)
    net_dir = os.path.join(base_net_dir, net_name)

    ## Extract network file
    tar_input_name = net_name + '.tar.gz'
    full_tar_input_name = os.path.join(net_dir, tar_input_name)
    if not os.path.exists(full_tar_input_name):
        print('TAR file do not exist:', tar_input_name)
        continue
    tar = tarfile.open(full_tar_input_name, 'r:gz')
    tar.extractall(net_dir)
    tar.close()
    
    input_name = net_name + '.txt'   
    full_input_name = os.path.join(net_dir, input_name)

    data_dir = os.path.join(net_dir, attack)
    oi_file = os.path.join(data_dir, 'oi_list.txt')
    if not os.path.isfile(oi_file):
        print("FILE " + oi_file + " NOT FOUND")
        continue
    try:
        oi_values = np.loadtxt(oi_file, dtype=int)
    except:
        print('Cannot read file from seed', seed)
        continue
    
    new_seeds.append(seed)

    g = ig.Graph().Read_Edgelist(full_input_name, directed=False)   

    ## Remove network file
    os.remove(full_input_name)     

    if not g.is_simple():
        print('Network "' + net_name + '" will be considered as simple.')
        g.simplify()
        
    if g.is_directed():
        print('Network "' + net_name + '" will be considered as undirected.')
        g.to_undirected()

    N0 = g.vcount()
    g.vs['original_index'] = range(N0)
    
    f = float(str_f)

    try:
        len(oi_values)
    except:
        continue

    if len(oi_values) < int(f*N):
        continue
    oi_values = oi_values[:int(f*N)]
    g.delete_vertices(oi_values)

    components = g.components(mode='WEAK')
    Ngcc = components.giant().vcount()
    comp_sizes = [len(c) for c in components]
    comp_sizes.remove(Ngcc)
    with open(components_file, 'a') as c_file:
        for c_size in comp_sizes:
            c_file.write('{:d}\n'.format(c_size))

new_seeds = np.array(new_seeds, dtype=int)
all_seeds = np.sort(np.concatenate((past_seeds, new_seeds)))
if len(all_seeds):
    np.savetxt(seed_file, all_seeds, fmt='%d')
else:
    print('No new seeds')
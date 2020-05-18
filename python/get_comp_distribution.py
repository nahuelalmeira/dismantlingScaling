import igraph as ig
import numpy as np
import os
import sys
import tarfile
from auxiliary import get_base_network_name, read_data_file

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

include_gcc = False
if 'gcc' in sys.argv:
    include_gcc = True

dir_name = os.path.join('../networks', net_type)
base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
base_net_dir = os.path.join(dir_name, base_net_name, base_net_name_size)

seeds = range(max_seed)

if include_gcc:
    name = 'comp_sizes_gcc_{}_f{}_seeds.txt'.format(attack, str_f)
else:
    name = 'comp_sizes_{}_f{}_seeds.txt'.format(attack, str_f)

seed_file = os.path.join(dir_name, base_net_name, base_net_name_size, name)

if os.path.isfile(seed_file):
    if overwrite:
        os.remove(seed_file)
        past_seeds = []
    else:
        print('Past seeds will be considered')
        past_seeds = np.loadtxt(seed_file, dtype=int)
else:
    past_seeds = np.array([])

if include_gcc:
    name = 'comp_sizes_gcc_{}_f{}.txt'.format(attack, str_f)
else:
    name = 'comp_sizes_{}_f{}.txt'.format(attack, str_f)

components_file = os.path.join(dir_name, base_net_name, base_net_name_size, name)
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
    attack_dir = os.path.join(net_dir, attack)

    try:
        g = read_data_file(net_dir, net_name, reader='igraph')
        oi_values = read_data_file(attack_dir, 'oi_list', reader='numpyInt')
    except FileNotFoundError:
        print('File could not be read')
        continue

    new_seeds.append(seed)

    if not g.is_simple():
        print('Network "' + net_name + '" will be considered as simple.')
        g.simplify()

    if g.is_directed():
        print('Network "' + net_name + '" will be considered as undirected.')
        g.to_undirected()

    N0 = g.vcount()
    g.vs['original_index'] = range(N0)

    f = float(str_f)

    if len(oi_values) and len(oi_values) < int(f*N):
        print('Not enough oi_values')
        continue

    oi_values = oi_values[:int(f*N)]
    g.delete_vertices(oi_values)

    components = g.components(mode='WEAK')
    Ngcc = components.giant().vcount()
    comp_sizes = [len(c) for c in components]

    if not include_gcc: ## Do not count GCC
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
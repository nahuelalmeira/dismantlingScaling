import os
import sys
import tarfile
import numpy as np
import igraph as ig

from auxiliary import get_base_network_name, supported_attacks, get_property_file_name, simple_props
from auxiliary import get_edge_weights

def compute_properties(G, oi_values, properties):

    prop_values = {}
    for prop in properties:
        prop_values[prop] = []

    for i, oi in enumerate(oi_values):
        g = G.copy()
        g.delete_vertices(oi_values[:i])
        if 'Ngcc' in properties:
            components = g.components(mode='weak')
            gcc = components.giant()
            N = gcc.vcount()
            M = gcc.ecount()
            Nc = len(components)
            Gamma = M - N + Nc
            prop_values['Ngcc'].append(N)
            #prop_values['Gamma'].append(Gamma)

        if 'C' in properties:
            C = g.transitivity_undirected(mode='zero')
            prop_values['C'].append(C)

        if 'Cws' in properties:
            Cws = g.transitivity_avglocal_undirected(mode='zero')
            prop_values['Cws'].append(Cws)

        if 'r' in properties:
            r = g.assortativity_degree(directed=False)
            prop_values['r'].append(r)

        if 'D' in properties:
            D = g.diameter(directed=False)
            prop_values['D'].append(D)

        if 'meanl' in properties:
            meanl = g.average_path_length(directed=False)
            prop_values['meanl'].append(meanl)

        if 'meanlw' in properties:
            meanlw = np.mean(g.shortest_paths(weights='weight', mode=ig.ALL))
            prop_values['meanlw'].append(meanlw)

        if 'meank' in properties:
            meank = np.mean(g.degree())
            prop_values['meank'].append(meank)

    return prop_values

def save_properties(prop_values):
    for prop, values in prop_values.items():
        output_file = get_property_file_name(prop, output_dir)
        if prop in simple_props:
            np.savetxt(output_file, values)

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])
properties = simple_props

if net_type in ['ER', 'RR', 'BA', 'MR', 'DT']:
    N = int(size)

overwrite = False
if 'overwrite' in sys.argv:
    overwrite = True

verbose = False
if 'verbose' in sys.argv:
    verbose = True

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

dir_name = os.path.join('../networks', net_type)
base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
base_net_dir = os.path.join(dir_name, base_net_name, base_net_name_size)

for attack in attacks:
    print(attack)
    prop_dfs = {}
    for seed in range(min_seed, max_seed):
        net_name = base_net_name_size + '_{:05d}'.format(seed)
        net_dir = os.path.join(base_net_dir, net_name)
        print(net_name, end='\t')

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
        oi_input_name = 'oi_list.txt'
        full_oi_input_name = os.path.join(output_dir, oi_input_name)
        if os.path.isfile(full_oi_input_name):
            if os.path.getsize(full_oi_input_name) == 0:
                os.remove(full_oi_input_name)
                continue
            num_lines = sum(1 for line in open(full_oi_input_name))
            if num_lines < N:
                print('ERROR in seed ', seed, ': File too short. Num lines', num_lines)
                continue
        else:
            continue

        oi_values = np.loadtxt(full_oi_input_name, dtype='int', comments='\x00')
        ## Read network file
        g = ig.Graph().Read_Edgelist(full_input_name, directed=False)

        if 'meanlw' in properties:
            g.es['weight'] = get_edge_weights(g, net_type, size, param, seed)

        ## Remove network file
        os.remove(full_input_name)

        if not overwrite:
            remove_props = []
            for prop in properties:
                prop_output_file = get_property_file_name(prop, output_dir)
                if os.path.isfile(prop_output_file):
                    num_lines = sum(1 for line in open(prop_output_file))
                    if num_lines < N:
                        os.remove(prop_output_file)
                    else:
                        remove_props.append(prop)
            for prop in remove_props:
                properties.remove(prop)

        ## Compute properties
        print(properties)
        prop_values = compute_properties(g, oi_values, properties)

        ## Write properties in files
        save_properties(prop_values)
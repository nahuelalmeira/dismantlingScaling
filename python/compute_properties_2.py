import os
import sys
import json
import pickle
import numpy as np
import igraph as ig

from auxiliary import get_base_network_name, supported_attacks, get_property_file_name, simple_props
from auxiliary import get_edge_weights, get_number_of_nodes, read_data_file

def get_prop(g, prop):

    if prop == 'C':
        return g.transitivity_undirected(mode='zero')

    if prop == 'Cws':
        return g.transitivity_avglocal_undirected(mode='zero')

    if prop == 'r':
        return g.assortativity_degree(directed=False)

    if prop == 'D':
        return g.diameter(directed=False)

    if prop == 'meanl':
        return g.average_path_length(directed=False)

    if prop == 'meanlw':
        return np.mean(g.shortest_paths(weights='weight', mode=ig.ALL))

    if prop == 'meank':
        return np.mean(g.degree())

    if prop == 'maxk':
        return max(g.degree())

def compute_props_step(g, properties, data_dict, step):

    for prop in properties:
        if step in data_dict[prop]:
            continue
        else:
            value = get_prop(g, prop)
            data_dict[prop][i] = value

    return

def load_data_json(properties_file):
    if os.path.isfile(properties_file) and not overwrite:
        with open(properties_file, 'r') as f:
            data_dict = json.load(f)
    else:
        data_dict = {}

    return data_dict

def load_data(properties_file):
    if os.path.isfile(properties_file) and not overwrite:
        with open(properties_file, 'rb') as f:
            data_dict = pickle.load(f)
    else:
        data_dict = {}

    return data_dict


def save_properties_json(data_dict, properties_file):
    #if verbose:
    #    print('Saving data to file:', properties_file)
    with open(properties_file, 'w') as f:
        json.dump(data_dict, f, sort_keys=True, indent=4)

def save_properties(data_dict, properties_file):
    #if verbose:
    #    print('Saving data to file:', properties_file)
    with open(properties_file, 'wb') as f:
        pickle.dump(data_dict, f)


net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])
nsteps = int(sys.argv[6])

supported_props = [
    'C',
    'Cws',
    'r',
    'meank',
    'maxk',
    'D',
    'meanl',
    'meanlw'
]

properties = []
for prop in supported_props:
    if prop in sys.argv:
        properties.append(prop)

N = get_number_of_nodes(net_type, size)

overwrite = False
if 'overwrite' in sys.argv:
    overwrite = True

verbose = False
if 'verbose' in sys.argv:
    verbose = True

steps = [int(N*i/nsteps) for i in range(nsteps)]

attacks = []
for attack in supported_attacks:
    if attack in sys.argv:
        attacks.append(attack)

print('------- Params -------')
print('net_type =', net_type)
print('param    =', param)
print('min_seed =', min_seed)
print('max_seed =', max_seed)
print('Properties:', properties)
print('----------------------', end='\n\n')

dir_name = os.path.join('../networks', net_type)
base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
base_net_dir = os.path.join(dir_name, base_net_name, base_net_name_size)

for attack in attacks:
    print(attack)
    for seed in range(min_seed, max_seed):
        net_name = base_net_name_size + '_{:05d}'.format(seed)
        net_dir = os.path.join(base_net_dir, net_name)
        attack_dir = os.path.join(net_dir, attack)
        if verbose:
            print(net_name)

        properties_file = os.path.join(attack_dir, 'properties.pickle')
        data_dict = load_data(properties_file)

        for prop in properties:
            if prop not in data_dict:
                data_dict[prop] = {}

        g = read_data_file(net_dir, net_name, reader='igraph')
        if 'meanlw' in properties:
            g.es['weight'] = get_edge_weights(g, net_type, size, param, seed)

        try:
            oi_values = read_data_file(attack_dir, 'oi_list', reader='numpyInt')
        except FileNotFoundError:
            print('Inexisting file:', seed)
            continue

        if oi_values.size < N:
            print('ERROR in seed ', seed, ': File too short. Num lines', oi_values.size)
            continue

        g.vs['oi'] = range(N)

        for i in range(N):

            oi = int(oi_values[i])
            idx = g.vs['oi'].index(oi)

            if i in steps:
                compute_props_step(g, properties, data_dict, i)

            if i == steps[-1]:
                break

            g.vs[idx].delete()

        ## Write properties in files
        save_properties(data_dict, properties_file)
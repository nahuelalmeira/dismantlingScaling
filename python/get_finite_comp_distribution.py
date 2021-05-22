import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from auxiliary import get_base_network_name, read_data_file


def parse_args():
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description='Perform centrality-based attack on a given network'
    )
    parser.add_argument(
        'net_type', type=str, help='Network type'
    )
    parser.add_argument(
        'size', type=int, help='the path to list'
    )
    parser.add_argument(
        'param', type=str,
        help='Parameter characterizing the network (e.g., its mean degree)'
    )
    parser.add_argument(
        'str_f', type=str,
        help='Fraction of nodes removed where components are to be computed'
    )
    parser.add_argument(
        'attack', type=str, help='Attack employed'
    )
    parser.add_argument(
        'nseeds', type=int, help='Number of random seeds'
    )
    parser.add_argument(
        '--dropLargest', type=int, default=1, 
        help='Discard the n largest components'
    )
    parser.add_argument(
        '--overwrite', action='store_true', help='Overwrite procedure'
    )
    parser.add_argument(
        '--log', type=str, default='warning',
        choices=['debug', 'info', 'warning', 'error', 'exception', 'critical']
    )
    return parser.parse_args()

args = parse_args() 

net_type        = args.net_type
size            = args.size
param           = args.param
str_f           = args.str_f
attack          = args.attack
nseeds          = args.nseeds
overwrite       = args.overwrite
logging_level   = args.log.upper()
dropLargest     = args.dropLargest

logging.basicConfig(
    format='%(levelname)s: %(asctime)s %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=getattr(logging, logging_level)
)


dir_name = Path('../networks') / net_type
base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
base_net_dir = dir_name / base_net_name / base_net_name_size

seeds = range(nseeds)

base_output_name = f'comp_sizes_{attack}_f{str_f}_drop{dropLargest}'
components_file = base_net_dir / (base_output_name + '.txt')
seed_file = base_net_dir / (base_output_name + '_seeds.txt')

if seed_file.is_file():
    if overwrite:
        seed_file.unlink()
        past_seeds = []
    else:
        print('Past seeds will be considered')
        past_seeds = np.loadtxt(seed_file, dtype=int)
else:
    past_seeds = np.array([])

if components_file.is_file() and overwrite:
    components_file.unlink()

new_seeds = []
for seed in seeds:

    if seed in past_seeds:
        continue

    net_name = base_net_name_size + '_{:05d}'.format(seed)
    logging.debug(net_name)

    net_dir = base_net_dir / net_name
    attack_dir = net_dir / attack

    try:
        g = read_data_file(str(net_dir), net_name, reader='igraph')
        oi_values = read_data_file(str(attack_dir), 'oi_list', reader='numpyInt')
    except FileNotFoundError:
        logging.warning('File could not be read')
        continue

    if not g.is_simple():
        logging.info(f'Network "{net_name}" will be considered as simple.')
        g.simplify()

    if g.is_directed():
        logging.info(f'Network "{net_name}" will be considered as undirected.')
        g.to_undirected()

    N = g.vcount()
    g.vs['original_index'] = range(N)

    f = float(str_f)

    if oi_values.size < int(f*N):
        logging.warning('Not enough oi_values')
        continue

    oi_values = oi_values[:int(f*N)]
    g.delete_vertices(oi_values)

    components = g.components(mode='WEAK')
    comp_sizes = np.sort([len(c) for c in components])
    if dropLargest:
        comp_sizes = comp_sizes[:-dropLargest]

    with open(components_file, 'a') as c_file:
        for c_size in comp_sizes:
            c_file.write('{:d}\n'.format(c_size))

    new_seeds.append(seed)

new_seeds = np.array(new_seeds, dtype=int)
all_seeds = np.sort(np.concatenate((past_seeds, new_seeds)))
if len(all_seeds):
    np.savetxt(seed_file, all_seeds, fmt='%d')
else:
    print('No new seeds')
import os
import argparse
import logging
from pathlib import Path
from dismantling import get_index_list
from auxiliary import (
    get_base_network_name, 
    get_edge_weights, 
    read_data_file
)

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
        'min_seed', type=int, help='Minimum random seed'
    )
    parser.add_argument(
        'max_seed', type=int, help='Maximum random seed'
    )
    parser.add_argument(
        '--attacks', nargs='+', type=str, default=[],
        help='Attacks to be performed.'
    )
    parser.add_argument(
        '--package', type=str, default='igraph', 
        choices=['igraph', 'networkit'],
        help='Python package to be used'
    )
    parser.add_argument(
        '--overwrite', action='store_true', help='Overwrite procedure'
    )
    parser.add_argument(
        '--log', type=str, default='warning',
        choices=['debug', 'info', 'warning', 'error', 'exception', 'critical']
    )
    parser.add_argument(
        '--saveCentrality', action='store_true', 
        help='Save initial centrality values'
    )
    return parser.parse_args()

args = parse_args() 

net_type        = args.net_type
size            = args.size
param           = args.param
min_seed        = args.min_seed
max_seed        = args.max_seed
package         = args.package
attacks         = args.attacks
overwrite       = args.overwrite
save_centrality = args.saveCentrality
logging_level   = args.log.upper()

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, logging_level))

python_file_dir_name = os.path.dirname(__file__)
dir_name = os.path.join(python_file_dir_name, '..', 'networks', net_type)
seeds = range(min_seed, max_seed)

base_net_name, base_net_name_size = get_base_network_name(
    net_type, size, param
)
base_net_dir = os.path.join(dir_name, base_net_name, base_net_name_size)

for attack in attacks:

    logging.info(attack)

    for seed in seeds:
        net_name = base_net_name_size + '_{:05d}'.format(seed)
        net_dir = os.path.join(base_net_dir, net_name)

        logger.info(net_name)

        output_dir = os.path.join(net_dir, attack)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_name = 'oi_list.txt'
        full_output_name = os.path.join(output_dir, output_name)

        full_c_output_name = None
        if save_centrality:
            c_output_name = 'initial_centrality.txt'
            full_c_output_name = os.path.join(output_dir, c_output_name)

        if os.path.isfile(full_output_name) and overwrite:
            os.remove(full_output_name)
        if save_centrality and os.path.isfile(full_c_output_name) and overwrite:
            os.remove(full_c_output_name)

        g = read_data_file(net_dir, net_name, reader=package)

        if 'BtwWU' in attack:
            g.es['weight'] = get_edge_weights(g, net_type, size, param, seed)

        ## Perform the attack
        get_index_list(
            g, attack, full_output_name,
            save_centrality=save_centrality,
            out_centrality=full_c_output_name,
            random_state=seed
        )
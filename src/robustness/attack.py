import os
import argparse
import logging
import time
import igraph as ig

from robustness import NETWORKS_DIR
from robustness.dismantling import get_index_list
from robustness.auxiliary import (
    get_base_network_name, 
    get_edge_weights, 
    read_data_file
)

start = time.time()

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
    parser.add_argument(
        '--percolate', action='store_true', 
        help='Perform Newman-Ziff algorithm for percolation'
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
percolate_true  = args.percolate

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, logging_level))

dir_name = NETWORKS_DIR / net_type
seeds = range(min_seed, max_seed)

base_net_name, base_net_name_size = get_base_network_name(
    net_type, size, param
)
base_net_dir = dir_name / base_net_name / base_net_name_size

for attack in attacks:

    logging.info(attack)
    
    
    for seed in seeds:
        net_name = base_net_name_size + '_{:05d}'.format(seed)
        net_dir = base_net_dir / net_name

        logger.info(net_name)

        output_dir = net_dir / attack
        if not output_dir.is_dir():
            os.mkdir(output_dir)
        output_name = 'oi_list.txt'
        full_output_name = output_dir / output_name

        full_c_output_name = None
        if save_centrality:
            c_output_name = 'initial_centrality.txt'
            full_c_output_name = output_dir / c_output_name

        if full_output_name.is_file() and overwrite:
            full_output_name.unlink()
        if save_centrality and full_c_output_name.is_file and overwrite:
            full_c_output_name.unlink()

        try:
            g = read_data_file(net_dir, net_name, reader=package)
        except ig._igraph.InternalError as e:
            ## TODO: See why this error sometimes arises
            logger.exception(e)
            continue


        if 'BtwWU' in attack:
            g.es['weight'] = get_edge_weights(g, net_type, size, param, seed)

        ## Perform the attack
        get_index_list(
            g, attack, full_output_name,
            save_centrality=save_centrality,
            out_centrality=full_c_output_name,
            random_state=seed
        )

        
        if percolate_true:
            logger.debug('pre percolate')
            import numpy as np
            from robustness.percolation import percolate_fast
            from robustness.auxiliary import edgelist_to_adjlist

            attack_dir = f'{str(net_dir)}/{attack}'
            comp_data_file = attack_dir + '/comp_data_fast.txt'
            if os.path.exists(comp_data_file):
                from datetime import datetime
                reference_date = datetime(2021, 5, 17) 
                modification_time = datetime.fromtimestamp(
                    os.path.getctime(comp_data_file)
                )
                if modification_time < reference_date:
                    os.remove(comp_data_file)
                elif not overwrite:
                    continue

            time1 = time.time()
            edgelist = read_data_file(net_dir, net_name, reader='numpyInt')
            time2 = time.time()
            logger.debug(f'load edgelist {time2-time1}')
            time1 = time2

            adjlist = edgelist_to_adjlist(edgelist, size)
            time2 = time.time()
            logger.debug(f'to adjlist {time2-time1}')
            time1 = time2

            time2 = time.time()
            order = read_data_file(
                attack_dir, 'oi_list', reader='numpyInt'
            )[::-1]
            logger.debug(f'load order {time2-time1}')
            time1 = time2

            ## If there are isolated nodes
            if len(order) < size:
                order = np.append(order, range(len(order), size))

            logger.debug(f'percolate {time2-time1}')
            perc_data = percolate_fast(adjlist, order)
            comp_data = np.zeros((size, 3))
            comp_data[:,0] = perc_data['N1']
            comp_data[:,1] = np.NaN
            comp_data[:,2] = perc_data['meanS']
            np.savetxt(comp_data_file, comp_data, fmt='%f %f %f')
            logger.debug('post percolate')
        logger.debug('finish seed')

finish = time.time()
logger.info(f'Elapsed time: {finish-start:.6f}')
import logging
import argparse
import numpy as np

from robustness import NETWORKS_DIR
from robustness.auxiliary import (
    get_base_network_name, 
    read_data_file, 
    get_number_of_nodes
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
        '--overwrite', action='store_true', help='Overwrite procedure'
    )
    parser.add_argument(
        '--log', type=str, default='info',
        choices=['debug', 'info', 'warning', 'error', 'exception', 'critical']
    )
    parser.add_argument(
        '--chiDelta', action='store_true', 
        help='Compute diff in Sgcc (slow)'
    )
    parser.add_argument(
        '--n_deltas', type=int, help='number of gaps to be stored',
        default=10
    )
    return parser.parse_args()

args = parse_args() 

net_type        = args.net_type
size            = args.size
param           = args.param
min_seed        = args.min_seed
max_seed        = args.max_seed
attacks         = args.attacks
overwrite       = args.overwrite
logging_level   = args.log.upper()
chiDelta        = args.chiDelta
n_deltas        = args.n_deltas

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, logging_level))

N = get_number_of_nodes(net_type, size)

print('------- Params -------')
print('net_type =', net_type)
print('param    =', param)
print('min_seed =', min_seed)
print('max_seed =', max_seed)
print('----------------------', end='\n\n')

dir_name = NETWORKS_DIR / net_type
base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
base_network_dir_name = dir_name / base_net_name / base_net_name_size

for attack in attacks:
    logger.info(attack)
    n_seeds = max_seed - min_seed
    output_file_name = (
        base_network_dir_name /
        f'{n_deltas}_delta_values_{attack}_nSeeds{n_seeds}.txt'
    )
    if not overwrite:
        if output_file_name.is_file():
            continue

    delta_max_values = []

    valid_its = 0
    for seed in range(min_seed, max_seed):

        network = base_net_name_size + '_{:05d}'.format(seed)
        attack_dir_name = base_network_dir_name / network / attack

        ## Read data
        try: 
            aux = read_data_file(
                str(attack_dir_name), 'comp_data', reader='numpy'
            )
        except FileNotFoundError:
            continue
        except ValueError:
            logger.error(seed)
            raise

        logger.debug(seed)

        len_aux = aux.shape[0]
        if len_aux > N:
            logger.error(
                f'Seed {seed}. Len of array is greater than network size'
            )
            continue
        if len_aux < 0.9*N:
            logger.error(f'Seed {seed}. Len of array is too short')
            continue

        valid_its += 1

        Ngcc_values = aux[:,0][::-1]
        delta_values = np.abs(np.diff(Ngcc_values))

        ## Take the indices for the 'n_deltas' largest gaps
        gap_positions = np.argsort(delta_values)[-n_deltas:]

        ## Sort in descending position
        gap_positions = gap_positions[::-1]

        data = []
        for pos in gap_positions:
            data += [pos/N, delta_values[pos]/N]

        delta_max_values.append(data)

    np.savetxt(output_file_name, delta_max_values)

    logger.info(f'Correct seeds = {valid_its}')

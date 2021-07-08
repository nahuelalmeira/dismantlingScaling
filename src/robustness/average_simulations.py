import os
import logging
import argparse
import numpy as np
import pandas as pd

from robustness import NETWORKS_DIR
from robustness.auxiliary import get_base_network_name, read_data_file
from robustness.auxiliary import get_number_of_nodes

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
        '--log', type=str, default='warning',
        choices=['debug', 'info', 'warning', 'error', 'exception', 'critical']
    )
    parser.add_argument(
        '--chiDelta', action='store_true', 
        help='Compute diff in Sgcc (slow)'
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

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, logging_level))

N = get_number_of_nodes(net_type, size)

print('------- Params -------')
print('net_type =', net_type)
print('param    =', param)
print('size     =', size)
print('min_seed =', min_seed)
print('max_seed =', max_seed)
print('----------------------', end='\n\n')


dir_name = NETWORKS_DIR / net_type
base_net_name, base_net_name_size = get_base_network_name(
    net_type, size, param
)
base_net_dir = dir_name / base_net_name / base_net_name_size
comp_data_file = 'comp_data'

for attack in attacks:
    logger.info(attack)

    n_seeds = max_seed - min_seed
    csv_file_name = base_net_dir / f'{attack}_nSeeds{n_seeds:d}_cpp.csv'
    
    if not overwrite:
        if csv_file_name.is_file():
            continue

    Ngcc_values          = np.zeros(N)
    Ngcc_sqr_values      = np.zeros(N)
    Nsec_values          = np.zeros(N)
    meanS_values         = np.zeros(N)
    num_values           = np.zeros(N)
    denom_values         = np.zeros(N)
    avg_comp_size_values = np.zeros(N)
    meanS_pow2_values    = np.zeros(N)
    meanS_pow4_values    = np.zeros(N)
    chiDelta_values      = np.zeros(N)

    valid_its = 0
    for seed in range(min_seed, max_seed):
        network = f'{base_net_name_size}_{seed:05d}'
        attack_dir_name = base_net_dir / network / attack

        ## Read data
        try:
            aux = read_data_file(
                str(attack_dir_name), comp_data_file, 
                reader='pandas', sep=' ', header=None
            ).values
        except FileNotFoundError:
            continue
        except ValueError:
            logger.error(f'ValueError: {seed} {comp_data_file}')
            continue

        logger.info(seed)

        len_aux = aux.shape[0]
        if len_aux > N:
            logger.error(
                f' Seed {seed}. Len of array is greater than network size'
            )
            continue
        if len_aux < 0.9*N:
            logger.error(f'ERROR: Seed {seed}. Len of array is too short')
            continue

        if np.min(aux) < 0:
            logger.error(seed)

        valid_its += 1
        
        if len_aux < N:
            aux = np.append(
                aux[::-1], np.repeat([1]*aux.shape[1], (N-len_aux))
            )
        else:
            aux = aux[::-1]

        Ngcc_values_it = aux[:,0]
        Nsec_values_it = aux[:,1]
        meanS_values_it = aux[:,2]
        try:
            num_values_it = aux[:,3]
            denom_values_it = aux[:,4]
            avg_comp_size_values_it = aux[:,5]
        except IndexError as e:
            logger.exception(e)
            raise

        Ngcc_values += Ngcc_values_it
        Ngcc_sqr_values += Ngcc_values_it**2
        Nsec_values += Nsec_values_it
        meanS_values += meanS_values_it
        num_values += num_values_it
        denom_values += denom_values_it
        avg_comp_size_values += avg_comp_size_values_it

        meanS_pow2_values += meanS_values_it**2
        meanS_pow4_values += meanS_values_it**4

        if chiDelta:
            chiDelta_values_it = np.append(np.diff(Ngcc_values_it), 0)
            chiDelta_values += chiDelta_values_it

    varSgcc_values       = (Ngcc_sqr_values/valid_its - (Ngcc_values/valid_its)**2) / (N)
    Ngcc_values          = Ngcc_values / valid_its
    Sgcc_values          = Ngcc_values / N
    Nsec_values          = Nsec_values / valid_its
    meanS_values         = meanS_values / valid_its
    num_values           = num_values / valid_its
    denom_values         = denom_values / valid_its
    avg_comp_size_values = avg_comp_size_values / valid_its
    chiDelta_values      = chiDelta_values / valid_its
    meanS_pow2_values    = meanS_pow2_values / valid_its
    meanS_pow4_values    = meanS_pow4_values / valid_its

    d = {
        'f': np.arange(N)/N,
        'Sgcc': Sgcc_values,
        'varSgcc': varSgcc_values,
        'Nsec': Nsec_values,
        'meanS': meanS_values,
        'num': num_values,
        'denom': denom_values,
        'avg_comp_size': avg_comp_size_values,
        'chiDelta': chiDelta_values,
        'meanS_pow2': meanS_pow2_values,
        'meanS_pow4': meanS_pow4_values
    }
    df = pd.DataFrame(data=d)
    df.to_csv(csv_file_name)

    logger.info(f'Correct seeds = {valid_its}')

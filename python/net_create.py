import os
from pathlib import Path
import tarfile
import argparse
import logging
import numpy as np
import igraph as ig

from auxiliary import get_base_network_name
from planar import create_proximity_graph, get_r_from_meank

def create_network(
    net_type: str, 
    size: int, 
    param: str, 
    seed: int
) -> ig.Graph():
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
    elif net_type == 'PLattice':
        L = int(size)
        G = ig.Graph().Lattice(dim=[L, L], circular=True)
    elif net_type == 'Ld3':
        L = int(size)
        G = ig.Graph().Lattice(dim=[L, L, L], circular=False)
    elif net_type == 'MR':
        N = int(size)
        if 'k' in param:
            meank = float(param[1:])
            r = get_r_from_meank(meank, N)
            G = create_proximity_graph(net_type, N=N, r=r, random_seed=seed)
        elif param == 'rMST':
            G = create_proximity_graph(net_type, N=N, random_seed=seed)
        else:
            r = float(param[1:])
            G = create_proximity_graph(net_type, N=N, r=r, random_seed=seed)
    elif net_type in ['DT', 'GG', 'RN', 'PDT']:
        N = int(size)
        G = create_proximity_graph(net_type, N=N, random_seed=seed)
    return G

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
        '--overwrite', action='store_true', help='Overwrite procedure'
    )
    parser.add_argument(
        '--no-compress', action='store_true', help='Do not compress file'
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
min_seed        = args.min_seed
max_seed        = args.max_seed
overwrite       = args.overwrite
logging_level   = args.log.upper()
compress        = not args.no_compress

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, logging_level))
print(logger)

seeds = range(min_seed, max_seed)

python_file_dir_name = os.path.dirname(__file__)
dir_name = os.path.join(python_file_dir_name, '../networks', net_type)

base_net_name, base_net_name_size = get_base_network_name(
    net_type, size, param
)

for seed in seeds:

    seed_base_name = base_net_name_size + '_{:05d}'.format(seed)
    output_name = seed_base_name + '.txt'
    net_dir_name = (
        Path(dir_name) / base_net_name / base_net_name_size / output_name[:-4]
    )
    net_dir_name.mkdir(parents=True, exist_ok=True)
    full_name = net_dir_name / output_name

    tar_file_name = seed_base_name + '.tar.gz'
    full_tar_file_name = net_dir_name / tar_file_name

    if net_type == 'DT':
        full_position_file_name = net_dir_name / 'position.txt'

    if not overwrite:
        if full_name.is_file() or full_tar_file_name.is_file():
            continue

    logger.info(output_name)
    G = create_network(net_type, size, param, seed)

    G.write_edgelist(str(full_name))
    if net_type == 'DT':
        points = G.vs['position']
        np.savetxt(full_position_file_name, points)

    if compress:

        ## Compress network file
        tar = tarfile.open(full_tar_file_name, 'w:gz')
        tar.add(full_name, arcname=output_name)
        tar.close()

        ## Remove network file
        full_name.unlink()

        if net_type == 'DT':
            ## Compress position file
            tar = tarfile.open(
                net_dir_name / 'position.tar.gz', 'w:gz'
            )
            tar.add(full_position_file_name, arcname='position.txt')
            tar.close()

            ## Remove network file
            full_position_file_name.unlink()
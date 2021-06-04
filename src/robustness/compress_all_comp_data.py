import tarfile
import pathlib
import logging
import argparse

from robustness import NETWORKS_DIR
from robustness.auxiliary import get_base_network_name

def parse_args():
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description='Perform centrality-based attack on a given network'
    )
    parser.add_argument(
        '--net_type', type=str, help='Network type', default='*'
    )
    parser.add_argument(
        '--size', type=str, help='Network size', default='*'
    )
    parser.add_argument(
        '--attack', type=str, help='Attack', default='*'
    )
    parser.add_argument(
        '--param', type=str,
        help='Parameter characterizing the network (e.g., its mean degree)',
        default='*'
    )
    parser.add_argument(
        '--log', type=str, default='info',
        choices=['debug', 'info', 'warning', 'error', 'exception', 'critical']
    )
    return parser.parse_args()

args = parse_args() 

net_type        = args.net_type
size            = args.size
param           = args.param
attack          = args.attack
logging_level   = args.log.upper()

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, logging_level))
files_compressed = 0

base_net_name = '*'
base_net_name_size = '*'
if net_type != '*' and size != '*' and param != '*':
    base_net_name, base_net_name_size = get_base_network_name(
        net_type, int(size), param
    )

files_compressed = 0

pattern = f'{net_type}/{base_net_name}/{base_net_name_size}/*/{attack}'
logger.info(f'Pattern: {pattern}')
for p in NETWORKS_DIR.glob(pattern):
    if not p.is_dir():
        continue

    for fast_file in ['comp_data_fast.txt', 'comp_data_fast.tar.gz']:
        full_fast_file = p / 'comp_data_fast.txt'
        if full_fast_file.is_file():
            full_fast_file.unlink()

    base_name = 'comp_data'
    full_file_name = p / f'{base_name}.txt'
    if not full_file_name.is_file():
        continue

    ## Compress comp_data file
    full_tar_input_name = p / f'{base_name}.tar.gz'
    tar = tarfile.open(full_tar_input_name, 'w:gz')
    tar.add(full_file_name, arcname= f'{base_name}.txt')
    tar.close()

    ## Remove comp_data file
    full_file_name.unlink()
    files_compressed += 1
    logger.info(full_file_name)
    #input()

logger.info(f'Files compressed: {files_compressed}')
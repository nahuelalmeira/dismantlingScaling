import os
import sys
import logging
import igraph as ig

from robustness.auxiliary import get_base_network_name

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])

overwrite = False
if 'overwrite' in sys.argv:
    overwrite = True

logger = logging.getLogger(__name__)
#logger.setLevel(getattr(logging, logging_level))

dir_name = os.path.join('../networks', net_type)

seeds = range(min_seed, max_seed)

base_net_name = get_base_network_name(net_type, size, param)

for seed in seeds:

    net_name = base_net_name + '_{:05d}'.format(seed)
    logger.info(net_name)

    net_dir_name = os.path.join(dir_name, base_net_name, net_name)
    input_name = net_name + '.txt'
    full_input_name = os.path.join(net_dir_name, input_name)
    if not os.path.isfile(full_input_name):
        continue

    output_name = net_name + '_gcc.txt'
    gcc_file = os.path.join(net_dir_name, output_name)
    if not overwrite:
        if os.path.isfile(gcc_file):
            continue

    g = ig.Graph().Read_Edgelist(full_input_name, directed=False)

    if not g.is_simple():
        logger.info(f'Network "{net_name}" will be considered as simple.')
        g.simplify()

    if g.is_directed():
        logger.info(f'Network "{net_name}" will be considered as undirected.')
        g.to_undirected()

    if not g.is_connected():
        logger.info(f'Only giant component of network "{net_name}" will be considered.')
        components = g.components(mode='weak')
        g = components.giant()

    g.write_edgelist(gcc_file)

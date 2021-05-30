import os
import sys
import logging
import numpy as np
from robustness.auxiliary import get_base_network_name
from robustness.planar import create_proximity_graph, get_r_from_meank

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
max_seed = int(sys.argv[4])

overwrite = False
if 'overwrite' in sys.argv:
    overwrite = True


logger = logging.getLogger(__name__)

python_file_dir_name = os.path.dirname(__file__)
dir_name = os.path.join(python_file_dir_name, '../networks', net_type)  
base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
base_net_dir = os.path.join(dir_name, base_net_name, base_net_name_size)

output_file = os.path.join(base_net_dir, 'diameters_N{}.txt'.format(size))

if os.path.isfile(output_file) and not overwrite:
    diameters = np.loadtxt(output_file, dtype='int')
    min_seed = diameters.size
else:
    min_seed = 0

if overwrite:
    f = open(output_file, 'w')
else:
    f = open(output_file, 'a+')

logger.debug('MIN_SEED', min_seed)
logger.debug('MAX_SEED', max_seed)

for seed in range(min_seed, max_seed):

    logger.debug(seed)

    if net_type == 'MR':
        N = int(size)
        if param == 'rMST':
            G = create_proximity_graph(net_type, N=N, random_seed=seed)
        elif 'meank' in sys.argv:
            meank = float(param)
            r = get_r_from_meank(meank, N)
            G = create_proximity_graph(net_type, N=N, r=r, random_seed=seed)
        else:
            r = float(param)
            G = create_proximity_graph(net_type, N=N, r=r, random_seed=seed)
    else:
        g = create_proximity_graph(net_type, N=size, random_seed=seed)
    d = g.diameter()

    f.write('{:d}\n'.format(d))
    f.flush()
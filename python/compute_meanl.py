import os
import sys
import numpy as np
from auxiliary import get_base_network_name
from planar import create_proximity_graph

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
max_seed = int(sys.argv[4])

overwrite = False
if 'overwrite' in sys.argv:
    overwrite = True

verbose = False
if 'verbose' in sys.argv:
    verbose = True

python_file_dir_name = os.path.dirname(__file__)
dir_name = os.path.join(python_file_dir_name, '../networks', net_type)

if net_type == 'MR':
    if 'meank' in sys.argv:
        base_net_name, base_net_name_size = get_base_network_name(net_type, size, param, meank=True)
    elif 'rMST' in sys.argv:
        base_net_name, base_net_name_size = get_base_network_name(net_type, size, param, rMST=True)
    else:
        base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
else:    
    base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
base_net_dir = os.path.join(dir_name, base_net_name, base_net_name_size)

output_file = os.path.join(base_net_dir, 'meanl_N{}.txt'.format(size))


if os.path.isfile(output_file) and not overwrite:
    meanls = np.loadtxt(output_file)
    min_seed = meanls.size
else:
    min_seed = 0

if overwrite:
    f = open(output_file, 'w')
else:
    f = open(output_file, 'a+')

if verbose:
    print('MIN_SEED', min_seed)
    print('MAX_SEED', max_seed)

for seed in range(min_seed, max_seed):
    if verbose:
        print(seed)
    g = create_proximity_graph(net_type, N=size, random_seed=seed)
    l = g.average_path_length(directed=False)

    f.write('{:.6f}\n'.format(l))
    f.flush()
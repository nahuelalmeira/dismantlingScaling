import os
import sys
import tarfile

from auxiliary import get_base_network_name

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])

python_file_dir_name = os.path.dirname(__file__)
dir_name = os.path.join(python_file_dir_name, '../networks', net_type)
seeds = range(min_seed, max_seed)

base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
base_net_dir = os.path.join(dir_name, base_net_name, base_net_name_size)

for seed in seeds:
    net_name = base_net_name_size + '_{:05d}'.format(seed)
    net_dir = os.path.join(base_net_dir, net_name)

    print(net_name)

    ## Extract network file
    tar_input_name = net_name + '.tar.gz'
    full_tar_input_name = os.path.join(net_dir, tar_input_name)
    if not os.path.exists(full_tar_input_name):
        continue
    tar = tarfile.open(full_tar_input_name, 'r:gz')
    tar.extractall(net_dir)
    tar.close()

    os.remove(full_tar_input_name)
import os
import sys
import tarfile

from auxiliary import get_base_network_name

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])

verbose = False
if 'verbose' in sys.argv:
    verbose = True

dir_name = os.path.join('../networks', net_type)
seeds = range(min_seed, max_seed)

base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)
base_net_dir = os.path.join(dir_name, base_net_name, base_net_name_size)

good_seeds = 0
for seed in seeds:
    net_name = base_net_name_size + '_{:05d}'.format(seed)
    net_dir = os.path.join(base_net_dir, net_name)

    input_name = net_name + '.txt'
    full_input_name = os.path.join(net_dir, input_name)

    if not os.path.exists(full_input_name):
        continue

    if verbose:
        print(net_name)

    ## Compress network file
    tar_input_name = net_name + '.tar.gz'
    full_tar_input_name = os.path.join(net_dir, tar_input_name)
    tar = tarfile.open(full_tar_input_name, 'w:gz')
    tar.add(full_input_name, arcname=input_name)
    tar.close()

    ## Remove network file
    os.remove(full_input_name)

    good_seeds += 1
print('Files compressed: ', good_seeds)



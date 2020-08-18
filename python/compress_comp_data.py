import os
import sys
import tarfile
from auxiliary import get_base_network_name, supported_attacks

net_type = sys.argv[1]
size = int(sys.argv[2])
param = sys.argv[3]
min_seed = int(sys.argv[4])
max_seed = int(sys.argv[5])

verbose = False
if 'verbose' in sys.argv:
    verbose = True

attacks = []
for attack in supported_attacks:
    if attack in sys.argv:
        attacks.append(attack)

python_file_dir_name = os.path.dirname(__file__)
dir_name = os.path.join(python_file_dir_name, '../networks', net_type)
base_net_name, base_net_name_size = get_base_network_name(net_type, size, param)

for attack in attacks:
    print(attack)
    good_seeds = 0
    for seed in range(min_seed, max_seed):

        network = base_net_name_size + '_{:05d}'.format(seed)
        attack_dir_name = os.path.join(dir_name, base_net_name, base_net_name_size, network, attack)

        full_file_name  = os.path.join(attack_dir_name, 'comp_data.txt')
        if not os.path.isfile(full_file_name):
            continue

        if verbose:
            print(seed)

        ## Compress network file
        tar_input_name = 'comp_data.tar.gz'
        full_tar_input_name = os.path.join(attack_dir_name, tar_input_name)
        tar = tarfile.open(full_tar_input_name, 'w:gz')
        tar.add(full_file_name, arcname='comp_data.txt')
        tar.close()

        ## Remove network file
        os.remove(full_file_name)

        good_seeds += 1
    print('Files compressed: ', good_seeds)
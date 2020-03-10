import sys

def get_base_network_name(net_type, size, param):
    if net_type == 'ER':
        N = int(size)
        k = float(param)
        base_net_name = 'ER_k{:.2f}'.format(k)
        base_net_name_size = base_net_name + '_N{}'.format(N)
    elif net_type == 'RR':
        N = int(size)
        k = int(param)
        base_net_name = 'RR_k{:02d}'.format(k)
        base_net_name_size = base_net_name + '_N{}'.format(N)   
    elif net_type == 'BA':
        N = int(size)
        m = int(param)
        base_net_name = 'BA_m{:02d}'.format(m)
        base_net_name_size = base_net_name + '_N{}'.format(N)               
    else:
        print('ERROR: net_type not supported', file=sys.stderr)
        base_net_name = ''
    
    return base_net_name, base_net_name_size

supported_attacks = [
    'Ran', 'Deg', 'DegU', 'Btw', 'BtwU', 'Eigenvector', 'EigenvectorU',
    'BtwU1nn', 'CIU'
]
supported_attacks += ['BtwU_cutoff{}'.format(l) for l in range(2, 100)]
import sys

def get_base_network_name(net_type, size, param):
    N = int(size)

    if net_type == 'ER':
        k = float(param)
        base_net_name = 'ER_k{:.2f}'.format(k)
    elif net_type == 'RR':
        k = int(param)
        base_net_name = 'RR_k{:02d}'.format(k)
    elif net_type == 'BA':
        m = int(param)
        base_net_name = 'BA_m{:02d}'.format(m)
    elif net_type == 'MR':
        base_net_name = 'MR_rMST'
    elif net_type == 'DT':
        base_net_name = 'DT_param'
    else:
        print('ERROR: net_type not supported', file=sys.stderr)
        base_net_name = ''

    base_net_name_size = base_net_name + '_N{}'.format(N)
    return base_net_name, base_net_name_size

supported_attacks = [
    'Ran', 'Deg', 'DegU', 'CIU', 'CIU2', 'Eigenvector', 'Btw',
    'BtwU1nn', 'EigenvectorU', 'BtwU',
]
supported_attacks += ['BtwU_cutoff{}'.format(l) for l in range(2, 100)]

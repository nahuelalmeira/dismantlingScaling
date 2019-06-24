import sys

def get_base_network_name(net_type, size, param):
    if net_type == 'ER':
        N = size
        p = param
        base_net_name = 'ER_N{}_p{}'.format(N, p)
    elif net_type == 'BA':
        N = size
        m = param
        base_net_name = 'BA_N{}_m{}'.format(N, m)
    elif net_type == 'Lattice':
        L = size
        p = param
        base_net_name = 'Lattice_L{}_f{}'.format(L, p)
    else:
        print('ERROR: net_type not supported', file=sys.stderr)
        base_net_name = ''
    
    return base_net_name
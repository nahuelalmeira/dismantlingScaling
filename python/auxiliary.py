import sys

def get_base_network_name(net_type, size, param):
    if net_type == 'ER':
        N = int(size)
        k = float(param)
        base_net_name = 'ER_k{:.2f}'.format(k)
        base_net_name_size = base_net_name + '_N{}'.format(N)
    else:
        print('ERROR: net_type not supported', file=sys.stderr)
        base_net_name = ''
    
    return base_net_name, base_net_name_size
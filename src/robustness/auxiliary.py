import os
import errno
import tarfile
import pickle
import json
import logging
import igraph as ig
import numpy as np
from typing import Iterable, Tuple, Optional, Set

from robustness.planar import spatial_net_types, distance

logger = logging.getLogger(__name__)

simple_props = ['Ngcc', 'C', 'Cws', 'r', 'meank', 'D', 'meanl', 'meanlw']

def ig_graph_to_adjlist(G: ig.Graph) -> Iterable[Set[int]]:
    """
    >>> G = ig.Graph()
    >>> G.add_vertices(3)
    >>> G.add_edges([(0, 1), (1, 2)])
    >>> ig_graph_to_adjlist(G)
    [{1}, {0, 2}, {1}]
    """
    n = G.vcount()
    adjlist = [set([]) for _ in range(n)]
    for e in G.es():
        s, t = e.tuple
        adjlist[s].add(t)
        adjlist[t].add(s)
    return adjlist

def edgelist_to_adjlist(
    edgelist: Iterable[Tuple[int, int]],
    size: Optional[int] = None
) -> Iterable:
    """
    >>> # Empty list
    >>> edgelist = []
    >>> edgelist_to_adjlist(edgelist)
    []
    >>> # P4 graph
    >>> edgelist = [(0, 1), (1, 2), (2, 3)]
    >>> edgelist_to_adjlist(edgelist)
    [{1}, {0, 2}, {1, 3}, {2}]
    >>> # Accept lists instead of tuples
    >>> edgelist = [[0, 1], [1, 2], [2, 3]]
    >>> edgelist_to_adjlist(edgelist)
    [{1}, {0, 2}, {1, 3}, {2}]
    >>> # C4 graph
    >>> edgelist = [(0, 1), (1, 2), (2, 3), (3, 0)]
    >>> edgelist_to_adjlist(edgelist)
    [{1, 3}, {0, 2}, {1, 3}, {0, 2}]
    """
    if len(edgelist) == 0 and not size:
        return []

    n = size if size else np.max(edgelist) + 1

    adjlist = [set() for _ in range(n)]
    for source, target in edgelist:
        adjlist[source].add(target)
        adjlist[target].add(source)

    return adjlist

def get_property_file_name(prop, directory):
    if prop in simple_props:
        file_name = os.path.join(directory, prop + '_values.txt')
    return file_name

def get_base_network_name(
    net_type: str, 
    size: int, 
    param: str
) -> Tuple[str, str]:
    if net_type == 'ER':
        base_net_name = 'ER_k{:.2f}'.format(float(param))
    elif net_type == 'RR':
        base_net_name = 'RR_k{:02d}'.format(float(param))
    elif net_type == 'BA':
        base_net_name = 'BA_m{:02d}'.format(int(param))
    elif net_type == 'MR':
        if 'k' in param:
            base_net_name = 'MR_k{:.2f}'.format(float(param[1:]))
        elif 'rMST' == param:
            base_net_name = 'MR_rMST'
        else:
            base_net_name = 'MR_r{:.6f}'.format(float(param[1:]))
    elif net_type in ['Lattice', 'PLattice', 'Ld3', 'DT', 'PDT', 'GG', 'RN']:
        base_net_name = f'{net_type}_param'
    elif net_type == 'qDT':
        base_net_name = 'qDT_k{:.2f}'.format(float(param))
    else:
        logger.error(f'{net_type} not supported')
        base_net_name = ''

    if net_type in ['Lattice', 'PLattice', 'Ld3']:
        base_net_name_size = base_net_name + '_L{}'.format(int(size))
    else:
        base_net_name_size = base_net_name + '_N{}'.format(int(size))
    return base_net_name, base_net_name_size

supported_attacks = [
    'Ran', 'Deg', 'DegU', 'CIU', 'CIU2', 'Eigenvector', 'Btw',
    'BtwU1nn', 'EigenvectorU', 'BtwU', 'BtwWU'
]
supported_attacks += ['BtwU_cutoff{}'.format(l) for l in range(2, 1000)]
supported_attacks += ['Btw_cutoff{}'.format(l) for l in range(2, 1000)]
supported_attacks += ['BtwWU_cutoff{}'.format(l) for l in range(2, 1000)]

supported_attacks += ['Edge_Ran', 'Edge_BtwU']

def get_edge_weights(g, net_type, size, param, seed):
    if net_type not in spatial_net_types:
        logger.info('Network type not supported for this attack')

    N = size

    base_net_dir_name = '../networks/DT/DT_param/DT_param_N{}'.format(N)
    net_dir_name = os.path.join(
        base_net_dir_name, 'DT_param_N{}_{:05d}'.format(N, seed)
    )
    position_file_name = 'position.txt'
    full_position_file_name = os.path.join(net_dir_name, position_file_name)

    ## Extract positions from file
    tar_input_name = 'position.tar.gz'
    full_tar_input_name = os.path.join(net_dir_name, tar_input_name)
    if not os.path.exists(full_tar_input_name):
        logger.error('File ' + full_tar_input_name + ' does not exist')
    tar = tarfile.open(full_tar_input_name, 'r:gz')
    tar.extractall(net_dir_name)
    tar.close()

    positions = np.loadtxt(full_position_file_name)
    os.remove(full_position_file_name)

    edge_weights = []
    for e in g.es():
        s, t = e.tuple
        edge_weights.append(distance(s, t, positions))

    return edge_weights

def read_data_file(
    directory, 
    base_name, 
    reader, 
    file_ext='.txt', 
    compress_ext='.tar.gz'
):
    """Auxiliary function for reading common data files, 
    which could be compressed.

    Arguments:
        directory {[type]} -- [description]
        base_name {[type]} -- [description]
        reader {[type]} -- [description]

    Keyword Arguments:
        file_ext {str} -- [description] (default: {'.txt'})
        compress_ext {str} -- [description] (default: {'.tar.gz'})
    """


    def read(file_name, reader):
        if reader == 'numpy':
            return np.loadtxt(file_name)
        elif reader == 'numpyInt':
            return np.loadtxt(file_name, dtype='int')
        elif reader == 'igraph':
            return ig.Graph().Read_Edgelist(file_name, directed=False)
        elif reader == 'networkit':
            import networkit as netKit
            return netKit.readGraph(
                file_name, fileformat=netKit.Format.EdgeListSpaceZero, 
                directed=False
            )

    compress_file_name = base_name + compress_ext
    full_compress_file_name = os.path.join(directory, compress_file_name)
    tar_exist = os.path.isfile(full_compress_file_name)

    data_file_name = base_name + file_ext
    full_data_file_name = os.path.join(directory, data_file_name)
    if os.path.isfile(full_data_file_name):
        data = read(full_data_file_name, reader)

    elif tar_exist:
        tar = tarfile.open(full_compress_file_name, 'r:gz')
        tar.extractall(directory)
        tar.close()

        data = read(full_data_file_name, reader)

    else:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), full_data_file_name
            )

    if tar_exist:
        os.remove(full_data_file_name)

    return data


def get_number_of_nodes(net_type, size):
    if net_type in ['Lattice', 'PLattice']:
        L = int(size)
        N = L*L
    elif net_type == 'Ld3':
        L = int(size)
        N = L*L*L
    else:
        N = int(size)
    return N

####################################################
### Auxiliar methods for cut_nodes_statistics.py ###
####################################################

def get_position(net_type, size, net_dir=None):

    if net_type in ['Lattice', 'PLattice']:
        L = size
        position = np.array([[i//L, i%L] for i in range(L*L)])
    else:
        L = np.sqrt(size)
        position = read_data_file(net_dir, 'position', reader='numpy')
        position = position * L

    return position

def get_max_pos(dir_name):

    aux = read_data_file(dir_name, 'comp_data', reader='numpy')

    Ngcc_values = aux[:,0][::-1]
    delta_values = np.abs(np.diff(Ngcc_values))
    max_pos = np.argmax(delta_values)
    delta_max = delta_values[max_pos]

    return max_pos, delta_max


def load_delta_data(net_type, size, param, attack, seed, **kwargs):

    dir_name = os.path.join('../networks', net_type)  
    base_net_name, base_net_name_size = get_base_network_name(
        net_type, size, param
    )
    net_name = base_net_name_size + '_{:05d}'.format(seed)
    base_net_dir = os.path.join(dir_name, base_net_name, base_net_name_size)
    net_dir = os.path.join(base_net_dir, net_name)

    if net_type in ['PDT', 'GG', 'RN', 'MR']:
        ## Directory corresponding to DT 
        ## (in case other spatial network is used)
        DT_dir_name = os.path.join('../networks', 'DT')
        DT_base_net_name, DT_base_net_name_size = (
            get_base_network_name('DT', size, param)
        )
        DT_net_name = DT_base_net_name_size + '_{:05d}'.format(seed)
        DT_base_net_dir = os.path.join(
            DT_dir_name, DT_base_net_name, DT_base_net_name_size
        )
        pos_net_dir = os.path.join(DT_base_net_dir, DT_net_name)
    else:
        pos_net_dir = net_dir

    attack_dir_name = os.path.join(base_net_dir, net_name, attack)
    index_list = read_data_file(attack_dir_name, 'oi_list', reader='numpyInt')

    g = read_data_file(net_dir, net_name, reader='igraph')

    g.vs['oi'] = range(g.vcount())
    g.vs['position'] = get_position(net_type, size, pos_net_dir)
    g['attack_order'] = index_list

    max_pos, delta_max = get_max_pos(attack_dir_name)

    return g, max_pos, delta_max


def get_prop(g, prop):
    prop_dict = {
        'C':        g.transitivity_undirected(mode='zero'),
        'Cws':      g.transitivity_avglocal_undirected(mode='zero'),
        'r':        g.assortativity_degree(directed=False),
        'D':        g.diameter(directed=False),
        'meanl':    g.average_path_length(directed=False),
        'meanlw':   np.mean(g.shortest_paths(weights='weight', mode=ig.ALL)),
        'meank':    np.mean(g.degree()),
        'maxk':     np.max(g.degree()),
        'Btw_dist': g.betweenness(directed=False, nobigint=False)
    }
    return prop_dict[prop]

def save_json_data(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)

def save_pickle_data(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

####################################################
###                                              ###
####################################################


def powerlaw(X, a, c):
    return c*np.array(X)**a

def getLinearReg(sizes, values, scale='loglog', t=1.96):

    if scale == 'loglog':
        X = np.log(sizes)
        Y = np.log(values)
    elif scale == 'logy':
        X = np.array(sizes)
        Y = np.log(values)
    elif scale == 'logx':
        X = np.log(sizes)
        Y = np.array(values)
    elif scale == 'linear':
        X = np.array(sizes)
        Y = np.array(values)
    else:
        raise ValueError('ERROR: scale', scale, 'not supported')

    coeffs, cov = np.polyfit(X, Y, 1, cov=True)
    errors = np.sqrt(np.diag(cov))

    intercept = coeffs[1]
    slope = coeffs[0]
    std = t*errors[0] 
    Y_pred = intercept + X*slope
    y_error = t*std

    return np.exp(Y_pred), slope, y_error

if __name__ == '__main__':
    import doctest
    doctest.testmod()
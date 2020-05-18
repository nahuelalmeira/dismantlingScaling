import os
import sys
import errno
import tarfile
import igraph as ig
import numpy as np
from planar import spatial_net_types, distance

simple_props = ['Ngcc', 'C', 'Cws', 'r', 'meank', 'D', 'meanl', 'meanlw']

def get_property_file_name(prop, directory):
    if prop in simple_props:
        file_name = os.path.join(directory, prop + '_values.txt')
    return file_name

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
    elif net_type == 'Lattice':
        L = int(size)
        base_net_name = 'Lattice_param'
    elif net_type == 'Ld3':
        L = int(size)
        base_net_name = 'Ld3_param'
    elif net_type == 'MR':
        base_net_name = 'MR_rMST'
    elif net_type == 'DT':
        base_net_name = 'DT_param'
    elif net_type == 'GG':
        base_net_name = 'GG_param'
    elif net_type == 'RN':
        base_net_name = 'RN_param'
    else:
        print('ERROR: net_type not supported', file=sys.stderr)
        base_net_name = ''

    if net_type in ['Lattice', 'Ld3']:
        base_net_name_size = base_net_name + '_L{}'.format(L)
    else:
        base_net_name_size = base_net_name + '_N{}'.format(N)
    return base_net_name, base_net_name_size

supported_attacks = [
    'Ran', 'Deg', 'DegU', 'CIU', 'CIU2', 'Eigenvector', 'Btw',
    'BtwU1nn', 'EigenvectorU', 'BtwU', 'BtwWU'
]
supported_attacks += ['BtwU_cutoff{}'.format(l) for l in range(2, 1000)]
supported_attacks += ['BtwWU_cutoff{}'.format(l) for l in range(2, 1000)]

supported_attacks += ['Edge_Ran', 'Edge_BtwU']

def get_edge_weights(g, net_type, size, param, seed):
    if net_type not in spatial_net_types:
        print('Network type not supported for this attack')

    N = size

    base_net_dir_name = '../networks/DT/DT_param/DT_param_N{}'.format(N)
    net_dir_name = os.path.join(base_net_dir_name, 'DT_param_N{}_{:05d}'.format(N, seed))
    position_file_name = 'position.txt'
    full_position_file_name = os.path.join(net_dir_name, position_file_name)

    ## Extract positions from file
    tar_input_name = 'position.tar.gz'
    full_tar_input_name = os.path.join(net_dir_name, tar_input_name)
    if not os.path.exists(full_tar_input_name):
        print('ERROR: File', full_tar_input_name, 'does not exist')
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

def read_data_file(directory, base_name, reader, file_ext='.txt', compress_ext='.tar.gz'):
    """Auxiliary function for reading common data files, which could be compressed.

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
                file_name, fileformat=netKit.Format.EdgeListSpaceZero, directed=False
            )

    compress_file_name = base_name + compress_ext
    full_compress_file_name = os.path.join(directory, compress_file_name)
    tar_exist = False
    if os.path.isfile(full_compress_file_name):
        tar_exist = True

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
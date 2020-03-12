import os
import sys
import numpy as np
import igraph as ig
import queue

sys.path.append('./fast')
from functions import RD_attack, RCI_attack, graph_to_nn_set

def initial_attack(g, attack, out, random_state=0):

    if os.path.isfile(out):
        original_indices = np.loadtxt(out, dtype='int')
        return original_indices

    ## Set random seed for reproducibility
    np.random.seed(random_state)

    if attack == 'Ran':
        n = g.vcount()
        oi_arr = np.array(range(n))
        np.random.shuffle(oi_arr)
        original_indices = oi_arr
    else:
        ## Compute centrality
        if attack == 'Btw':
            c_values = g.betweenness(directed=False, nobigint=False)
        elif attack == 'Deg':
            c_values = g.degree()
        elif attack == 'Eigenvector':
            from igraph import arpack_options
            arpack_options.maxiter = 300000
            c_values = g.eigenvector_centrality(directed=False, arpack_options=arpack_options)
        original_indices = np.argsort(c_values)[::-1]

    if out:
        np.savetxt(out, original_indices, fmt='%d')

    return original_indices

def updated_attack(graph, attack, out=None, random_state=0):

    ## Set random seed for reproducibility
    np.random.seed(random_state)

    ## Create a copy of graph so as not to modify the original
    g = graph.copy()

    ## Save original index as a vertex property
    N0 = g.vcount()
    g.vs['name'] = range(N0)

    ## List with the node original indices in removal order
    original_indices = []

    j = 0
    if out:
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            oi_values = np.loadtxt(out, dtype='int', comments='\x00')
            g.delete_vertices(oi_values)
            oi_values = np.array(oi_values) ## In case oi_values is one single integer
            j += len(oi_values)
            np.savetxt(out, oi_values, fmt='%d')

        f = open(out, 'a+')

    while j < N0:

        ## Identify node to be removed
        if attack == 'BtwU':
            c_values = g.betweenness(directed=False, nobigint=False)
        elif 'BtwU_cutoff' in attack:
            cutoff = int(attack.split('cutoff')[1])
            c_values = g.betweenness(directed=False, nobigint=False, cutoff=cutoff)
        elif attack == 'DegU':
            c_values = g.degree()
        elif attack == 'EigenvectorU':
            from igraph import arpack_options
            arpack_options.maxiter = 300000
            c_values = g.eigenvector_centrality(directed=False, arpack_options=arpack_options)
        idx = np.argmax(c_values)

        ## Add index to list
        original_idx = g.vs[idx]['name']
        original_indices.append(original_idx)

        ## Remove node
        g.vs[idx].delete()

        j += 1

        if out:
            f.write('{:d}\n'.format(original_idx))
            f.flush()

    if out:
        f.close()

    return original_indices

def fast_updated_attack(g, attack, out=None, random_state=0):

    ## Set random seed for reproducibility
    np.random.seed(random_state)

    ## Save original index as a vertex property
    N0 = g.vcount()
    
    nn_set = graph_to_nn_set(g)

    if out:
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            oi_values = np.loadtxt(out, dtype='int', comments='\x00')
            oi_values = np.array(oi_values) ## In case oi_values is one single integer
            if len(oi_values) < N0:
                os.remove(out)

    if attack == 'DegU':
        original_indices = RD_attack(nn_set)
    elif attack == 'CIU':
        original_indices = RCI_attack(nn_set, l=1)
    elif attack == 'CIU2':
        original_indices = RCI_attack(nn_set, l=2)

    if out:
        np.savetxt(out, original_indices, fmt='%d')

    return original_indices

def updated_local_attack(graph, attack, out=None, random_state=0):

    ## Set random seed for reproducibility
    np.random.seed(random_state)

    ## Create a copy of graph so as not to modify the original
    g = graph.copy()

    ## Save original index as a vertex property
    N0 = g.vcount()
    g.vs['name'] = range(N0)

    ## List with the node original indices in removal order
    original_indices = []

    j = 0
    if out:
        f = open(out, 'a+')

    c_values = g.betweenness(directed=False, nobigint=False)
    g.vs['centrality'] = c_values
    while j < N0:

        ## Identify node to be removed
        idx = np.argmax(g.vs['centrality'])

        ## Add index to list
        original_idx = g.vs[idx]['name']
        original_indices.append(original_idx)

        ## Update centrality of neighbors
        nn_indices = [w.index for w in g.vs[idx].neighbors()]


        ## Remove node
        g.vs[idx].delete()

        j += 1

        if out:
            f.write('{:d}\n'.format(original_idx))
            f.flush()

    if out:
        f.close()

    return original_indices

def updated_hybrid_attack(graph, attacks, probabilities, out=None, random_state=0):

    ## Set random seed for reproducibility
    np.random.seed(random_state)

    ## Create a copy of graph so as not to modify the original
    g = graph.copy()

    if 'EigenvalueU' in attacks:
        from igraph import arpack_options
        arpack_options.maxiter = 300000

    ## Save original index as a vertex property
    N0 = g.vcount()
    g.vs['name'] = range(N0)

    assert(len(attacks) == len(probabilities))

    ## Normalize probabilities
    probabilities = np.array(probabilities, dtype='float')
    probabilities = probabilities / np.sum(probabilities)

    ## List with the node original indices in removal order
    original_indices = []

    j = 0
    if out:
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            oi_values = np.loadtxt(out, dtype='int', comments='\x00')
            g.delete_vertices(oi_values)
            oi_values = np.array(oi_values) ## In case oi_values is one single integer
            j += len(oi_values)
            np.savetxt(out, oi_values, fmt='%d')

        f = open(out, 'a+')

    while j < N0:

        attack = np.random.choice(attacks, 1, p=probabilities)[0]

        if attack == 'Ran':
            idx = np.random.randint(0, g.vcount())
        else:
            ## Identify node to be removed
            if attack == 'BtwU':
                c_values = g.betweenness(directed=False, nobigint=False)
            elif attack == 'DegU':
                c_values = g.degree()
            elif attack == 'EigenvectorU':
                c_values = g.eigenvector_centrality(directed=False, arpack_options=arpack_options)
            idx = np.argmax(c_values)

        ## Add index to list
        original_idx = g.vs[idx]['name']
        original_indices.append(original_idx)

        ## Remove node
        g.vs[idx].delete()

        j += 1

        if out:
            f.write('{:d}\n'.format(original_idx))
            f.flush()

    if out:
        f.close()

    return original_indices


def get_index_list(G, attack, out=None, random_state=0):
    """
    Write to output out index list in order of removal
    """

    if G.is_directed():
        print('ERROR: G must be undirected.', file=sys.stderr)
        return 1

    #if not G.is_connected():
    #    print('ERROR: G must be connected.', file=sys.stderr)
    #    return 1

    if not G.is_simple():
        print('ERROR: G must be simple.', file=sys.stderr)
        return 1

    supported_attacks = {
        'initial': ['Ran', 'Deg', 'Btw', 'Eigenvector'],
        'updated': ['DegU', 'BtwU', 'EigenvectorU'] + \
                   ['BtwU_cutoff{}'.format(i) for i in range(2, 100)],
        'updated_local': ['BtwU1nn'],
        'fast_updated': ['DegU', 'CIU', 'CIU2']
    }

    if attack in supported_attacks['initial']:
        index_list = initial_attack(G, attack, out, random_state=random_state)
    elif attack in supported_attacks['updated']:
        index_list = updated_attack(G, attack, out, random_state=random_state)
    elif attack in supported_attacks['updated_local']:
        index_list = updated_local_attack(G, attack, out, random_state=random_state)
    elif attack in supported_attacks['fast_updated']:
        index_list = fast_updated_attack(G, attack, out, random_state=random_state)
    else:
        print('ERROR: Attack {} not supported.'.format(attack),
              file=sys.stderr)

    return index_list


def get_index_list_hybrid(G, attacks, probabilities, out=None, random_state=0):
    """
    Write to output out index list in order of removal
    """

    if G.is_directed():
        print('ERROR: G must be undirected.', file=sys.stderr)
        return 1

    if not G.is_simple():
        print('ERROR: G must be simple.', file=sys.stderr)
        return 1

    index_list = updated_hybrid_attack(G, attacks, probabilities, out, random_state=random_state)

    return index_list


def get_index_list_nk(G, attack, out=None, random_state=0):
    """
    Write to output out index list in order of removal
    (it uses the networKit package)
    TODO: Write function
    """

    index_list = []

    return index_list


if __name__ == '__main__':

    test_net_1 = [
        (0, 1),
        (1, 2),
        (1, 3),
        (3, 4),
        (3, 5),
        (3, 6)
    ]

    test_net_2 = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        (1, 4),
        (2, 5),
        (3, 6),
        (6, 7)
    ]

    test_net_3 = [
        (0, 1),
        (1, 2),
        (2, 3),
        (2, 4),
        (4, 5),
        (5, 6),
        (5, 7),
        (7, 8)
    ]

    print('Testing network 1')
    n = np.max(test_net_1) + 1
    g = ig.Graph()
    g.add_vertices(n)
    g.add_edges(test_net_1)
    oi_list = get_index_list(g, 'Deg')
    print(collective_influence(g, 1))
    assert(oi_list[:2].tolist() == [3, 1])

    print('Testing network 2')
    n = np.max(test_net_2) + 1
    g = ig.Graph()
    g.add_vertices(n)
    g.add_edges(test_net_2)
    oi_list = get_index_list(g, 'Deg')
    assert(oi_list[3] == 0)
    oi_list = get_index_list(g, 'DegU')
    assert(0 not in oi_list[:4])

    print('Testing network 3')
    n = np.max(test_net_3) + 1
    g = ig.Graph()
    g.add_vertices(n)
    g.add_edges(test_net_3)
    oi_list = get_index_list(g, 'Btw')
    assert(set(oi_list[:2].tolist()) == set([2, 5]))
    assert(oi_list[2] == 4)
    oi_list = get_index_list(g, 'BtwU')
    assert(4 in oi_list[4:])

    print('Testing ER 500')
    g = ig.Graph().Read_Edgelist('./test/ER_N500_p0.008_00000_gcc.txt',
                                 directed=False)
    oi_list = get_index_list(g, 'Deg')
    oi_list2 = np.loadtxt('./test/oi_list_ER_N500_p0.008_00000_Deg.txt', dtype='int')
    for i in range(10):
        print(oi_list[i], g.vs[oi_list[i]].degree(), oi_list2[i], g.vs[oi_list2[i]].degree())
    #print(oi_list[:10])
    #print(oi_list2[:10])

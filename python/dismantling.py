import os
import sys
import numpy as np
import igraph as ig
import queue

sys.path.append(os.path.join(os.path.dirname(__file__), 'fast'))
from functions import RD_attack, RCI_attack, graph_to_nn_set

def initial_attack(g, attack, out=None, random_state=0):

    if out and os.path.isfile(out):
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

def edge_initial_attack(g, attack, out=None, random_state=0):

    if out and os.path.isfile(out):
        tuple_values = np.loadtxt(out, dtype='int')
        return tuple_values

    ## Set random seed for reproducibility
    np.random.seed(random_state)

    if attack == 'Edge_Ran':
        m = g.ecount()
        oi_arr = np.array(range(m))
        np.random.shuffle(oi_arr)
        tuple_values = [g.es[idx].tuple for idx in oi_arr]

    if out:
        np.savetxt(out, tuple_values, fmt='%d')

    return tuple_values

def updated_attack(graph, attack, out=None, random_state=0):

    ## Set random seed for reproducibility
    np.random.seed(random_state)

    ## Create a copy of graph so as not to modify the original
    g = graph.copy()
    if 'BtwWU' in attack:
        g.es['weight'] = graph.es['weight']
        #print(g.es['weight'])

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
        elif attack == 'BtwWU':
            #print(g.es['weight'])
            weights = g.es['weight'] if g.ecount() else None
            c_values = g.betweenness(directed=False, weights=weights, nobigint=False)
        elif 'BtwU_cutoff' in attack:
            cutoff = int(attack.split('cutoff')[1])
            c_values = g.betweenness(directed=False, nobigint=False, cutoff=cutoff)
        elif 'BtwWU_cutoff' in attack:
            cutoff = int(attack.split('cutoff')[1])
            weights = g.es['weight'] if g.ecount() else None
            c_values = g.betweenness(directed=False, weights=weights, nobigint=False, cutoff=cutoff)
        elif attack == 'DegU':
            c_values = g.degree()
        elif attack == 'EigenvectorU':
            from igraph import arpack_options
            arpack_options.maxiter = 300000
            c_values = g.eigenvector_centrality(directed=False, arpack_options=arpack_options)

        #idx = np.argmax(c_values)
        c_values = np.around(c_values, decimals=8)
        m = np.max(c_values)
        m_indices = [i for i, value in enumerate(c_values) if value==m]

        idx = np.random.choice(m_indices)
        #if j < 10:
        #    print(j, idx, len(m_indices), m_indices, sorted(c_values, reverse=True)[:4])


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

def edge_updated_attack(graph, attack, out=None, random_state=0):

    ## Set random seed for reproducibility
    np.random.seed(random_state)

    ## Create a copy of graph so as not to modify the original
    g = graph.copy()
    if 'BtwWU' in attack:
        g.es['weight'] = graph.es['weight']

    M0 = g.ecount()

    tuple_list = []

    j = 0
    if out:
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            tuple_list = np.loadtxt(out, dtype='int', comments='\x00')
            tuple_list = np.array(tuple_list) ## In case oi_values is one single integer
            tuple_list = [(s,t) for (s,t) in tuple_list]
            g.delete_edges(tuple_list)
            j += len(tuple_list)
            np.savetxt(out, tuple_list, fmt='%d')

        f = open(out, 'a+')

    while j < M0:

        ## Identify edge to be removed
        if attack == 'Edge_BtwU':
            c_values = g.edge_betweenness(directed=False)

        idx = np.argmax(c_values)
        tuple_value = g.es[idx].tuple
        tuple_list.append(tuple_value)

        ## Remove edge
        g.es[idx].delete()

        j += 1

        if out:
            f.write('{:d} {:d}\n'.format(*tuple_value))
            f.flush()

    if out:
        f.close()

    return tuple_list

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
        'updated': ['BtwU', 'EigenvectorU', 'BtwWU'] + \
                   ['BtwU_cutoff{}'.format(i) for i in range(2, 1000)] + \
                   ['BtwWU_cutoff{}'.format(i) for i in range(2, 1000)],
        'updated_local': ['BtwU1nn'],
        'fast_updated': ['DegU', 'CIU', 'CIU2'],
        'edge_initial': ['Edge_Ran'],
        'edge_updated': ['Edge_BtwU']
    }

    if attack in supported_attacks['initial']:
        index_list = initial_attack(G, attack, out=out, random_state=random_state)
    elif attack in supported_attacks['updated']:
        index_list = updated_attack(G, attack, out=out, random_state=random_state)
    elif attack in supported_attacks['updated_local']:
        index_list = updated_local_attack(G, attack, out=out, random_state=random_state)
    elif attack in supported_attacks['fast_updated']:
        index_list = fast_updated_attack(G, attack, out=out, random_state=random_state)
    elif attack in supported_attacks['edge_initial']:
        index_list = edge_initial_attack(G, attack, out=out, random_state=random_state)
    elif attack in supported_attacks['edge_updated']:
        index_list = edge_updated_attack(G, attack, out=out, random_state=random_state)
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

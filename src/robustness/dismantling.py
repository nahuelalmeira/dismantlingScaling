import os
import logging
import numpy as np
import igraph as ig

from robustness.fast.functions import (
    get_CI, RD_attack, RCI_attack, graph_to_nn_set
)
from robustness.auxiliary import ig_graph_to_adjlist

logger = logging.getLogger(__name__)
#logger.setLevel(getattr(logging, logging_level))

def initial_attack(
    g, attack, out=None,
    save_centrality=False,
    out_centrality=None,
    random_state=0
):

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
            if isinstance(g, ig.Graph):
                c_values = g.betweenness(directed=False, nobigint=False)
            else:
                import networkit as netKit
                btw = netKit.centrality.Betweenness(g)
                btw.run()
                ranking = btw.ranking()
                indices, c_values = zip(*list(sorted(ranking, key=lambda x: x[0])))
                c_values = [c/2 for c in c_values]
        elif 'Btw_cutoff' in attack:
            cutoff = int(attack.split('cutoff')[1])
            c_values = g.betweenness(
                directed=False, nobigint=False, cutoff=cutoff
            )
        elif attack == 'Deg':
            c_values = g.degree()
        elif attack == 'Eigenvector':
            from igraph import arpack_options
            arpack_options.maxiter = 300000
            c_values = g.eigenvector_centrality(
                directed=False, arpack_options=arpack_options
            )
        elif attack == 'CI':
            adjlist = ig_graph_to_adjlist(g)
            c_values = get_CI(adjlist, l=1)
        original_indices = np.argsort(c_values)[::-1]

    if out:
        np.savetxt(out, original_indices, fmt='%d')

    if save_centrality:
        np.savetxt(out_centrality, c_values)

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

def updated_attack_ig(
    graph,
    attack,
    save_centrality=False, 
    out=None, 
    out_centrality=None,
    random_state=0
):

    ## Set random seed for reproducibility
    np.random.seed(random_state)

    ## Create a copy of graph so as not to modify the original
    g = graph.copy()
    if 'BtwWU' in attack:
        g.es['weight'] = graph.es['weight']

    ## Save original index as a vertex property
    N0 = g.vcount()
    g.vs['name'] = range(N0)

    ## List with the node original indices in removal order
    original_indices = []
    centrality_stats = []

    j = 0
    if out:
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            oi_values = np.loadtxt(out, dtype='int', comments='\x00')
            if oi_values.size == 1:
                oi_values = np.array([oi_values], dtype='int')
            g.delete_vertices(oi_values)
            #j += len(oi_values)
            j += oi_values.size
            np.savetxt(out, oi_values, fmt='%d')
        f = open(out, 'a+')
    if save_centrality:
        if not out_centrality:
            raise ValueError('Missing value for out_centrality')
        if os.path.isfile(out_centrality) and os.path.getsize(out_centrality) > 0:
                centrality_stats = np.loadtxt(
                    out_centrality, comments='\x00'
                ).tolist()
        f_centrality = open(out_centrality, 'a+')

    while j < N0:

        ## Identify node to be removed
        if attack == 'BtwU':
            c_values = g.betweenness(directed=False, nobigint=False)
        elif attack == 'BtwWU':
            weights = g.es['weight'] if g.ecount() else None
            c_values = g.betweenness(
                directed=False, weights=weights, nobigint=False
            )
        elif 'BtwU_cutoff' in attack:
            cutoff = int(attack.split('cutoff')[1])
            c_values = g.betweenness(
                directed=False, nobigint=False, cutoff=cutoff
            )
        elif 'BtwWU_cutoff' in attack:
            cutoff = int(attack.split('cutoff')[1])
            weights = g.es['weight'] if g.ecount() else None
            c_values = g.betweenness(
                directed=False, weights=weights, nobigint=False, cutoff=cutoff
            )
        elif attack == 'DegU':
            c_values = g.degree()
        elif attack == 'EigenvectorU':
            from igraph import arpack_options
            arpack_options.maxiter = 300000
            c_values = g.eigenvector_centrality(
                directed=False, arpack_options=arpack_options
            )

        c_values = np.around(c_values, decimals=8)
        m = np.max(c_values)
        if c_values.shape[0] > 1:
            m_sec = np.sort(c_values)[-2]
        else:
            m_sec = np.NaN
        mean = np.mean(c_values)
        c_stats = [m, mean, m_sec]
        m_indices = [i for i, value in enumerate(c_values) if value==m]

        idx = np.random.choice(m_indices)

        ## Add index to list
        original_idx = g.vs[idx]['name']
        original_indices.append(original_idx)
        centrality_stats.append(c_stats)

        ## Remove node
        g.vs[idx].delete()

        j += 1

        if out_centrality:
            f.write('{:d}\n'.format(original_idx))
            f.flush()
            f_centrality.write(
                ','.join([str(elem) for elem in c_stats]) + '\n'
            )
            f_centrality.flush()

    if out:
        f.close()
    if out_centrality:
        f_centrality.close()

    return original_indices, centrality_stats

def updated_attack_nk(
    g, 
    attack, 
    save_centrality=False, 
    out=None, 
    out_centrality=None,
    random_state=0
):

    def perform_step(g, attack):
        if attack == 'BtwU':
            btw = netKit.centrality.Betweenness(g)
            btw.run()
            ranking = btw.ranking()
            c_values = np.array([elem[1]/2 for elem in ranking])
            idx = ranking[0][0]
            c_value, mean_c = c_values.max(), c_values.mean()
            c_stats = [c_value, mean_c]
        return idx, c_stats

    import networkit as netKit

    ## Set random seed for reproducibility
    np.random.seed(random_state)

    order = []
    centrality_stats = []
    if out:
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            oi_values = np.loadtxt(out, dtype='int', comments='\x00')
            if oi_values.size == 1:
                oi_values = np.array([oi_values], dtype='int')
            if save_centrality:
                if not out_centrality:
                    raise ValueError('Missing value for out_centrality')
                centrality_stats = np.loadtxt(
                    out_centrality, comments='\x00'
                ).tolist()
            for oi in oi_values:
                g.removeNode(oi)
                order.append(oi)

        f = open(out, 'a+')
        if save_centrality:
            f_centrality = open(out_centrality, 'a+')

    N = g.numberOfNodes()
    while N:

        idx, c_stats = perform_step(g, attack)

        order.append(idx)
        centrality_stats.append(c_stats)
        g.removeNode(idx)

        N -= 1

        if out:
            f.write('{:d}\n'.format(idx))
            f.flush()
            if save_centrality:
                f_centrality.write(
                    ','.join([str(elem) for elem in c_stats]) + '\n'
                )
                f_centrality.flush()

    if out:
        f.close()
        if save_centrality:
            f_centrality.close()

    return order, centrality_stats

def updated_attack(
    graph, 
    attack, 
    out=None, 
    save_centrality=False,
    out_centrality=None, 
    random_state=0
):

    if isinstance(graph, ig.Graph):
        return updated_attack_ig(
            graph, 
            attack, 
            out=out, 
            save_centrality=save_centrality, 
            out_centrality=out_centrality, 
            random_state=random_state
        )
    return updated_attack_nk(
            graph, 
            attack, 
            out=out, 
            save_centrality=save_centrality, 
            out_centrality=out_centrality, 
            random_state=random_state
        )

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
            ## In case oi_values is one single integer
            tuple_list = np.array(tuple_list) 
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
            ## In case oi_values is one single integer
            oi_values = np.array(oi_values) 
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

def updated_hybrid_attack(
    graph, 
    attacks, 
    probabilities, 
    out=None, 
    random_state=0
):

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
            if oi_values.size == 1:
                oi_values = np.array([oi_values], dtype='int')
            g.delete_vertices(oi_values)
            #j += len(oi_values)
            j += oi_values.size
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
                c_values = g.eigenvector_centrality(
                    directed=False, arpack_options=arpack_options
            )
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


def get_index_list(
    G, attack, out=None,
    save_centrality=False,
    out_centrality=None,
    random_state=0
):
    """
    Write to output out index list in order of removal
    """

    supported_attacks = {
        'initial': ['Ran', 'Deg', 'CI', 'Btw', 'Eigenvector'] + \
                   ['Btw_cutoff{}'.format(i) for i in range(2, 1000)],
        'updated': ['BtwU', 'EigenvectorU', 'BtwWU'] + \
                   ['BtwU_cutoff{}'.format(i) for i in range(2, 1000)] + \
                   ['BtwWU_cutoff{}'.format(i) for i in range(2, 1000)],
        'updated_local': ['BtwU1nn'],
        'fast_updated': ['DegU', 'CIU', 'CIU2'],
        'edge_initial': ['Edge_Ran'],
        'edge_updated': ['Edge_BtwU']
    }
    if attack in supported_attacks['initial']:
        index_list = initial_attack(
            G, attack, out=out, save_centrality=save_centrality,
            out_centrality=out_centrality,
            random_state=random_state
        )
    elif attack in supported_attacks['updated']:
        index_list = updated_attack(
            G, attack, out=out, save_centrality=save_centrality,
            out_centrality=out_centrality,
            random_state=random_state
        )
    elif attack in supported_attacks['updated_local']:
        index_list = updated_local_attack(
            G, attack, out=out, random_state=random_state
        )
    elif attack in supported_attacks['fast_updated']:
        index_list = fast_updated_attack(
            G, attack, out=out, random_state=random_state
        )
    elif attack in supported_attacks['edge_initial']:
        index_list = edge_initial_attack(
            G, attack, out=out, random_state=random_state
        )
    elif attack in supported_attacks['edge_updated']:
        index_list = edge_updated_attack(
            G, attack, out=out, random_state=random_state
        )
    else:
        logger.error(f'Attack {attack} not supported.')

    return index_list


def get_index_list_hybrid(
    G, 
    attacks, 
    probabilities, 
    out=None, 
    random_state=0
):
    """
    Write to output out index list in order of removal
    """

    if G.is_directed():
        logger.error('G must be undirected.')
        return 1

    if not G.is_simple():
        logger.error('G must be simple.')
        return 1

    index_list = updated_hybrid_attack(
        G, attacks, probabilities, out, random_state=random_state
    )

    return index_list

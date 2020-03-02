import numpy as np
import copy
import queue

## Auxiliar methods
def powerlaw(X, a, c):
    return c*X**a

##############################

## Graph manipulation methods

def edgelist_to_nn_set(edgelist):
    N = np.max(edgelist) + 1
    nn_set = []
    for _ in range(N):
        nn_set.append(set([]))

    for u, v in edgelist:
        nn_set[u].add(v)
        nn_set[v].add(u)

    return nn_set

################################

## Percolation

def findroot(ptr, i):
    """
    Auxiliary method for 'percolate'
    Finds the root of node i and modify the structure
    on-the-fly.

    Arguments:
        ptr {list} -- Recursive list structure
        i {int} -- node

    Returns:
        int -- root of node i
    """

    if ptr[i] < 0:
        return i
    ptr[i] = findroot(ptr, ptr[i])
    return ptr[i]

def percolate(nn_set, order):
    """
    Newman-Ziff algorithm for percolation.
    Adds nodes in a specific order and computes
    giant component on-the-fly.

    Arguments:
        nn_set {list} -- adjacency list
        order {list} -- order in which nodes are removed

    Returns:
        list -- Size of giant component
    """

    N = len(order)
    EMPTY = -(N+1)

    ptr = np.zeros(N, dtype='int') + EMPTY

    N1_values = []
    N1 = 1

    for i in range(N):
        r1 = s1 = order[i]
        ptr[s1] = -1
        for s2 in nn_set[s1]:
            if ptr[s2] != EMPTY:
                r2 = findroot(ptr, s2)
                if r2 != r1:
                    if ptr[r1] > ptr[r2]: ## s2 belongs to a greater component than s1
                        ptr[r2] += ptr[r1] ## Merge s1 to s2
                        ptr[r1] = r2
                        r1 = r2

                    else:
                        ptr[r1] += ptr[r2]
                        ptr[r2] = r1
                    if -ptr[r1] > N1:
                        N1 = -ptr[r1]

        N1_values.append(N1)

    return N1_values

###########################################

## Attacks

def ID_attack(nn_set):
    """
    Attack based on Initial Degree centrality

    Arguments:
        nn_set {list} -- List of neighor sets for each node

    Returns:
        list -- Order in which the nodes have been removed
    """

    deg_seq = [len(s) for s in nn_set]
    order = np.argsort(deg_seq)[::-1]
    return order.tolist()

def RD_naive_attack(nn_set):
    """
    Attack based on Recalculated Degree centrality.
    NOTE: This is a slow version of the algorithm, and runs
    in O(N^2).

    Arguments:
        nn_set {list} -- List of neighor sets for each node

    Returns:
        list -- Order in which the nodes have been removed
    """

    def _get_deg_seq(nn_set):
        """
        Auxiliar method for RD_attack_naive.
        Computes the degree sequence, where nodes
        that have been previously removed are given a
        value -1.

        Arguments:
            nn_set {list} -- Adjacency list

        Returns:
            numpy.array -- Degree sequence
        """

        N = len(nn_set)
        deg_seq = np.zeros(N, dtype='int')
        for i in range(N):
            if isinstance(nn_set[i], set):
                deg_seq[i] = len(nn_set[i])
            else:
                deg_seq[i] = -1 ## Represents previously removed node
        return deg_seq

    def _remove_node(nn_set, idx):
        for nn in nn_set[idx]:
            nn_set[nn].remove(idx)
        nn_set[idx] = np.NaN

    nn_set = copy.deepcopy(nn_set)

    N = len(nn_set)
    order = []
    for _ in range(N):
        deg_seq = _get_deg_seq(nn_set)
        idx = np.argmax(deg_seq)
        _remove_node(nn_set, idx)
        order.append(idx)
    return order

def create_deg_struct(nn_set):

    deg_seq = [len(s) for s in nn_set]
    kmax = np.max(deg_seq)
    assert(np.min(deg_seq) >= 0)
    deg_struct = []

    for _ in range(kmax+1):
        deg_struct.append(set([]))

    for i, k in enumerate(deg_seq):
        deg_struct[k].add(i)

    return deg_struct

def RD_attack(nn_set):
    """
    Attack based on Recalculated Degree centrality.
    Complexity: O(N)

    Arguments:
        nn_set {list} -- List of neighor sets for each node

    Returns:
        list -- Order in which the nodes have been removed
    """

    N = len(nn_set)

    deg_seq = [len(s) for s in nn_set]
    deg_struct = create_deg_struct(nn_set)
    kmax = len(deg_struct) - 1

    order = []
    for _ in range(N):
        while not deg_struct[kmax]:
            kmax = kmax - 1

        ## Take node v
        v = deg_struct[kmax].pop()
        order.append(v)

        ## Update data structures
        for w in nn_set[v]:
            kw = deg_seq[w]

            ## Update deg_struct
            deg_struct[kw].remove(w)
            if kw > 0:
                deg_struct[kw-1].add(w)

            deg_seq[w] = deg_seq[w] - 1
            nn_set[w].remove(v)
        nn_set[v].clear()
        deg_seq[v] = 0

    return order


def get_neighbors_1(nn_set, v):
    return nn_set[v]

def get_neighbors_2(nn_set, v):
    neighbors = nn_set[v]

    for nn in nn_set[v]:
        neighbors = neighbors.union(nn_set[nn])
    if v in neighbors:
        neighbors.remove(v)
    return neighbors

def get_neighbors_3(nn_set, v):

    neighbors = nn_set[v]
    for n1 in nn_set[v]:
        neighbors = neighbors.union(nn_set[n1])
        for n2 in nn_set[n1]:
            neighbors = neighbors.union(nn_set[n2])
    if v in neighbors:
        neighbors.remove(v)
    return neighbors

def get_neighbors_ball(nn_set, v, l):
    if l == 1:
        return get_neighbors_1(nn_set, v)
    if l == 2:
        return get_neighbors_2(nn_set, v)
    if l == 3:
        return get_neighbors_3(nn_set, v)

def get_neighbors_border_2(nn_set, v):
    neighbors = set([])

    for nn in nn_set[v]:
        neighbors = neighbors.union(nn_set[nn])
    if v in neighbors:
        neighbors.remove(v)
    neighbors = neighbors.difference(nn_set[v])
    return neighbors

def get_neighbors_border_3(nn_set, v):

    neighbors = set([])
    for n1 in nn_set[v]:
        for n2 in nn_set[n1]:
            neighbors = neighbors.union(nn_set[n2])
    if v in neighbors:
        neighbors.remove(v)

    for n1 in nn_set[v]:
        neighbors = neighbors.difference(nn_set[n1])
    neighbors = neighbors.intersection(nn_set[v])
    return neighbors


def get_neighbors_ball_border(nn_set, v, l):
    if l == 1:
        return get_neighbors_1(nn_set, v)
    if l == 2:
        return get_neighbors_border_2(nn_set, v)
        #return get_neighbors_2(nn_set, v).intersection(get_neighbors_1(nn_set, v))
    if l == 3:
        return get_neighbors_border_3(nn_set, v)

def _get_neighbors_ball(nn_set, v, l):
    """
    Finds neighbors ball up to distance 'l' from
    node 'v' using BFS.

    Arguments:
        nn_set {list} -- Adjacency list
        v {int} -- Base node
        l {int} -- Distance to base node

    Returns:
        list -- Neighbors ball
    """
    Q = queue.Queue()
    Q.put(v)
    N = len(nn_set)
    distances = [-1]*N ## Distances to node v (initially set to -1)
    distance = 0
    distances[v] = distance
    ball = []
    while not Q.empty():
        u = Q.get()
        distance = distances[u]
        if distances[u] == l:
            break

        for nn in nn_set[u]: ## Iterate over neighbors of u
            if distances[nn] < 0: ## Unseen node
                distances[nn] = distance + 1
                Q.put(nn)
                ball.append(nn)

    return ball

def _get_neighbors_ball_border(nn_set, v, l):
    """
    Finds nodes at distance 'l' from node 'v' using BFS.

    Arguments:
        nn_set {list} -- Adjacency list
        v {int} -- Base node
        l {int} -- Distance to base node

    Returns:
        list -- Neighbors
    """
    Q = queue.Queue()
    Q.put(v)
    N = len(nn_set)
    distances = [-1]*N ## Distances to node v (initially set to -1)
    distance = 0
    distances[v] = distance
    while not Q.empty():
        u = Q.get()
        distance = distances[u]
        if distances[u] == l:
            break

        for nn in nn_set[u]: ## Iterate over neighbors of u
            if distances[nn] < 0: ## Unseen node
                distances[nn] = distance + 1
                Q.put(nn)
    neighbors = np.where(np.array(distances)==l)[0].tolist()
    return neighbors

def get_CI(nn_set, l):
    """
    Compute Collective Influence up to length 'l'

        CI_i = (k_i - 1) * sum_{j in Ball(i, l)} (k_j - 1)

    Arguments:
        nn_set {[type]} -- [description]
        l {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    deg_seq = [len(s) for s in nn_set]

    CI_values = []
    for i, k_i in enumerate(deg_seq):
        ball_nodes = get_neighbors_ball(nn_set, i, l)
        ball_deg = sum([(deg_seq[n] - 1) for n in ball_nodes])
        CI_values.append((k_i - 1) * ball_deg)

    return CI_values

def ICI_attack(nn_set, l):
    """
    Attack based on Initial Collective Influence centrality
    with ball distance 'l'.
    Complexity: o(N^2)

    Arguments:
        nn_set {list} -- List of neighor sets for each node
        l {int} -- Length of ball
    Returns:
        list -- Order in which the nodes have been removed
    """

    CI_values = get_CI(nn_set, l)
    order = np.argsort(CI_values)[::-1]
    return order.tolist()

def create_CI_struct(nn_set, l):

    CI_seq = get_CI(nn_set, l)
    CImax = np.max(CI_seq)
    CI_struct = []

    for _ in range(CImax+1):
        CI_struct.append(set([]))

    for i, ci in enumerate(CI_seq):
        CI_struct[ci].add(i)

    return CI_struct

def update_ci(v, w, nn_set, deg_seq, CI_seq, l):

    kv = deg_seq[v]
    kw = deg_seq[w]
    ci_w = CI_seq[w]

    common_nn = nn_set[v].intersection(nn_set[w])

    if l == 1:
        if w in nn_set[v]: ## Case w nearest neighbor of v
            w_neighbors = nn_set[w].difference(set([v]))

            s = 0
            for w_nn in w_neighbors:
                k_nn = deg_seq[w_nn]
                s += (k_nn - 1)

            new_ci_w = ci_w - (kw-1)*(kv-1) - s - (kw-2)*len(common_nn)

        else: ## Case w not nearest neighbor of v.
            new_ci_w = ci_w - (kw-1)*len(common_nn)
    
    if l == 2:
        second_neighbors = set(get_neighbors_ball_border(nn_set, w, 2))
        common_second_neighbors = nn_set[v].intersection(second_neighbors)
        if w in nn_set[v]:
            
            S1 = nn_set[v].intersection(nn_set[w])
            S2 = set([])
            for u in nn_set[w].difference(S1).difference(set([v])):
                S2 = S2.union(nn_set[u])
            S2 = S2.intersection(nn_set[v]).difference(set([w]))
            S3 = nn_set[v].difference(S1.union(S2).union([w]))
            
            sum1 = 0
            for w_nn in nn_set[w].difference(set([v])):
                k_nn = deg_seq[w_nn]
                sum1 += (k_nn - 1)
            sum2 = 0
            for w_sn in set(get_neighbors_ball_border(nn_set, w, 2)):
                k_sn = deg_seq[w_sn]
                sum2 += (k_sn - 1)

            sum3 = 0
            for r in S3:
                k = deg_seq[r]
                sum3 += (k - 1)

            new_ci_w = ci_w - (kw-1)*(kv-1) - sum1 - sum2 - (kw-2)*(len(S1) + len(S2) + sum3)
        elif w in set(get_neighbors_ball_border(nn_set, v, 2)):
            #print('2 nn', common_second_neighbors)
            new_ci_w = ci_w - (kw-1)*(kv-1) - (kw-1)*(len(common_nn) + len(common_second_neighbors))
        else:
            #print('3 nn')
            new_ci_w = ci_w - (kw-1)*len(common_second_neighbors)

    return new_ci_w  

def _update_ci(v, w, nn_set, deg_seq, CI_seq, l):

    kv = deg_seq[v]
    kw = deg_seq[w]
    ci_w = CI_seq[w]

    common_nn = nn_set[v].intersection(nn_set[w])

    if l == 1:
        if w in nn_set[v]: ## Case w nearest neighbor of v
            w_neighbors = nn_set[w].difference(set([v]))

            s = 0
            for w_nn in w_neighbors:
                k_nn = deg_seq[w_nn]
                s += (k_nn - 1)

            new_ci_w = ci_w - (kw-1)*(kv-1) - s - (kw-2)*len(common_nn)

        else: ## Case w not nearest neighbor of v.
            new_ci_w = ci_w - (kw-1)*len(common_nn)
    
    if l == 2:
        second_neighbors = set(get_neighbors_ball_border(nn_set, v, 2))
        common_second_neighbors = nn_set[v].intersection(second_neighbors)
        if w in nn_set[v]:
            w_neighbors = nn_set[w].difference(set([v]))

            s = 0
            for w_nn in w_neighbors:
                k_nn = deg_seq[w_nn]
                s += (k_nn - 1)

            for w_sn in second_neighbors:
                k_sn = deg_seq[w_sn]
                s += (k_sn - 1)

            new_ci_w = ci_w - (kw-1)*(kv-1) - s - (kw-2)*(len(common_nn) + len(common_second_neighbors))
        elif w in second_neighbors:
            new_ci_w = ci_w - (kw-1)*(len(common_nn) + len(common_second_neighbors))
        else:
            new_ci_w = ci_w - (kw-1)*len(common_second_neighbors)

    return new_ci_w    

def RCI_attack(nn_set, l=1):

    N = len(nn_set)
    order = []
    deg_seq = [len(s) for s in nn_set]
    CI_seq = get_CI(nn_set, l)
    CI_struct = create_CI_struct(nn_set, l)
    CImax = len(CI_struct) - 1

    for _ in range(N):
        while not CI_struct[CImax]:
            CImax = CImax - 1

        v = CI_struct[CImax].pop()
        order.append(v)

        ## Update CI structure
        first_neighbors = nn_set[v]
        for w in first_neighbors:

            ci_w = CI_seq[w]
            CI_struct[ci_w].remove(w)

            new_ci_w = update_ci(v, w, nn_set, deg_seq, CI_seq, l)
            CI_seq[w] = new_ci_w

            if new_ci_w >= 0:
                CI_struct[new_ci_w].add(w)

        second_neighbors = set(get_neighbors_ball_border(nn_set, v, 2))

        for w in second_neighbors:
            ci_w = CI_seq[w]
            CI_struct[ci_w].remove(w)
            new_ci_w = update_ci(v, w, nn_set, deg_seq, CI_seq, l)
            CI_seq[w] = new_ci_w
            if new_ci_w >= 0:
                CI_struct[new_ci_w].add(w)

        for w in first_neighbors:
            deg_seq[w] = deg_seq[w] - 1
            nn_set[w].remove(v)
        nn_set[v].clear()
        deg_seq[v] = 0
        CI_seq[v] = 0

    return order

###########################################


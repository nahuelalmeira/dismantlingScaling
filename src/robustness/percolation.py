import heapq
import numpy as np
from typing import Set, Dict, Iterable, Any
from robustness.auxiliary import ig_graph_to_adjlist

def findroot(ptr: Iterable[int], i: int) -> int:
    """
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

def compute_metrics(ptr):
    N = len(ptr)
    EMPTY = -N-1
    sizes = [-r for r in ptr if (r<0) and (r != EMPTY)]
    n_clusters = len(sizes)
    if (n_clusters == 1):
        sizes.append(0)
    if (n_clusters == 2):
        sizes.append(0)

    N1_idx = np.argmax(sizes)
    N1 = sizes[N1_idx]
    sizes.remove(N1)
    N2 = np.max(sizes)
    numerator, denominator = compute_meanS(sizes)
    if denominator == 0:
        meanS = 1.0
    else:
        meanS = numerator / denominator
    metrics = {}
    metrics['N1'] = N1
    metrics['N2'] = N2
    metrics['meanS'] = meanS
    metrics['num'] = numerator
    metrics['denom'] = denominator
    return metrics

def compute_meanS(sizes):
    numerator = 0
    denominator = 0
    N = len(sizes)
    for i in range(N):
        s = sizes[i]
        numerator += s*s
        denominator += s
    return numerator, denominator


def percolate_slow(
    adjlist: Iterable[Set[int]], 
    order: Iterable[int]
) -> Dict[str, Iterable[Any]]:
    """
    Newman-Ziff algorithm for percolation.
    Adds nodes in a specific order and computes
    giant component on-the-fly.

    Also computes other metrics, such as <s> and N2. Complexity O(N^2)

    Arguments:
        adjlist {list} -- adjacency list
        order {list} -- order in which nodes are removed

    Returns:
        metrics -- dictionary of the computed metrics

    >>> # Empty graph
    >>> adjlist = []
    >>> order = []
    >>> percolate_slow(adjlist, order)
    {'p': array([], dtype=float64), 'N1': [], 'N2': [], 'meanS': [], 'num': [], 'denom': []}
    >>> # P3 graph with Degree based attack
    >>> adjlist = [{1}, {0, 2}, {1}]
    >>> order = [1, 0, 2][::-1]
    >>> metrics = percolate_slow(adjlist, order)
    >>> {metric: [round(value, 2) for value in values] for metric, values in metrics.items()}
    {'p': [0.0, 0.33, 0.67], 'N1': [1, 1, 3], 'N2': [0, 1, 0], 'meanS': [1.0, 1.0, 1.0], 'num': [0, 1, 0], 'denom': [0, 1, 0]}
    >>> # P3 graph with ordered attack
    >>> adjlist = [{1}, {0, 2}, {1}]
    >>> order = [0, 1, 2][::-1]
    >>> metrics = percolate_slow(adjlist, order)
    >>> {metric: [round(value, 2) for value in values] for metric, values in metrics.items()}
    {'p': [0.0, 0.33, 0.67], 'N1': [1, 2, 3], 'N2': [0, 0, 0], 'meanS': [1.0, 1.0, 1.0], 'num': [0, 0, 0], 'denom': [0, 0, 0]}
    >>> # Works with np.array
    >>> order = np.array([0., 1., 2.][::-1])
    >>> metrics = percolate_slow(adjlist, order)
    >>> {metric: [round(value, 2) for value in values] for metric, values in metrics.items()}
    {'p': [0.0, 0.33, 0.67], 'N1': [1, 2, 3], 'N2': [0, 0, 0], 'meanS': [1.0, 1.0, 1.0], 'num': [0, 0, 0], 'denom': [0, 0, 0]}
    >>> adjlist = [{1, 3}, {0, 2, 3}, {1, 3}, {0, 1, 2, 4, 5}, {3, 5}, {3, 4}]
    >>> order = [3, 2, 1, 0, 4, 5][::-1]
    >>> metrics = percolate_slow(adjlist, order)
    >>> {metric: [round(value, 2) for value in values] for metric, values in metrics.items()}
    {'p': [0.0, 0.17, 0.33, 0.5, 0.67, 0.83], 'N1': [1, 2, 2, 2, 3, 6], 'N2': [0, 0, 1, 2, 2, 0], 'meanS': [1.0, 1.0, 1.0, 2.0, 2.0, 1.0], 'num': [0, 0, 1, 4, 4, 0], 'denom': [0, 0, 1, 2, 2, 0]}
    >>> adjlist = [{3}, {2, 6}, {1, 4}, {0}, {2}, {}, {1}]
    >>> order = [0, 1, 2, 3, 4, 5, 6]
    >>> metrics = percolate_slow(adjlist, order)
    >>> metrics = {metric: [round(value, 2) for value in values] for metric, values in metrics.items()}
    >>> metrics
    {'p': [0.0, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86], 'N1': [1, 1, 2, 2, 3, 3, 4], 'N2': [0, 1, 1, 2, 2, 2, 2], 'meanS': [1.0, 1.0, 1.0, 2.0, 2.0, 1.67, 1.67], 'num': [0, 1, 1, 4, 4, 5, 5], 'denom': [0, 1, 1, 2, 2, 3, 3]}
    """

    order = [int(v) for v in order]

    N = len(order)
    EMPTY = -(N+1)

    ptr = np.zeros(N, dtype='int') + EMPTY

    N1_values = []
    N2_values = []
    meanS_values = []
    num_values = []
    denom_values = []
    N1 = 1
    for i in range(N):
        r1 = s1 = order[i]
        ptr[s1] = -1
        for s2 in adjlist[s1]:
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

        _metrics = compute_metrics(ptr)
        N1_values.append(_metrics['N1'])
        N2_values.append(_metrics['N2'])
        meanS_values.append(_metrics['meanS'])
        num_values.append(_metrics['num'])
        denom_values.append(_metrics['denom'])

    metrics: Dict[str, Iterable[Any]] = {}
    metrics['p'] = np.arange(N) / N
    metrics['N1'] = N1_values
    metrics['N2'] = N2_values
    metrics['meanS'] = meanS_values
    metrics['num'] = num_values
    metrics['denom'] = denom_values
    return metrics


def percolate_fast(
    adjlist: Iterable[Set[int]], 
    order: Iterable[int]
) -> Dict[str, Iterable[Any]]:
    """
    Newman-Ziff algorithm for percolation.
    Adds nodes in a specific order and computes
    giant component and other metrics on-the-fly.
    Complexity O(N)

    Arguments:
        adjlist {list} -- adjacency list
        order {list} -- order in which nodes are removed

    Returns:
        metrics {dict} -- dictionary with the computed metrics

    >>> # Empty graph
    >>> adjlist = []
    >>> order = []
    >>> percolate_fast(adjlist, order)
    {'p': array([], dtype=float64), 'N1': [], 'meanS': [], 'num': [], 'denom': []}
    >>> # One isolated node
    >>> adjlist = [{}]
    >>> order = [0]
    >>> metrics = percolate_fast(adjlist, order)
    >>> {metric: [round(value, 2) for value in values] for metric, values in metrics.items()}
    {'p': [0.0], 'N1': [1], 'meanS': [1.0], 'num': [0], 'denom': [0]}
    >>> # P3 graph with Degree based attack
    >>> adjlist = [{1}, {0, 2}, {1}]
    >>> order = [1, 0, 2][::-1]
    >>> metrics = percolate_fast(adjlist, order)
    >>> {metric: [round(value, 2) for value in values] for metric, values in metrics.items()}
    {'p': [0.0, 0.33, 0.67], 'N1': [1, 1, 3], 'meanS': [1.0, 1.0, 1.0], 'num': [0, 1, 0], 'denom': [0, 1, 0]}
    >>> # P3 graph with ordered attack
    >>> adjlist = [{1}, {0, 2}, {1}]
    >>> order = [0, 1, 2][::-1]
    >>> metrics = percolate_fast(adjlist, order)
    >>> {metric: [round(value, 2) for value in values] for metric, values in metrics.items()}
    {'p': [0.0, 0.33, 0.67], 'N1': [1, 2, 3], 'meanS': [1.0, 1.0, 1.0], 'num': [0, 0, 0], 'denom': [0, 0, 0]}
    >>> # Works with np.array
    >>> order = np.array([0., 1., 2.][::-1])
    >>> metrics = percolate_fast(adjlist, order)
    >>> {metric: [round(value, 2) for value in values] for metric, values in metrics.items()}
    {'p': [0.0, 0.33, 0.67], 'N1': [1, 2, 3], 'meanS': [1.0, 1.0, 1.0], 'num': [0, 0, 0], 'denom': [0, 0, 0]}
    >>> adjlist = [{1, 3}, {0, 2, 3}, {1, 3}, {0, 1, 2, 4, 5}, {3, 5}, {3, 4}]
    >>> order = [3, 2, 1, 0, 4, 5][::-1]
    >>> metrics = percolate_fast(adjlist, order)
    >>> metrics = {metric: [round(value, 2) for value in values] for metric, values in metrics.items()}
    >>> metrics
    {'p': [0.0, 0.17, 0.33, 0.5, 0.67, 0.83], 'N1': [1, 2, 2, 2, 3, 6], 'meanS': [1.0, 1.0, 1.0, 2.0, 2.0, 1.0], 'num': [0, 0, 1, 4, 4, 0], 'denom': [0, 0, 1, 2, 2, 0]}
    >>> adjlist = [{3}, {2, 6}, {1, 4}, {0}, {2}, {}, {1}]
    >>> order = [0, 1, 2, 3, 4, 5, 6]
    >>> metrics = percolate_fast(adjlist, order)
    >>> metrics = {metric: [round(value, 2) for value in values] for metric, values in metrics.items()}
    >>> metrics
    {'p': [0.0, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86], 'N1': [1, 1, 2, 2, 3, 3, 4], 'meanS': [1.0, 1.0, 1.0, 2.0, 2.0, 1.67, 1.67], 'num': [0, 1, 1, 4, 4, 5, 5], 'denom': [0, 1, 1, 2, 2, 3, 3]}
    """

    order = [int(v) for v in order]

    N = len(order)
    EMPTY = -(N+1)

    ptr = np.zeros(N, dtype='int') + EMPTY

    N1_values = []
    meanS_values = []
    num_values = []
    denom_values = []
    N1 = 1
    num = 0
    denom = 0
    n_comps = 0
    for i in range(N):
        r1 = s1 = order[i]
        ptr[s1] = -1
        n_comps += 1
        num += 1
        denom += 1
        for s2 in adjlist[s1]:
            overpass = False
            new_gcc = False

            if ptr[s2] != EMPTY:
                r2 = findroot(ptr, s2)
                if r2 != r1:
                    n_comps -= 1
                    if ptr[r1] > ptr[r2]: ## s2 belongs to a greater component than s1
                        large = -ptr[r2]
                        small = -ptr[r1]
                        ptr[r2] += ptr[r1] ## Merge s1 to s2
                        ptr[r1] = r2
                        r1 = r2
                    else:
                        large = -ptr[r1]
                        small = -ptr[r2]
                        ptr[r1] += ptr[r2]
                        ptr[r2] = r1

                    ## New GCC
                    if -ptr[r1] > N1:
                        new_gcc = True
                        if large < N1:
                            overpass = True
                        prev_N1 = N1
                        N1 = -ptr[r1]

                    if new_gcc:
                        if overpass:
                            num = (
                                num - small*small - large*large 
                                + prev_N1*prev_N1
                            )
                            denom = denom - small - large + prev_N1
                        else:
                            num = num - small*small
                            denom = denom - small
                    else:
                        num = (
                            num - small*small - large*large 
                            + (small+large)*(small+large)
                        )

        if denom == 0:
            meanS = 0.0
        else:
            meanS = num/denom

        if n_comps == 1:
            num = denom = meanS = 0

        if meanS == 0:
            meanS = 1.0

        N1_values.append(N1)
        meanS_values.append(float(meanS))
        num_values.append(num)
        denom_values.append(denom)

    metrics: Dict[str, Iterable[Any]] = {}
    metrics['p'] = np.arange(N) / N
    metrics['N1'] = N1_values
    metrics['meanS'] = meanS_values
    metrics['num'] = num_values
    metrics['denom'] = denom_values
    return metrics

def percolate_heap(
    adjlist: Iterable[Set[int]], 
    order: Iterable[int]
) -> Dict[str, Iterable[Any]]:
    """
    Newman-Ziff algorithm for percolation.
    Adds nodes in a specific order and computes
    giant component and other metrics on-the-fly.
    Uses min-heap to compute N2
    Complexity O(N log(N)) (?)

    Arguments:
        adjlist {list} -- adjacency list
        order {list} -- order in which nodes are removed

    Returns:
        metrics {dict} -- dictionary with the computed metrics

    """
    
    heap = []
    heapq.heapify()
    order = [int(v) for v in order]

    N = len(order)
    EMPTY = -(N+1)

    ptr = np.zeros(N, dtype='int') + EMPTY

    N1_values = []
    meanS_values = []
    num_values = []
    denom_values = []
    N1 = 1
    num = 0
    denom = 0
    n_comps = 0
    for i in range(N):
        r1 = s1 = order[i]
        ptr[s1] = -1
        n_comps += 1
        num += 1
        denom += 1
        for s2 in adjlist[s1]:
            overpass = False
            new_gcc = False

            if ptr[s2] != EMPTY:
                r2 = findroot(ptr, s2)
                if r2 != r1:
                    n_comps -= 1
                    if ptr[r1] > ptr[r2]: ## s2 belongs to a greater component than s1
                        large = -ptr[r2]
                        small = -ptr[r1]
                        ptr[r2] += ptr[r1] ## Merge s1 to s2
                        ptr[r1] = r2
                        r1 = r2
                    else:
                        large = -ptr[r1]
                        small = -ptr[r2]
                        ptr[r1] += ptr[r2]
                        ptr[r2] = r1

                    ## New GCC
                    if -ptr[r1] > N1:
                        new_gcc = True
                        if large < N1:
                            overpass = True
                        prev_N1 = N1
                        N1 = -ptr[r1]

                    if new_gcc:
                        if overpass:
                            num = (
                                num - small*small - large*large 
                                + prev_N1*prev_N1
                            )
                            denom = denom - small - large + prev_N1
                        else:
                            num = num - small*small
                            denom = denom - small
                    else:
                        num = (
                            num - small*small - large*large 
                            + (small+large)*(small+large)
                        )

        if denom == 0:
            meanS = 0.0
        else:
            meanS = num/denom

        if n_comps == 1:
            num = denom = meanS = 0

        if meanS == 0:
            meanS = 1.0

        N1_values.append(N1)
        meanS_values.append(float(meanS))
        num_values.append(num)
        denom_values.append(denom)

    metrics: Dict[str, Iterable[Any]] = {}
    metrics['p'] = np.arange(N) / N
    metrics['N1'] = N1_values
    metrics['meanS'] = meanS_values
    metrics['num'] = num_values
    metrics['denom'] = denom_values
    return metrics

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    # P5
    adjlist = [{1}, {0, 2}, {1, 3}, {2, 4}, {3}]
    order = [2, 1, 0, 3, 4][::-1]
    metrics = percolate_fast(adjlist, order)
    print({metric: [round(value, 2) for value in values] for metric, values in metrics.items()})

    metrics = percolate_slow(adjlist, order)
    print({metric: [round(value, 2) for value in values] for metric, values in metrics.items() if metric != 'N2'})

    import igraph as ig
    import matplotlib.pyplot as plt
    N = 10000
    k = 3.5
    p = k / N
    g = ig.Graph().Erdos_Renyi(N, p)
    adjlist = ig_graph_to_adjlist(g)
    order = range(N)

    metrics_fast = percolate_fast(adjlist, order)
    #print({metric: [round(value, 2) for value in values] for metric, values in metrics.items()})

    metrics_slow = percolate_slow(adjlist, order)
    print({metric: [round(value, 2) for value in values] for metric, values in metrics.items() if metric != 'N2'})

    for i, metrics in enumerate([metrics_slow, metrics_fast]):
        label = 'fast' if i==1 else 'slow'
        marker = '*' if i == 0 else 's'
        plt.plot(metrics['p'], metrics['meanS'], marker, fillstyle='none', label=label)

    plt.show()
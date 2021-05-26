import numpy as np


def get_box_count(positions, indices):

    N = len(positions)
    L = int(np.sqrt(N))

    l_values = []
    box_count = []

    l = L
    i = 0
    while l >= 2:
        l_values.append(l)
        box_number_set = set([])
        for idx in indices:
            x, y = positions[idx]
            box_number = (y//l) * (L/l) + x//l
            box_number_set.add(box_number)
        box_count.append(len(box_number_set))
        l //= 2
        i += 1

    return box_count, l_values

def get_nodes_in_centered_window(positions, x, y, l):

    nodes_in_window = []
    for i, (xi, yi) in enumerate(positions):

        if (x - l/2 <= xi < x + l/2):
            if (y - l/2 <= yi < y + l/2):
                nodes_in_window.append(i)

    return set(nodes_in_window)

def get_local_mass(positions, x, y, l):
    return (len(get_nodes_in_centered_window(positions, x, y, l)) - 0)

def get_cluster_densities(positions, indices, l_values, seeds=None):

    N = len(positions)
    positions = np.array(positions)
    if not seeds:
        seeds = int(np.sqrt(N))

    rho_values = np.zeros((len(l_values),seeds))
    for i, l in enumerate(l_values):
        for j in range(seeds):
            idx = np.random.choice(indices)
            x, y = positions[idx]
            rho_values[i,j] = get_local_mass(positions[indices], x, y, l) / l**2

    return rho_values.mean(axis=1)
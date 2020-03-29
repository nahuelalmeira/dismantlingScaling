import numpy as np
import scipy as sp
import igraph as ig
import scipy.spatial
import scipy.sparse.csgraph

def distance(s, t, points=None):

    points = np.array(points)

    if points.any():
        x1, y1 = points[s,:]
        x2, y2 = points[t,:]
    else:
        x1, y1 = s
        x2, y2 = t

    return np.sqrt((x2-x1)**2+(y2-y1)**2)

def create_points(N, d=2, random_seed=None):
    np.random.seed(random_seed)
    return np.random.random((N, d))

def create_dt_edgelist(points):

    points = np.array(points)

    tri = sp.spatial.Delaunay(points)

    edgeset = set([])
    for t in tri.simplices:
        u, v, w = t
        if u < v:
            edgeset.add((u,v))
        else:
            edgeset.add((v,u))
        if v < w:
            edgeset.add((v,w))
        else:
            edgeset.add((w,v))
        if u < w:
            edgeset.add((u,w))
        else:
            edgeset.add((w,u))

    edgelist = list(edgeset)

    return edgelist

def get_mst(edgelist, points, distances=None):
    N = np.max(edgelist)
    edgelist = np.concatenate((np.array(edgelist), np.flip(np.array(edgelist), axis=1)))

    nodes, indices, inverse = np.unique(edgelist, return_index=True, return_inverse=True)
    rows, cols = inverse.reshape(edgelist.shape).T

    if not distances:
        distances = []
        for s, t in zip(rows, cols):
            d = distance(s, t, points)
            distances.append(d)

    matrix = sp.sparse.coo_matrix((distances, (rows,cols)))

    rows, cols = np.where(sp.sparse.csgraph.minimum_spanning_tree(matrix).toarray())
    mst = np.zeros((N, 2), int)
    mst[:,0] = rows
    mst[:,1] = cols

    assert(np.max(edgelist) == len(mst))

    return mst

def get_cell(point, N):
    """
    Returns the cell to which point belongs.
    Each cell is an LxL square, where L = sqrt(N)
    """
    L = np.sqrt(N)
    x, y = point
    cell = (int(x*L), int(y*L))
    return cell

def build_cell_matrix(points):

    N = len(points)
    L = np.sqrt(N)

    cell_matrix = []
    for i in range(int(L)+1):
        cell_matrix.append([])
        for j in range(int(L)+1):
            cell_matrix[i].append(set([]))

    for v, point in enumerate(points):
        i, j = get_cell(point, N)
        cell_matrix[i][j].add(v)

    return cell_matrix

def create_rn_edgelist(points):

    points = np.array(points)

    cell_matrix = build_cell_matrix(points)

    N = len(points)
    L = np.sqrt(N)

    dt_edgelist = create_dt_edgelist(points)

    edges_to_delete = []
    for e in dt_edgelist:
        s, t = e

        d = distance(s, t, points)
        d_cell = int(d*L) + 2

        s_cell = get_cell(points[s], N)

        i_min = max(s_cell[0]-d_cell, 0)
        i_max = min(s_cell[0]+d_cell, len(cell_matrix))

        j_min = max(s_cell[1]-d_cell, 0)
        j_max = min(s_cell[1]+d_cell, len(cell_matrix))

        i = i_min
        j = j_min
        delete = False
        while (i < i_max) and not delete:

            for r in cell_matrix[i][j]:
                if distance(s, r, points) < d and distance(t, r, points) < d:
                    edges_to_delete.append(e)
                    delete = True

            j += 1
            if j == j_max:
                j = j_min
                i += 1

    rn_edgelist = list(set(dt_edgelist).difference(set(edges_to_delete)))

    return rn_edgelist, dt_edgelist

def get_middle_point(points, s, t):
    return (points[s] + points[t]) / 2

def create_gabriel_edgelist(points):

    points = np.array(points)
    cell_matrix = build_cell_matrix(points)

    N = len(points)
    L = np.sqrt(N)

    dt_edgelist = create_dt_edgelist(points)

    edges_to_delete = []
    for e in dt_edgelist:
        s, t = e

        m = (points[s] + points[t]) / 2

        d = distance(s, t, points)
        d_cell = int(d*L) + 2

        s_cell = get_cell(points[s], N)

        i_min = max(s_cell[0]-d_cell, 0)
        i_max = min(s_cell[0]+d_cell, len(cell_matrix))

        j_min = max(s_cell[1]-d_cell, 0)
        j_max = min(s_cell[1]+d_cell, len(cell_matrix))

        i = i_min
        j = j_min
        delete = False
        while (i < i_max) and not delete:

            for r in cell_matrix[i][j]:
                dist_to_m = np.sqrt((m[0] - points[r][0])**2 + (m[1] - points[r][1])**2)
                if dist_to_m < d/2 and r not in [s,t]:
                    edges_to_delete.append(e)
                    delete = True

            j += 1
            if j == j_max:
                j = j_min
                i += 1

    gabriel_edgelist = list(set(dt_edgelist).difference(set(edges_to_delete)))

    return gabriel_edgelist, dt_edgelist

def get_r_from_mst(points):

    dt_edgelist = create_dt_edgelist(points)
    mst_edgelist = get_mst(dt_edgelist, points)

    distances = []
    for edge in mst_edgelist:
        s, t = edge
        #if s < t:
        if True:
            d = distance(s, t, points)
            distances.append(d)

    r = np.max(distances)
    return r


def create_mr_edgelist(points, r=None, distances=None):

    N = len(points)
    L = np.sqrt(N)
    cell_matrix = build_cell_matrix(points)

    if not r:
        r = get_r_from_mst(points)

    d_cell = int(r*L) + 2

    edgelist = []
    for s in range(N):

        s_cell = get_cell(points[s], N)

        i_min = max(s_cell[0]-d_cell, 0)
        i_max = min(s_cell[0]+d_cell, len(cell_matrix))

        j_min = max(s_cell[1]-d_cell, 0)
        j_max = min(s_cell[1]+d_cell, len(cell_matrix))

        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                for t in cell_matrix[i][j]:
                    if s < t and distance(s, t, points) <= r:
                        edgelist.append((s, t))

    return edgelist

def create_proximity_graph(model, N=None, points=None, r=None, distances=None,
                           random_seed=None):

    if N:
        points = create_points(N, random_seed=random_seed)
    else:
        points = np.array(points)
        N = len(points)

    if model == 'MR':
        edgelist = create_mr_edgelist(points, r, distances)
    elif model == 'DT':
        edgelist = create_dt_edgelist(points)
    elif model == 'RN':
        edgelist, _ = create_rn_edgelist(points)
    elif model == 'GG':
        edgelist, _ = create_gabriel_edgelist(points)

    G = ig.Graph()
    G.add_vertices(N)
    G.add_edges(edgelist)
    G.vs['position'] = points.tolist()

    return G
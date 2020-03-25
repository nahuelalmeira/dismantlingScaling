import numpy as np
import scipy as sp
import scipy.spatial 
import scipy.sparse.csgraph

def distance(points, s, t):
    x1, y1 = points[s,:]
    x2, y2 = points[t,:]
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

def create_points(N, d=2):
    return np.random.random((N, d))

def create_dt_graph(points):
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
    
    return edgelist, tri
    
def get_mst(edgelist, points, distances=None):
    N = np.max(edgelist)
    edgelist = np.concatenate((np.array(edgelist), np.flip(np.array(edgelist), axis=1)))

    nodes, indices, inverse = np.unique(edgelist, return_index=True, return_inverse=True)
    rows, cols = inverse.reshape(edgelist.shape).T
    
    if not distances:
        distances = []
        for s, t in zip(rows, cols):
            d = distance(points, s, t)
            distances.append(d)
        
    matrix = sp.sparse.coo_matrix((distances, (rows,cols)))
    
    rows, cols = np.where(sp.sparse.csgraph.minimum_spanning_tree(matrix).toarray())
    mst = np.zeros((N, 2), int)
    mst[:,0] = rows
    mst[:,1] = cols
    
    assert(np.max(edgelist) == len(mst))
    
    return mst


def create_mr_graph_naive(points, r, distances=None):
    N = len(points)
    edgelist = []
    for s in range(N):
        for t in range(s+1, N):
            if distance(points, s, t) <= r:
                edgelist.append((s, t))
    
    return edgelist

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

def create_mr_graph(points, r, distances=None):
    
    cell_matrix = build_cell_matrix(points)
    
    N = len(points)
    L = np.sqrt(N)
    
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
                    if s < t and distance(points, s, t) <= r:
                        edgelist.append((s, t))

    return edgelist
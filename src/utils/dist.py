import numpy as np
from numba import jit
from rich.progress import track

@jit(nopython=True, fastmath=True)
def cal_distance_matrix(X: np.ndarray, Y: np.ndarray):
    x2 = np.sum(X**2, axis=1) # shape of (m)
    y2 = np.sum(Y**2, axis=1) # shape of (n)
    xy = X.dot(Y.T)
    x2 = x2.reshape(-1, 1)
    dists = np.sqrt(x2 - 2*xy + y2)
    
    return dists

def init_cluster_centers(nClusters: int):
    
    thetas = np.linspace(0, np.pi, 500)
    phis = np.linspace(0, 2 * np.pi, 500)[:-1]
    points = np.array([sph_to_cart(1, theta, phi) for theta in thetas for phi in phis])
    dists = np.zeros((nClusters, len(points)))
    iClusters = 0
    selPoints = [points[0]]
    pointPool = np.full(len(points), True)
    for iClusters in track(range(nClusters - 1), description='Initializing cluster centers...'):
        delPoints = ~np.alltrue(selPoints[-1] == points, axis=1)
        pointPool &= delPoints
        dists[iClusters] = np.linalg.norm(points - selPoints[-1], axis=1)
        dists[iClusters, ~pointPool] = np.nan
        selPointsIdx = np.nanargmax(np.sum(dists, axis=0))
        selPoints.append(points[selPointsIdx])
        iClusters += 1
        
    return np.array(selPoints)

def sph_to_cart(r, theta, phi):
    
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    
    return (x, y, z)
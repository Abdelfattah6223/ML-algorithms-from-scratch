## Dimensionality reduction algorithms (MDS , isomap)

import numpy as np
from scipy.spatial.distance import cdist


# MDS
def X_centree(X):
    M = np.mean(X, axis=0)
    return X-M
def distance_matrix(X):
    return cdist(X,X,'euclidean')

def MDS(D,newdim=2):  # D = distance matrix of centred data

    n = D.shape[0]
    # Torgerson formula
    I = np.identity(n)
    J = np.ones(n)
    F = I-(1/n)*J
    B = (- 1/2 )* F @ (D ** 2) @ F  # or B = (-1/2)*np.dot(np.dot(F,D**2),F) ( ie : B = -(1/2).FD²F )

    # Vecteurs & valeurs propres de B
    eigenval, eigenvec = np.linalg.eig(B)
    indices = np.argsort(eigenval)[::-1]   # to sort eigenvalues & their corresponding eigenvectors
    eigenval = eigenval[indices]
    eigenvec = eigenvec[:, indices]

    # dimension reduction
    K = eigenvec[:, :newdim]
    L = np.diag(eigenval[:newdim])  # choose the largest 'newdim' eigenvalues
    # resultat
    Y = K @ L ** (1 / 2)    # or Y = np.dot(K,L**(1/2))
    return np.real(Y)

# Finding the shortest path
def Floyd(h):
    ndata = h.shape[0]
    for k in range(ndata):
        for i in range(ndata):
            for j in range(ndata):
                if h[i,j] > h[i,k] + h[k,j]:
                    h[i,j] = h[i,k] + h[k,j]
    return h

def Dijkstra(h):
    q = h.copy()
    for i in range(ndata):
        for j in range(ndata):
            k = np.argmin(q[i,:])
            while not(np.isinf(q[i,k])):
                q[i,k] = np.inf
                for l in neighbours[k,:]:
                    possible = h[i,l] + h[l,k]
                    if possible < h[i,k]:
                        h[i,k] = possible
                k = np.argmin(q[i,:])
    return h

def isomap(X,epsilon,d):

    nobs = np.shape(X)[0]  # observations number
    ndim = np.shape(X)[1]  # Dimension (features number)

    # Calculate the (euclidean) distance matrix
    D = distance_matrix(X)

    # using epsilon radius to obtain neighbours

    h = np.where(np.less(D, epsilon), D, np.inf)   # if two points aren't neighbours then the weight of their egde fixed in "inf"



    h = Floyd(h)   # estimate geodesic distance matrix

    return MDS(h,d)  # Applying MDS on h

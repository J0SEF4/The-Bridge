#Manifold-Manifold Distance Code (Paper: "Manifold–Manifold Distance and Its Application to Face Recognition With Image Sets", Wang et al. 2012))

#SETUP
import numpy as np
from typing import List, Tuple

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

EPS = 1e-10 # small constant to avoid division by zero
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#ISOMAP ALGORITHM (Paper: "Global Geometric Framework for Nonlinear Dimensionality Reduction", Tenenbaum et al. 2000)

#K-NN GRAPH (1. Construct neightborhood graph)
def build_knn_graph(X: np.ndarray, k: int = 10, metric: str = 'euclidean') -> csr_matrix: 
    # X is (n,D), n points in D dimensions, k neighbors, metric for distance euclidean, csr_matrix is sparse matrix
    #use k=10 because it is a good balance between local and global structure (DREAMS paper)
    n = X.shape[0]
    
    #'i is one of the K nearest neighbors of j (K-Isomap)' - found k neighbors more close to each point
    neighbors = NearestNeighbors(n_neighbors=min(k + 1, n), metric=metric).fit(X)
    distances, indexs = neighbors.kneighbors(X)
    #take out 1st neighbor. because it is the point itself (distance 0)
    distances = distances[:, 1:]                        #(n, k)
    indexs = indexs[:, 1:]                              #(n, k)
    #matrix of connectivity G, row origin to column destination neighbors, data is the distance between them
    rows = np.repeat(np.arange(n), indexs.shape[1])     # arrange(n) create each point index, indexs.shape[1] each point have 2 neighbors, repeat create row for each neighbor
    cols = indexs.flatten()                             # flatten create column for each neighbor
    
    #'set edge lengths equal to dX(i,j)'
    data = distances.flatten()
    G = csr_matrix((data, (rows, cols)), shape=(n, n))  # G[i,j] = distance if j is neighbor of i, 0 otherwise
    G =(G + G.T)/2                                      #symmetric matrix to ensure undirected graph, is the average of both directions so they have same weight
    return G

#GEODESIC DISTANCES (2. Compute shortest paths) 
def geodesic_distance_matrix(X: np.ndarray, k: int = 10) -> np.ndarray:
    G = build_knn_graph(X, k=k)                         #'Initialize dG(i,j) = dX(i,j) if linked'
    D = shortest_path(G, method='FW', directed=False)   #'Then for each value of k =1,2,...,N in turn, replace all entries dG(i, j) by min{dG(i, j),dG(i,k)+dG(k, j)}.' - this is Floyd-Warshall algorithm(https://www.geeksforgeeks.org/dsa/floyd-warshall-algorithm-dp-16/)
    return D                                            #'shortest path distances between all pairs of points in G' 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#BETA no linear
def estimate_beta(X: np.ndarray, k: int = 10) -> float:
    n = X.shape[0]
    if n < 2: # there is no, no linear structure
        return 1.0
    #'euclidean pairwise distances D_E'=||x_p - x_q|| 
    De = pairwise_distances(X, metric='euclidean')
    #'geodesic distance matrix D_G'
    Dg = geodesic_distance_matrix(X, k=k)

    # use upper triangle pairs where De > EPS and Dg finite
    valid = (De > EPS) & np.isfinite(Dg) #avoid division by zero and invalid distances (exclude disconnected pairs)
    if not np.any(valid):
        return 1.0

    #B^(i) = (1/N_i^2) * sum(R(x_p, x_q))= mean(R(x_p^(i) , x_q^(i))) = mean(Dg(x_p, x_q) / De(x_p, x_q))
    ratios = (Dg[valid] / (De[valid] + EPS)) 
    return float(np.mean(ratios))
# B=1, lineal cluster
# B>1, no lineal cluster, B=1.5 Shape of "S" cluster, B=3 "Swiss roll" cluster

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#ALGORITHM 1, Linearity-Constrained HDC (L-HDC)
def recursive_split(X: np.ndarray, delta_thresh: float = 1.1, k: int = 10) -> List[np.ndarray]: #X is (n,D), beta_thresh: threshold for beta, min_size: minimum cluster size, k: neighbors for k-NN graph, list of patches (arrays of indices into X) representing MLPs with β <= beta_thresh
    # delta_thresh: threshold for beta to stop splitting = 1.1 (slightly non-linear it starts from >1)

    n = X.shape[0]

    # "1. Initialization: X(1) = {x1, x2, ..., xN}, m = 1"
    X_full_indices = np.arange(n)               
    # "1. ....Compute the nonlinearity score B(1) according to (8)"
    beta_full = estimate_beta(X, k=k)           
    queue = [(X_full_indices, beta_full)] #list of (cluster_indices, beta_score)
    final_patches = []

    #matrix H used in 2.3.1, 2.3.2 -> H: k×N, each column contains k nearest neighbor indices for that point
    neighbors = NearestNeighbors(n_neighbors=min(k + 1, n), metric='euclidean').fit(X) #neighbors it is a NearestNeighbors object so then it can be used to find neighbors of points
    _, H = neighbors.kneighbors(X)  #get k+1 neighbors (including self) (n,k+1)
    H = H[:, 1:]                    #discard 1st column (auto neighbor)"..,each column H(:, j) (j = 1,..., N) holding the indices of k nearest neighbors of the data point x_j"   
    H = H.T                         #transpose to (k, n): so each column j, H[:, j] = k neighbors of point x_j
    
    
    while queue:
        # "2. Choose X(i) with the largest score B(i) as the parent cluster. Split X(i) as follows"
        idx_max = np.argmax([beta for _, beta in queue]) 
        X_i_indices, beta_i = queue.pop(idx_max)
        
        # "3. The splitting procedure continues until the nonlinearity score B(i) < threshold DELTA"
        if beta_i < delta_thresh: # If cluster too small or sufficiently linear, accept as final patch
            final_patches.append(X_i_indices)
            continue
        
        # "2.1. According to geodesic distance Dg, select two furthest seed points xl and xr"
        X_sub = X[X_i_indices]
        Dg = geodesic_distance_matrix(X_sub, k=k) #geodesic distance matrix for this cluster
        max_dist = -1 #2 point with max geodesic distance (two furthest points = diameter of manifold in geodesic sense)
        seed_l_local = 0
        seed_r_local = 0
        for i in range(len(X_i_indices)):
            for j in range(i + 1, len(X_i_indices)):
                if np.isfinite(Dg[i, j]) and Dg[i, j] > max_dist: #isfinite to avoid disconnected points
                    max_dist = Dg[i, j]
                    seed_l_local = i
                    seed_r_local = j        
        seed_l_global = X_i_indices[seed_l_local] #local to global indices
        seed_r_global = X_i_indices[seed_r_local]
        
        # "2.2. Initialize two child clusters: X(i)_l = {xl}, X(i)_r = {xr}. Update: X(i) <- X(i)\{xl, xr}"
        X_i_l = {seed_l_global}
        X_i_r = {seed_r_global}
        X_i_remaining = set(X_i_indices) - {seed_l_global, seed_r_global}
        
        # "2.3. while (X(i) different to null) do" - Expand clusters via k-NN neighbors
        while X_i_remaining:
            # "2.3.1 For current X(i)_l, construct its neighbor points set, denote by Pl.According to H,Pl gathers the k-NN samples of all the points in X(i)_l"
            P_l = set()
            for point_idx in X_i_l:
                neighbors_of_point = set(H[:, point_idx]) # H[:, point_idx] gives k nearest neighbors of point_idx in global space
                P_l.update(neighbors_of_point & X_i_remaining) #gather the k-NN samples of all the points in X(i)_l from the remaining pool X(i)
            
            # "2.3.2 For current X(i)_r, construct its neighbor points set Pr, in the similar way to step 2.3.1"        
            P_r = set()
            for point_idx in X_i_r:
                neighbors_of_point = set(H[:, point_idx])
                P_r.update(neighbors_of_point & X_i_remaining)
            
            # "2.3.3: Sequentially update:"
            # "X(i)_l <- X(i)_l ∪ (Pl ∩ X(i)), X(i) <- X(i) \ (Pl ∩ X(i))"
            points_to_l = P_l & X_i_remaining
            X_i_l.update(points_to_l)
            X_i_remaining -= points_to_l
            
            # "X(i)_r <- X(i)_r ∪ (Pr ∩ X(i)), X(i) <- X(i) \ (Pr ∩ X(i))"
            points_to_r = P_r & X_i_remaining
            X_i_r.update(points_to_r)
            X_i_remaining -= points_to_r
            
            if not points_to_l and not points_to_r: #Warning: No progress in cluster expansion, to avoid infinite loop assign remaining points
                # No progress: assign remaining points to the closest cluster by Euclidean distance.
                if X_i_remaining:
                    centroid_l = X[list(X_i_l)].mean(axis=0)
                    centroid_r = X[list(X_i_r)].mean(axis=0)
                    for point_idx in list(X_i_remaining):
                        dist_to_l = np.linalg.norm(X[point_idx] - centroid_l)
                        dist_to_r = np.linalg.norm(X[point_idx] - centroid_r)
                        if dist_to_l <= dist_to_r:
                            X_i_l.add(point_idx)
                        else:
                            X_i_r.add(point_idx)
                    X_i_remaining.clear()
                break
        
        # "2.4. X(i) is split into two smaller ones: X(i)_l and X(i)_r"
        # "Update: m <- m + 1, compute β(i)_l and β(i)_r"
        X_i_l_indices = np.array(sorted(X_i_l))
        X_i_r_indices = np.array(sorted(X_i_r))
        if len(X_i_l_indices) > 0:
            beta_l = estimate_beta(X[X_i_l_indices], k=k)
            queue.append((X_i_l_indices, beta_l))
        if len(X_i_r_indices) > 0:
            beta_r = estimate_beta(X[X_i_r_indices], k=k)
            queue.append((X_i_r_indices, beta_r))
    
    return final_patches

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#PATCH REPRESENTATION (e_i, P_i) via PCA p. 4470

def compute_patch_representation(X: np.ndarray, indices: np.ndarray, var_threshold: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    #var_threshold: 95% so it capture the 1st d eigenvectors that explain 95% of variance
    
    Xp = X[indices]                         # extract patch samples
    # "For each local model Ci, we denote its sample mean (i.e., exemplar) by ei"
    e = np.mean(Xp, axis=0) 
    e_norm = e / (np.linalg.norm(e) + EPS)  # normalize to unit vector
    Xc = Xp - e                             # centered data, xp is the samples in the patch - e is the mean of the patch
    
    if Xc.shape[0] <= 1:                    # not enough samples to compute PCA
        P = np.zeros((X.shape[1], 0))       # empty principal component matrix
        return e_norm, P
    
    # "and corresponding principal component matrix by Pi ∈ R^D×di that is computed as the leading eigenvectors of the covariance matrix and forms a set of orthonormal basis of the subspace."
    pca = PCA(n_components=min(Xc.shape[0], Xc.shape[1]))  #xc.shape[0] number of samples in patch, Xc.shape[1] dimension of data / min to avoid requesting more components than samples or dimensions
    pca.fit(Xc)                                             # fit PCA on centered data, has the eigenvectors(PC) and eigenvalues of Xc covariance matrix
    cumvar = np.cumsum(pca.explained_variance_ratio_)       # cumulative variance explained by each principal component
    d = int(np.searchsorted(cumvar, var_threshold) + 1)     # number of components to reach var_threshold, +1 to have the count of components from 1
    d = min(d, pca.components_.shape[0])                    # ensure d does not exceed available components
    P = pca.components_[:d].T                               # PC as columns (D, d)
    return e_norm, P                                        # "subspace (or local model) is spanned by a set of samples, ei(describe center patch) and Pi(variance directions)"

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#SSD DISTANCE BETWEEN TWO PATCHES - B) Local Model Distance Measure (p.4470-4471)
def ssd_distance(patchA: Tuple[np.ndarray, np.ndarray], patchB: Tuple[np.ndarray, np.ndarray]) -> float:
    #SSD: Subspace-to-Subspace Distance between two patches, eq16
    #patchA, patchB: (e, P) where e is normalized exemplar, P is principal component matrix, returns d_ssd (float)

    eA, PA = patchA
    eB, PB = patchB
    
    # EXEMPLAR DISTANCE MEASURE(Eq. 15, p.4471):
    cos0 = float(np.dot(eA, eB))            #||eA||= ||eB|| = 1, already normalized
    sin2_0 = 1.0 - (cos0**2)               
    
    #if either subspace has no principal components (d=0)
    if PA.shape[1] == 0 or PB.shape[1] == 0:
        return float(np.sqrt(sin2_0))       # d_ssd = sin θ (exemplar distance only)
    
    # VARIATION DISTANCE via Principal Angles (Eq. 10, p.4470):
    M = PA.T @ PB                         
    
    U, S, Vt = np.linalg.svd(M, full_matrices=False) #SVD of M, S contains singular values σ
    
    sigmas = np.clip(S, -1.0, 1.0)          # (Eq. 11) σ = cos θ,-1-1 because it is cosine values
    sin2 = 1.0 - (sigmas ** 2)              # (Eq. 13) Min correlation
    r = min(PA.shape[1], PB.shape[1])       # r is min dimension of the two subspaces (write after Eq. 9) here with PA.shape[1] and PB.shape[1] to set how much PC or angles to consider
    
    # PROJECTION METRIC (Eq. 14, p.4471):
    sin2_mean = float(np.sum(sin2[:r]) / (r + EPS))
    
    # FORMAL DEFINITION OF SSD (Eq. 16, p.4471):
    # This fuses exemplar distance (sin2_0) and variation distance (sin2_mean)
    d_ssd = float(np.sqrt(sin2_0 + sin2_mean))
    return d_ssd


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#BUILD PATCH SET
def build_patch_set(X: np.ndarray, indices_list: List[np.ndarray], var_threshold: float = 0.95) -> List[Tuple[np.ndarray, np.ndarray]]:
    #X is (N,D), indices_list is list of arrays of integer indices into X, returns list of patches (e,P)
    #var_threshold: PCA variance threshold
    patches = []
    for inds in indices_list:
        e, P = compute_patch_representation(X, inds, var_threshold=var_threshold)
        patches.append((e, P))
    return patches

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#BUILD DISTANCE MATRIX BETWEEN TWO PATCH SETS
def build_distance_matrix(patchesA: List[Tuple[np.ndarray, np.ndarray]], patchesB: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    m = len(patchesA)                   #m is number of patches in A
    n = len(patchesB)                   #n is number of patches in B
    D = np.zeros((m, n), dtype=float)   #distance matrix (m x n)
    for i in range(m):
        for j in range(n):
            D[i, j] = ssd_distance(patchesA[i], patchesB[j]) #compute d_ssd between patch i in A and patch j in B
    return D

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Option 3: Mean N^4 (NN's NN) - (Eq. 21)
# "the smaller the distance of the pair is, the larger its weight should be" - transfers weight from further pairs to closer ones via NN's NN correspondences
def compute_mmd_from_distance_matrix(D: np.ndarray) -> float:
    m, n = D.shape
    if m == 0 or n == 0:
        return 0.0
    
    distances = []
    #NN's NN = go to the neighbor more close in the other C and come back, this way chose reciprocal matches and less noise in correspondences
    #For each C_i in M1, transfer weight via NN's NN in M1
    for i in range(m):
        # C'_N(i): Find NN of C_i in M2 
        j_N_i = int(np.argmin(D[i, :]))
        # C_N'(N(i)): Find NN of C'_N(i) back in M1 (the NN's NN)
        i_N_prime_N_i = int(np.argmin(D[:, j_N_i]))
        # Use distance d(C_N'(N(i)), C'_N(i)) - transfers weight to closer pair
        distances.append(float(D[i_N_prime_N_i, j_N_i]))
    #For each C'_j in M2, transfer weight via NN's NN in M2  
    for j in range(n):
        # C_N'(j): Find NN of C'_j in M1
        i_N_prime_j = int(np.argmin(D[:, j]))
        # C'_N(N'(j)): Find NN of C_N'(j) back in M2 (the NN's NN)
        j_N_N_prime_j = int(np.argmin(D[i_N_prime_j, :]))
        # Use distance d(C_N'(j), C'_N(N'(j))) - transfers weight to closer pair
        distances.append(float(D[i_N_prime_j, j_N_N_prime_j]))
    
    #final MMD as mean distance across all pairs
    return float(np.mean(distances)) if distances else 0.0

#using option3 but returning detailed info
def compute_mmd_detailed(D: np.ndarray,
                         patchesA: List[Tuple[np.ndarray, np.ndarray]],
                         patchesB: List[Tuple[np.ndarray, np.ndarray]],
                         idsA: List[int] = None,
                         idsB: List[int] = None,
                         labelsA: List[int] = None,
                         labelsB: List[int] = None) -> Tuple[float, dict]:
    m, n = D.shape
    matches = []
    distances = []
    
    #For each C_i in M1, transfer weight via NN's NN in M1
    for i in range(m):
        # C'_N(i): Find NN of C_i in M2
        j_N_i = int(np.argmin(D[i, :]))
        # C_N'(N(i)): Find NN of C'_N(i) back in M1 (the NN's NN)
        i_N_prime_N_i = int(np.argmin(D[:, j_N_i]))
        
        d_ij = float(D[i_N_prime_N_i, j_N_i])
        distances.append(d_ij)
        di = patchesA[i_N_prime_N_i][1].shape[1]
        dj = patchesB[j_N_i][1].shape[1]
        
        matches.append({
            'i': i_N_prime_N_i, # index in patchesA
            'j': j_N_i,         # index in patchesB
            'd_ij': d_ij,       # distance between the matched patches
            'd_i': di,          # dimension of patch i
            'd_j': dj,          # dimension of patch j
            'id_i': idsA[i_N_prime_N_i] if idsA is not None else None,          # id of patch i
            'id_j': idsB[j_N_i] if idsB is not None else None,                  # id of patch j
            'label_i': labelsA[i_N_prime_N_i] if labelsA is not None else None, # label of patch i
            'label_j': labelsB[j_N_i] if labelsB is not None else None,         # label of patch j
        })
    
    #For each C'_j in M2, transfer weight via NN's NN in M2
    for j in range(n):
        # C_N'(j): Find NN of C'_j in M1
        i_N_prime_j = int(np.argmin(D[:, j]))
        # C'_N(N'(j)): Find NN of C_N'(j) back in M2 (the NN's NN)
        j_N_N_prime_j = int(np.argmin(D[i_N_prime_j, :]))
        
        d_ij = float(D[i_N_prime_j, j_N_N_prime_j])
        distances.append(d_ij)
        di = patchesA[i_N_prime_j][1].shape[1]
        dj = patchesB[j_N_N_prime_j][1].shape[1]
        
        matches.append({
            'i': i_N_prime_j,
            'j': j_N_N_prime_j,
            'd_ij': d_ij,
            'd_i': di,
            'd_j': dj,
            'id_i': idsA[i_N_prime_j] if idsA is not None else None,
            'id_j': idsB[j_N_N_prime_j] if idsB is not None else None,
            'label_i': labelsA[i_N_prime_j] if labelsA is not None else None,
            'label_j': labelsB[j_N_N_prime_j] if labelsB is not None else None,
        })
    
    #remove duplicate matches
    uniq = []
    seen = set()
    for mm in matches:
        key = (mm['i'], mm['j'])
        if key not in seen:
            seen.add(key)
            uniq.append(mm)
    
    #final MMD using consistent mean distance (Eq. 21)
    unique_distances = [mm['d_ij'] for mm in uniq]
    d_mmd = float(np.mean(unique_distances)) if unique_distances else 0.0 #final MMD as mean distance across unique pairs. all distances used equally now
    
    info = {
        'matches': uniq,
        'D_matrix': D,
        'patchesA_count': m,
        'patchesB_count': n,
        'mean_pair_dim': float(np.mean([(mm['d_i'] + mm['d_j']) / 2.0 for mm in uniq])) if uniq else 0.0,
        'median_pair_dim': float(np.median([(mm['d_i'] + mm['d_j']) / 2.0 for mm in uniq])) if uniq else 0.0,
    }
    return d_mmd, info
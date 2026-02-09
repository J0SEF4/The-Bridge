#setup
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.ndimage import gaussian_filter1d


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# "a) Graph induced geodesic distance between images is able to capture the topology of the image representation manifold more reliably."
def build_knn_graph(X, k=15, metric='cosine'):
    #K=15 for table 1, metric cosine similarity for ArcFace=SphereFace

    #normalize for ArcFace (they live on hypersphere)
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    
    #"graph induced build with k neighbors"
    #k+1 to include self, then drop self later
    n = X_norm.shape[0]
    n_neighbors = min(k + 1, n)
    if n_neighbors < 2:
        return csr_matrix((n, n))
    
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="brute")
    nn.fit(X_norm) #X_norm are the points on the hypersphere
    _, indices = nn.kneighbors(X_norm)  # kneighbors indices of each point (distances, indices)
    
    #n points = n images
    n = X_norm.shape[0] #"between images"
    rows, cols, data = [], [], [] #"distance r is computed as the shortest path induced by the graph" so this are the nodes, edges, weights to get this"
    
    #build graph between diferent pairs of images(points)
    for i in range(n):
        for j in indices[i, 1:]:  # skip self
            #"distance r is computed as the graph induced shortest path"
            #for ArcFace: d(xi, xj) = arccos(xi^T xj / ||xi|| ||xj||), p. 3993
            cos_sim = np.clip(np.dot(X_norm[i], X_norm[j]), -1.0, 1.0) #arccos only accepts [-1, 1], use clip to avoid numerical issues
            dist = float(np.arccos(cos_sim)) #angular distance between points, here the vectors already normalized so no need to divide by norms
            
            #undirected graph
            #"graphs edges for th surface of a unitary hypersphere and a face manifold"
            #it is bidirectional (i to j and j to i), because distance is symmetric
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([dist, dist])
    
    graph = csr_matrix((data, (rows, cols)), shape=(n, n)) #graph as sparse matrix where (data, (rows, cols)) are the edges and weights(angular distances) 
    return graph

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#3.1 "distance r between any pair of points is computed as the shortest path between the points as induced by the graph"
def compute_geodesic_distances(graph):
    return shortest_path(graph, directed=False, unweighted=False) #compute shortest paths between all pairs of nodes in the graph

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Eq. (1): C(r) = 2/(n(n-1)) Σ_{i<j} H(r - ||xi - xj||) = integral{r-0} p(r) dr
# H heaviside function and p(r) distance by pairs distributions
def compute_distance_distribution(D, bins="fd"):
    # D: geodesic distance matrix, bins: number of histogram bins
    # fd: Freedman–Diaconis rule, it is adaptive to data
    n = D.shape[0] #number of points/images
    triu_indices = np.triu_indices(n, k=1)  #Σ_{i<j} only upper triangle without diagonal
    distances = D[triu_indices] #||xi - xj||, extract upper triangle distances
    
    distances = distances[np.isfinite(distances) & (distances > 0)] #remove inf and zero distances (self-distances)
    if distances.size == 0:
        return None, None, None

    #histogram = H(r - ||xi - xj||)
    if isinstance(bins, set): #if bins is a set, it means it was passed as a string like "fd", we need to convert it to the actual number of bins using numpy histogram rules
        if len(bins) == 1: #
            bins = next(iter(bins)) # get the single value from the set, e.g., "fd"
        else:
            raise ValueError("bins must be an integer, string, or array-like; got a set with multiple values")
    counts, edges = np.histogram(distances, bins=bins) #histogram of distances, counts are the number of pairs with distance in each bin, edges are the bin edges
    total_pairs = distances.size #total number of pairs considered
    
    #p(r)=dC(r)/dr, C(r) is a escaloned cumulative histogram of distances (it does not exist derivative, it estimates with histogram)
    p = counts.astype(np.float64) / total_pairs # p(r): normalize histogram to get probability distribution
    r = 0.5 * (edges[:-1] + edges[1:]) #midpoints of histogram bins
    C = (2.0 / (n * (n - 1))) * np.cumsum(counts) # Eq. (1): C(r) = 2/(n(n-1)) Σ_{i<j} H(r - ||xi - xj||)

    return r, p, C

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#3.1 intrinsic dimension estimation using geodesic distance distribution
def estimate_intrinsic_dimension(X, k=15):
    n = X.shape[0]
    if n < 3:
        return np.nan, {"r": None, "p": None, "C": None, "r_max": np.nan, "sigma": np.nan}

    def _fit_id(r, p, min_points=5, range_scale=2.0): 
        # min points=5 and not 2, because 2 points fit a line but are unstable; 5 points makes the fit less noisy.
        if r is None or np.all(p == 0):
            return None, None, None, False

        p_smooth = gaussian_filter1d(p, sigma=1.0) #smooth p(r) to find mode r_max, sigma, use Gaussian filter to smooth the probability distribution p(r) to find a more stable mode r_max and estimate sigma, which is the standard deviation of the distribution around the mode
        r_max = r[np.argmax(p_smooth)] #"r_max = argmax p(r)"
        dr = np.mean(np.diff(r)) #bin width, used to estimate sigma as the standard deviation of the distribution around the mode
        sigma = np.sqrt(np.sum(p * (r - r_max) ** 2) * dr) #sigma = sqrt( ∫ (r - r_max)^2 p(r) dr ), where the integral is approximated by a sum over the histogram bins, and dr is the bin width
        
        if not np.isfinite(sigma) or sigma <= 0: #if sigma is not finite or non-positive, it means the distribution is degenerate or has no spread around the mode, which makes it impossible to fit the hypersphere model, so it return failure
            return None, r_max, sigma, False

        low = r_max - range_scale * sigma #r_max - 2σ
        mask = (r >= low) & (r <= r_max) & (p > 0) #range "r_max - 2σ ≤ r ≤ r_max", but we can try different range scales if not enough points
        r_fit = r[mask]
        p_fit = p[mask]
        if r_fit.size < min_points: #not enough points to fit, we can try different range scales or min_points thresholds
            return None, r_max, sigma, False

        y = np.log(p_fit / np.max(p_fit)) #log(p(r)/p(r_max))
        sin_arg = np.sin(np.pi * r_fit / (2.0 * r_max)) # sin(πr/2r_max)
        valid = sin_arg > 0 #only valid points where sin_arg is positive, because log(sin_arg) is only defined for positive values, and it also corresponds to the range where the hypersphere model is valid
        x = np.log(sin_arg[valid]) # log(sin(πr/2r_max))
        y = y[valid] # log(p(r)/p(r_max)) for the valid points, we will fit a line to x and y to estimate m, according to the hypersphere model log(p(r)/p(r_max)) ≈ (m-1) log(sin(πr/2r_max))
        if x.size < max(2, min_points - 1): #not enough valid points to fit
            return None, r_max, sigma, False

        m = 1.0 + np.dot(x, y) / np.dot(x, x) # m = 1 + (x·y) / (x·x), least-squares solution to fit the hypersphere model
        m = float(max(m, 1.0)) #intrinsic dimension cannot be less than 1, if the fit gives a value less than 1, set it to 1, which corresponds to a curve or line-like structure in the data
        return m, r_max, sigma, True

    def _try_fit(r, p):
        for range_scale, min_points in [(2.0, 5), (3.0, 4), (4.0, 3), (4.0, 2)]:
            m, r_max, sigma, ok = _fit_id(r, p, min_points=min_points, range_scale=range_scale)
            if ok:
                return m, r_max, sigma, True
        return None, None, None, False

    max_k = min(n - 1, max(k, 10))
    last = None
    for kk in range(k, max_k + 1):
        graph = build_knn_graph(X, k=kk, metric='cosine')
        n_comp, labels = connected_components(graph, directed=False, return_labels=True)
        if n_comp > 1:
            counts = np.bincount(labels)
            largest = np.argmax(counts)
            idx = np.where(labels == largest)[0]
            if idx.size < 3:
                continue
            graph = graph[idx][:, idx]

        D = compute_geodesic_distances(graph)
        for bins in ("fd", 30, 20):
            r, p, C = compute_distance_distribution(D, bins=bins)
            m, r_max, sigma, ok = _try_fit(r, p)
            if ok:
                return m, {"r": r, "p": p, "C": C, "r_max": r_max, "sigma": sigma}
            last = (r, p, C, r_max, sigma)

    #1. build k-NN graph - "k-nearest neighbors using cosine similarity for SphereFace"
    graph = build_knn_graph(X, k=min(n - 1, max(k, 2)), metric='cosine')
    
    #2. geodesic distances - "geodesic distance induced by a neighborhood graph"
    D = compute_geodesic_distances(graph)
    
    #3.distance distribution p(r) and C(r) - Eq. (1)
    r, p, C = compute_distance_distribution(D)
    if r is None or np.all(p == 0):
        if last is not None:
            r, p, C, r_max, sigma = last
            return np.nan, {"r": r, "p": p, "C": C, "r_max": r_max, "sigma": sigma}
        return np.nan, {"r": None, "p": None, "C": None, "r_max": np.nan, "sigma": np.nan}
    
    #4. compute r_max and sigma - Fig. 2(b), Sec. 3.1
    p_smooth = gaussian_filter1d(p, sigma=1.0) #smooth p(r) to find mode r_max, sigm
    r_max = r[np.argmax(p_smooth)] #find r_max where p(r) is maximum - " around the mode of p(r)
    # sigma = sqrt( ∫ (r - r_max)^2 p(r) dr )
    dr = np.mean(np.diff(r))
    sigma = np.sqrt(np.sum(p * (r - r_max) ** 2) * dr)
    
    #5. fitting range: r_max - 2σ ≤ r ≤ r_max - "The probability distribution p(r) at intermediate length-scales - "around the mode of p(r) i.e., (rmax − 2σ) ≤ r ≤ rmax"
    low = r_max - 2.0 * sigma
    mask = (r >= low) & (r <= r_max) & (p > 0)
    r_fit = r[mask]
    p_fit = p[mask]
    if r_fit.size < 5: #not enough points to fit
        m, r_max2, sigma2, ok = _try_fit(r, p)
        if ok:
            return m, {"r": r, "p": p, "C": C, "r_max": r_max2, "sigma": sigma2}
        return np.nan, {"r": r, "p": p, "C": C, "r_max": r_max, "sigma": sigma}
    
    #6. Hypersphere model (Eq. in Sec. 3.1)
    # log(p(r)/p(r_max)) ≈ (m-1) log(sin(πr / 2r_max))
    # "The geodesic distance distribution of an m-hypersphere is 
    y = np.log(p_fit / np.max(p_fit))
    sin_arg = np.sin(np.pi * r_fit / (2.0 * r_max))
    valid = sin_arg > 0
    x = np.log(sin_arg[valid])
    y = y[valid]
    if x.size < 3:
        m, r_max2, sigma2, ok = _try_fit(r, p)
        if ok:
            return m, {"r": r, "p": p, "C": C, "r_max": r_max2, "sigma": sigma2}
        return np.nan, {"r": r, "p": p, "C": C, "r_max": r_max, "sigma": sigma}

    #7. Least-squares solution - "The above optimization problem can be solved via a least-squares fit"
    # min_m ∫ |log(p(r)/p(rmax)) - (m-1)log(sin(πr/2rmax))|²
    # Solution: m = 1 + (x·y) / (x·x)
    m = 1.0 + np.dot(x, y) / np.dot(x, x)
    m = float(max(m, 1.0))

    return m, {
        "r": r,
        "p": p,
        "C": C,
        "r_max": r_max,
        "sigma": sigma
    }
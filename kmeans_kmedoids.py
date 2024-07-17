import numpy as np

def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def calculate_distances(data, centroids):
    return np.sqrt(((data[:, np.newaxis] - centroids)**2).sum(axis=2))


def update_centroids(data, clusters, k, algo):
    if algo == 'kmeans':
        return np.array([data[clusters == i].mean(axis=0) if len(
                data[clusters == i]) > 0 else np.zeros(data.shape[1]) for i in
                         range(k)])

    elif algo == 'kmedoids':
        centroids = np.zeros((k, data.shape[1]))
        for i in range(k):
            cluster_points = data[clusters == i]
            if len(cluster_points) > 0:
                medoid_index = np.argmin(np.sum(np.linalg.norm(
                    cluster_points[:, np.newaxis] - cluster_points, axis=2),
                                                axis=1))
                centroids[i] = cluster_points[medoid_index]
            else:
                centroids[i] = np.zeros(data.shape[1])
        return centroids


def check_convergence(new_centroids, centroids, tol):
    return np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol)

def k_means_medoids(data, k, algo='kmeans', max_iter=100, tol=0.0001):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iter):
        distances = calculate_distances(data, centroids)
        clusters = np.argmin(distances, axis=1)
        new_centroids = update_centroids(data, clusters, k, algo)
        if check_convergence(new_centroids, centroids, tol):
            break
        centroids = new_centroids
    return centroids, clusters

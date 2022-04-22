import numpy as np


def GetSubSet(randset, randsubsets, **kwargs):
    """
    Returns a function that takes a graph and returns a
    random subset of the graph.
    """

    def inner(data):
        graph, labels, sets = data

        nodeidxs = np.where(
            (sets[0][:, 0] == randset) & (np.isin(sets[0][:, 1], randsubsets))
        )[0]
        edgeidxs = np.where(
            (sets[1][:, 0] == randset) & (np.isin(sets[1][:, 1], randsubsets))
        )[0]

        node_features = graph[0][nodeidxs]
        edge_features = graph[1][edgeidxs]
        edge_connections = graph[2][edgeidxs]

        node_sets = sets[0][nodeidxs]
        edge_sets = sets[1][edgeidxs]

        return (
            (node_features, edge_features, edge_connections),
            labels[randset],
            (node_sets, edge_sets),
        )

    return inner


def AugmentCentroids(rotate, translate):
    """
    Returns a function that takes a graph and augments the centroids
    by applying a random rotation and translation
    """

    def inner(data):
        graph, labels, sets = data

        centroids = graph[0][:, :2]

        # Extract subsets ids
        subsets = sets[0][:, 1]

        # Count number of nodes per subset
        _, counts = np.unique(subsets, return_counts=True)

        # Repeat rotation and translation
        rotations, translations = (
            np.repeat(rotate, counts, axis=0),
            np.repeat(translate, counts, axis=0),
        )

        # Apply rotation and translation
        centroids = centroids - 0.5
        centroids_x = (
            centroids[:, 0] * np.cos(rotations)
            + centroids[:, 1] * np.sin(rotations)
            + translations[:, 0]
        )
        centroids_y = (
            centroids[:, 1] * np.cos(rotations)
            - centroids[:, 0] * np.sin(rotations)
            + translations[:, 1]
        )

        node_features = np.array(graph[0])
        node_features[:, 0] = centroids_x + 0.5
        node_features[:, 1] = centroids_y + 0.5

        return (node_features, *graph[1:]), labels

    return inner

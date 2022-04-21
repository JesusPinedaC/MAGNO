import numpy as np
import pandas as pd
import tqdm


def GetEdge(df: pd.DataFrame, radius: int = 0.2, **kwargs):
    """
    Extracts the edges from a windowed sequence of frames
    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing the extracted node properties.
    radius: int
        Search radius for the edge (pixel units).
    Returns
    -------
    edges: pd.DataFrame
        A dataframe containing the extracted
        properties of the edges.
    """

    # repeat nodes of dataframe
    rrowdf = df.reindex(df.index.repeat(df["length"])).reset_index(drop=True)

    # repeat the full dataframe
    rdf = (
        df.groupby("subset")
        .apply(lambda x: x.iloc[np.tile(np.arange(len(x)), len(x))])
        .reset_index(drop=True)
    )

    # Compute distances between centroids
    diffx = rrowdf["centroids-x"] - rdf["centroids-x"]
    diffy = rrowdf["centroids-y"] - rdf["centroids-y"]

    rrowdf["feature-dist"] = np.sqrt(diffx ** 2 + diffy ** 2)

    # drop self-edges
    rrowdf = rrowdf[rrowdf["feature-dist"] != 0.0]

    # Add column with neighbors subset ids
    rrowdf["neigh-subset"] = rdf["subset"]

    # Append neighbors indexes
    rrowdf["neigh-idx"] = rdf["idx"]
    rrowdf["index"] = rrowdf[["idx", "neigh-idx"]].values.tolist()

    # Filter out edges with a feature-distance less than scale * radius
    rrowdf = rrowdf[rrowdf["feature-dist"] < radius].filter(
        regex=("index|set|feature-dist")
    )

    return rrowdf


def EdgeExtractor(nodesdf, **kwargs):
    """
    Extracts edges from a sequence of frames
    Parameters
    ----------
    nodesdf: pd.DataFrame
        A dataframe containing the extracted node properties.
    """
    # Create a copy of the dataframe to avoid overwriting
    df = nodesdf.copy()

    edgedfs = []
    sets = np.unique(df["set"])

    for setid in tqdm.tqdm(sets, desc="Extracting edges"):
        df_set = df[df["set"] == setid].copy()
        edgedf = GetEdge(df_set, **kwargs)
        edgedfs.append(edgedf)

    # Concatenate the dataframes in a single
    # dataframe for the whole set of edges
    edgedf = pd.concat(edgedfs)

    return edgedf


def DataframeSplitter(df, props: tuple, to_array=True, **kwargs):
    """
    Splits a dataframe into features and labels
    Parameters
    ----------
    dt: pd.DataFrame
        A dataframe containing the extracted
        properties for the edges/nodes.
    atts: list
        A list of attributes to be used as features.
    to_array: bool
        If True, the features are converted to numpy arrays.
    Returns
    -------
    X: np.ndarray or pd.DataFrame
        Features.
    """
    # Extract features from the dataframe
    if len(props) == 1:
        features = df.filter(like=props[0])
    else:
        regex = ""
        for prop in props[0:]:
            regex += prop + "|"
        regex = regex[:-1]
        features = df.filter(regex=regex)

    if "index" in df:
        # convert output features to a numpy array
        outputs = [features.to_numpy(), np.array(df["index"].to_list())]
    else:
        outputs = features.to_numpy()

    return outputs


def GraphExtractor(
    nodesdf: pd.DataFrame,
    labels: list = None,
    properties: list = None,
    **kwargs
):
    """
    Extracts graphs from a node dataframe
    Parameters
    ----------
    nodesdf: pd.DataFrame
        A dataframe containing node properties.
    properties: list
        A list of properties to extract.
    """

    # Extract edges and edge features from nodes
    edgesdf = EdgeExtractor(nodesdf, **kwargs)

    # Split the nodes dataframe into features
    nodefeatures = DataframeSplitter(nodesdf, props=properties, **kwargs)

    # Split the edges dataframe into features and
    # sparse adjacency matrix
    edgefeatures, sparseadjmtx = DataframeSplitter(
        edgesdf, props=("feature",), **kwargs
    )

    # # Extract set ids
    nodesets = nodesdf[["set", "subset"]].to_numpy()
    edgesets = edgesdf[["set", "subset"]].to_numpy()

    labels = (
        labels
        or [
            0.0,
        ]
        * len(np.unique(nodesets[:, 0]))
    )

    return (
        (nodefeatures, edgefeatures, sparseadjmtx),
        labels,
        (nodesets, edgesets),
    )

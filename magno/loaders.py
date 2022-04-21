import numpy as np
import json

import itertools
import pandas as pd

import tqdm
from operator import itemgetter

_PATH_TO_DATASET = "datasets/{mode}/dataset.json"


class CentroidAugmentor:
    def __init__(self, locs, loc_prec=5):
        self.locs = locs
        self.loc_prec = loc_prec

    def augment(self, x, y):
        # Number of detections is drawn from a geometric
        # distribution.
        nblinks = np.random.geometric(0.125)

        # Compute localizations
        xblinks = x + np.random.normal(0, self.loc_prec, nblinks)
        yblinks = y + np.random.normal(0, self.loc_prec, nblinks)

        # NOTE: noise must be added to the localizations

        centroids = np.concatenate(
            [xblinks[..., np.newaxis], yblinks[..., np.newaxis]],
            axis=1,
        )

        return centroids

    def __call__(self):
        return np.stack(
            list(
                itertools.chain.from_iterable(
                    [self.augment(x, y) for x, y in self.locs]
                )
            )
        )


def NodeExtractor(
    mode="training",
    properties: dict = {},
    label_keys: list = ["class", "num_proteins"],
    graphs_per_set=500,
    **kwargs
):

    # Load dataset
    path = _PATH_TO_DATASET.format(mode=mode)
    with open(path) as f:
        data = json.load(f)

    # Update properties
    _properties = {"centroids": 128}
    _properties.update(properties)

    dfs, labels = [], []
    for _set, d in tqdm.tqdm(
        enumerate(data), total=len(data), desc="Loading node features"
    ):
        loc = d["loc"]
        augmentor = CentroidAugmentor(loc, **kwargs)

        for subset in range(graphs_per_set):
            # append data
            centroids = augmentor()
            df = pd.DataFrame(
                {
                    "centroids-x": centroids[:, 0],
                    "centroids-y": centroids[:, 1],
                    "set": _set,
                    "subset": subset,
                }
            )
            df["length"] = len(df)
            dfs.append(df)

        # append labels
        label = list(map(d.get, label_keys))
        labels.append(label)

    # Concatenate dataframes
    dfs = pd.concat(dfs)

    # Add a column for the indexes of the nodes
    dfs["idx"] = dfs.index
    dfs = dfs.reset_index(drop=True)

    # Normalize node attributes
    for key in _properties.keys():
        dfs.loc[:, dfs.columns.str.contains(key)] = (
            dfs.loc[:, dfs.columns.str.contains(key)] / _properties[key]
        )

    return dfs, labels

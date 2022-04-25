from .deeptrack.generators import ContinuousGenerator
import numpy as np


class ContinuousGraphGenerator(ContinuousGenerator):
    """
    Generator that asynchronously generates graph representations.
    The generator aims to speed up the training of networks by striking a
    balance between the generalization gained by generating new images and
    the speed gained from reusing images. The generator will continuously
    create new trainingdata during training, until `max_data_size` is reached,
    at which point the oldest data point is replaced.
    Parameters
    ----------
    feature : dt.Feature
        The feature to resolve the graphs from.
    label_function : Callable
        Function that returns the label corresponding to a feature output.
    batch_function : Callable
        Function that returns the training data corresponding a feature output.
    min_data_size : int
        Minimum size of the training data before training starts
    max_data_set : int
        Maximum size of the training data before old data is replaced.
    batch_size : int or Callable[int, int] -> int
        Number of images per batch. A function is expected to accept the current epoch
        and the size of the training data as input.
    shuffle_batch : bool
        If True, the batches are shuffled before outputting.
    feature_kwargs : dict or list of dicts
        Set of options to pass to the feature when resolving
    ndim : int
        Number of dimensions of each batch (including the batch dimension).
    """

    def __getitem__(self, idx):
        batch, labels = super().__getitem__(idx)

        # Converts each element of nested list to numy array
        batch = [list(map(np.array, b)) for b in batch[0]]

        return batch, labels

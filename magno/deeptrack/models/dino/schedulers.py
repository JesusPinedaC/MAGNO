import tensorflow as tf


class MomentumScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule, **kwargs):
        """
        Schedule for the momentum.
        Parameters
        ----------
        schedule: np.array
            schedule for the momentum. Size of the array 
            should be equal to the number of epochs.
        """
        super(MomentumScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_end(self, epoch, logs=None):
        # set momentum
        tf.keras.backend.set_value(
            self.model.momentum,
            self.schedule(epoch)
        )

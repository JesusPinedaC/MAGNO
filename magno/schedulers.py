import tensorflow as tf
import numpy as np


def CosineDecay(start, stop, epochs, warmup_epochs=0, **kwargs):

    # warmup_epochs
    warmup_schedule = tf.linspace(0, 1, warmup_epochs) * start

    # Cosine decay
    iters = tf.range(epochs - warmup_epochs, dtype=tf.float32)
    schedule = stop + 0.5 * (start - stop) * (
        1 + tf.math.cos(np.pi * iters / len(iters))
    )

    # Concatenate schedules
    schedule = tf.concat(
        (warmup_schedule, tf.cast(schedule, warmup_schedule.dtype)), axis=0
    )

    return schedule


class MomentumScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule: tf.Tensor = None, **kwargs):
        super(MomentumScheduler, self).__init__(**kwargs)

        self.schedule = schedule

    def on_train_begin(self, logs=None):
        self.epochs = self.params.get("epochs")

        # If schedule is None, initialize the schedule with default values
        self.schedule = self.schedule or self.default_schedule()

        # assert length of schedule is equal to epochs
        assert self.epochs == len(
            self.schedule
        ), "Schedule length must match epochs. Got {} and {}".format(
            self.epochs, len(self.schedule)
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.model.momentum = self.schedule[epoch]

    def default_schedule(self):
        return CosineDecay(0.996, 1.0, self.epochs)

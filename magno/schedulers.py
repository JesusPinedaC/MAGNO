from subprocess import call
import tensorflow as tf
import numpy as np


class Scheduler(tf.keras.callbacks.Callback):
    """
    Abstract base class used to build new schedulers.

    Schedulers can be passed to keras method `fit` in order to hook
    into the various stages of the model training lifecycle.
    """

    def __init__(self, schedule: tf.Tensor or callable = None, **kwargs):
        super(Scheduler, self).__init__(**kwargs)

        self.schedule = schedule

    def on_train_begin(self, logs=None):
        self.epochs = self.params.get("epochs")

        # If schedule is None, initialize the schedule with default values
        if self.schedule is None:
            self.schedule = self.default_schedule()

        # assert length of schedule is equal to epochs
        assert self.epochs == len(
            self.schedule
        ), "Schedule length must match epochs. Got {} and {}".format(
            self.epochs, len(self.schedule)
        )


class MomentumScheduler(Scheduler):
    """
    Momentum scheduler.

    At the beginning of every epoch, this callback gets the updated momentum value
    from `schedule` function provided at `__init__`, with the current epoch and
    current momentum, and applies the updated momentum on the model update.
    """

    def on_epoch_begin(self, epoch, logs=None):
        self.model.momentum.assign(
            tf.cast(self.schedule[epoch], self.model.momentum.dtype)
        )

    def default_schedule(self):
        return CosineDecay(0.996, 1.0, self.epochs)


class TemperatureScheduler(Scheduler):
    """
    Temperature scheduler.

    At the beginning of every epoch, this callback gets the updated temperature
    value from `schedule` function provided at `__init__`, with the current epoch
    and current temperature, and applies the updated temperature on the loss update.
    """

    def on_epoch_begin(self, epoch, logs=None):
        self.model.loss.temperature.assign(
            tf.cast(self.schedule[epoch], self.model.loss.temperature.dtype)
        )

    # NOTE: 'temperature' linearly ramps up from 0.04 to 0.07 during
    # the first 30 epochs.
    def default_schedule(self):
        return PiecewiseConstantDecay(
            0.04, 0.07, self.epochs, warmup_epochs=30
        )


class WeightDecayScheduler(Scheduler):
    """
    Weight decay scheduler.

    At the beginning of every epoch, this callback gets the updated weight decay
    value from `schedule` function provided at `__init__`, with the current epoch
    and current weight decay, and applies the updated weight decay on the optimizer.
    """

    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.weight_decay.assign(
            tf.cast(
                self.schedule[epoch], self.model.optimizer.weight_decay.dtype
            )
        )

    # NOTE: 'weight_decay' follows a cosine schedule from 0.04 to 0.4
    def default_schedule(self):
        return CosineDecay(0.04, 0.4, self.epochs)


class LearningRateScheduler(Scheduler):
    """
    Learning rate scheduler.

    At the beginning of every epoch, this callback gets the updated learning rate
    value from `schedule` function provided at `__init__`, with the current epoch
    and current learning rate, and applies the updated learning rate on the optimizer.
    """

    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.learning_rate.assign(
            tf.cast(
                self.schedule[epoch], self.model.optimizer.learning_rate.dtype
            )
        )

    # NOTE: 'learning_rate' linearly ramps up from 1e-6 to 1e-3 during the
    # first 10 epochs, and decays with a cosine schedule thereafter to 1e-6.
    def default_schedule(self):
        return CosineDecay(
            1e-3, 1e-6, self.epochs, base_value=1e-6, warmup_epochs=10
        )


class CenterSetter(tf.keras.callbacks.Callback):
    """
    Center setter.

    At the beginning of training, this callback zero-initializes the `center`
    matrix used in the loss function. This matrix is then updated at the end
    of each epoch by taking the mean of the teacher output.
    """

    def on_train_begin(self, logs=None):
        # Extract representation size from model
        representation_size = self.model.representation_size

        # Zero-initialize the `center` matrix
        self.model.loss.center = tf.Variable(
            tf.zeros((1, representation_size)), trainable=False
        )


def CosineDecay(
    start,
    stop,
    epochs,
    base_value=1e-6,
    warmup_epochs=0,
    return_callable=False,
    **kwargs
):

    # warmup_epochs
    warmup_schedule = np.linspace(base_value, start, warmup_epochs)

    # Cosine decay
    iters = tf.range(epochs - warmup_epochs, dtype=tf.float32)
    schedule = stop + 0.5 * (start - stop) * (
        1 + tf.math.cos(np.pi * iters / len(iters))
    )

    # Concatenate schedules
    schedule = tf.concat(
        (warmup_schedule, tf.cast(schedule, warmup_schedule.dtype)), axis=0
    )

    if return_callable:
        return lambda step: schedule[step]
    else:
        return schedule


def PiecewiseConstantDecay(
    start, stop, epochs, warmup_epochs=0, return_callable=False, **kwargs
):
    schedule = tf.concat(
        (
            np.linspace(
                start,
                stop,
                warmup_epochs,
            ),
            np.ones(epochs - warmup_epochs) * stop,
        ),
        axis=0,
    )

    if return_callable:
        return lambda step: schedule[step]
    else:
        return schedule

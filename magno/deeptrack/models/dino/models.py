import tensorflow as tf
from tensorflow.keras import layers

from ..utils import GELU, KerasModel
from ..layers import as_block, DenseBlock

from ....generators import ContinuousGraphGenerator


class MAGNO(tf.keras.Model):

    # Generator that asynchronously generates
    # graph representations.
    data_generator = ContinuousGraphGenerator

    def __init__(
        self,
        encoder,
        teacher,
        representation_size,
        num_projection_layers=1,
        dense_block=DenseBlock(activation=GELU, normalization=None),
        **kwargs
    ):
        super().__init__()

        dense_block = as_block(dense_block)

        # Define the student model.
        self.encoder = (
            encoder.model if isinstance(encoder, KerasModel) else encoder
        )

        # Define representation size
        self.representation_size = representation_size

        # Define representation layer. This layer is used to project the
        # latent representation of the encoder to higher dimensional space,
        # and is shared between the teacher and the student.
        input_layer = layers.Input(shape=(encoder.output_shape[1:]))

        layer = input_layer
        for _ in range(num_projection_layers):
            layer = dense_block(representation_size)(layer)

        self.projection_head = tf.keras.Model(
            input_layer,
            layers.Dense(representation_size)(layer),
        )

        # Define the teacher model. Importantly, The teacher is
        # initialized with the same weights as the student.
        self.teacher = (
            teacher.model if isinstance(teacher, KerasModel) else teacher
        )
        self.teacher.set_weights(self.encoder.get_weights())

        # Clone projection head for teacher.
        self.teacher_projection_head = tf.keras.models.clone_model(
            self.projection_head
        )
        self.teacher_projection_head.set_weights(
            self.projection_head.get_weights()
        )

        # Set the teacher to be not trainable
        self.teacher.trainable = False
        self.teacher_projection_head.trainable = False

    def train_step(self, data):

        data, *_ = data

        idx = len(data) // 2
        data = data[:idx], data[idx:]

        with tf.GradientTape() as tape:
            proj_s, proj_t = [], []
            for batch_s, batch_t in zip(*data):
                # Compute global representations from
                # student and teacher models.
                out_s = self.encoder(batch_s)
                out_t = self.teacher(batch_t)

                # Compute projections.
                out_s = self.projection_head(out_s)
                out_t = self.teacher_projection_head(out_t)

                # Normalize representations.
                out_s = tf.math.l2_normalize(out_s, axis=-1)
                out_t = tf.math.l2_normalize(out_t, axis=-1)

                proj_s.append(out_s)
                proj_t.append(out_t)

            # Concatenate representations into a single matrix.
            # SIZE: (batch_size//2, representation_size)
            proj_s = tf.concat(proj_s, axis=0)
            proj_t = tf.concat(proj_t, axis=0)

            # Compute the loss.
            loss = self.compiled_loss(proj_s, proj_t)

        # compute gradients
        trainable_vars = (
            self.encoder.trainable_weights
            + self.projection_head.trainable_weights
        )
        grads = tape.gradient(
            loss,
            trainable_vars,
        )
        # update weights
        self.optimizer.apply_gradients(
            zip(
                grads,
                trainable_vars,
            )
        )

        # Update weights of the teacher using an exponential
        # moving average (EMA) on the student weights.
        teacher_vars = (
            self.teacher.weights + self.teacher_projection_head.weights
        )
        teacher_vars = tf.nest.map_structure(
            lambda x, y: y.assign(
                (1 - tf.cast(self.momentum, x.dtype)) * x
                + tf.cast(self.momentum, y.dtype) * y
            ),
            trainable_vars,
            teacher_vars,
        )

        # Return dict mapping the loss to its current value
        return {
            "loss": loss,
            "lr": self.optimizer.learning_rate,
            "momentum": self.momentum,
            "temperature": self.loss.temperature,
        }

    def call(self, x):
        return self.encoder(x)

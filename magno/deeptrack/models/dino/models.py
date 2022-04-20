import tensorflow as tf
from tensorflow.keras import layers

from ..utils import GELU
from ..layers import as_block, DenseBlock


class MAGNO(tf.keras.Model):
    def __init__(
        self,
        encoder,
        teacher,
        representation_size,
        num_projection_layers=1,
        dense_block=DenseBlock(activation=GELU, normalization=None),
        **kwargs
    ):
        super(MAGNO, self).__init__(**kwargs)

        dense_block = as_block(dense_block)

        # Define the student model.
        self.encoder = encoder

        # Define representation layer. This layer is used to project the
        # latent representation of the encoder to higher dimensional space,
        # and is shared between the teacher and the student.
        input_layer = layers.Input(shape=encoder.output_shape)

        layer = input_layer
        for _ in range(num_projection_layers):
            layer = dense_block(representation_size)(layer)

        self.projection_head = tf.keras.Model(
            input_layer,
            layers.Dense(representation_size)(layer),
        )

        # Define the teacher model. Importantly, The teacher is
        # initialized with the same weights as the student.
        self.teacher = teacher
        self.teacher.set_weights(self.student.get_weights())

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

        idx = len(data) // 2
        data = data[:idx], data[idx:]

        with tf.GradientTape() as tape:
            proj_s, proj_t = [], []
            for batch_s, batch_t in zip(*data):
                # Compute global representations from
                # student and teacher models.
                out_s = self.student(batch_s)
                out_t = self.teacher(batch_t)

                # Compute projections.
                proj_s.append(self.projection_head(out_s))
                proj_t.append(self.teacher_projection_head(out_t))

            # Concatenate representations into a single matrix.
            # SIZE: (batch_size//2, representation_size)
            proj_s = tf.concat(proj_s, axis=0)
            proj_t = tf.concat(proj_t, axis=0)

            # Compute the loss.
            loss = self.compiled_loss(proj_s, proj_t)

        # compute gradients
        trainable_vars = (
            self.student.trainable_weights
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
            self.teacher.trainable_weights
            + self.teacher_projection_head.trainable_weights
        )
        teacher_vars = tf.nest.map_structure(
            lambda x, y: (1 - self.momentum) * x + self.momentum * y,
            trainable_vars,
            teacher_vars,
        )

        # Update the teacher model.
        self.teacher.set_weights(teacher_vars[: len(self.teacher.weights)])

        # Update the teacher projection head.
        self.teacher_projection_head.set_weights(
            teacher_vars[len(self.teacher.weights) :]
        )

        # Return dict mapping the loss to its current value
        return {"loss": loss}

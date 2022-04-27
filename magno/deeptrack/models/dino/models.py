import tensorflow as tf
from tensorflow.keras import layers

from ..utils import GELU, KerasModel
from ..layers import as_block, DenseBlock

from ....generators import ContinuousGraphGenerator


class MAGNOBaseModel(tf.keras.Model):

    # Generator that asynchronously generates
    # graph representations.
    data_generator = ContinuousGraphGenerator

    def __init__(
        self,
        encoder_temperature=0.1,
        center_momentum=0.9,
        temperature=0.04,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Encoder temperature
        self.encoder_temperature = encoder_temperature
        # Center momentum
        self.center_momentum = center_momentum
        # Temperature for the teacher output
        self.temperature = tf.Variable(temperature, trainable=False)
        # Initialize momentum
        self.momentum = tf.Variable(1.0, trainable=False)

        # Zero-initialize the `center` matrix
        self.center = tf.Variable(
            0.0, trainable=False, shape=tf.TensorShape(None)
        )

    def update_center(self, teacher_output):
        batch_center = tf.reduce_mean(teacher_output, axis=0)

        # Update batch center using an exponential
        # moving average (EMA)
        self.center.assign(
            self.center * self.center_momentum
            + (1 - self.center_momentum) * batch_center
        )

    def dino_loss(self, encoder_out, teacher_out):
        """
        Computes the cross-entropy between the softmax outputs of the
        teacher and encoder networks.
        """
        encoder_out /= self.encoder_temperature
        teacher_out = tf.nn.softmax(
            (teacher_out - tf.cast(self.center, teacher_out.dtype))
            / tf.cast(self.temperature, self.center.dtype)
        )

        # Compute the cross-entropy between the teacher and encoder outputs
        loss = tf.map_fn(
            lambda x: tf.reduce_sum(
                -teacher_out * tf.nn.log_softmax(x, axis=-1), axis=-1
            ),
            elems=encoder_out,
        )
        loss = tf.reduce_mean(loss)

        # Update center
        self.update_center(teacher_out)

        return loss


class MAGNO(MAGNOBaseModel):
    def __init__(
        self,
        encoder,
        teacher,
        representation_size,
        num_projection_layers=1,
        dense_block=DenseBlock(activation=GELU, normalization=None),
        **kwargs
    ):
        super().__init__(**kwargs)

        dense_block = as_block(dense_block)

        # Define the encoder model.
        self.encoder = (
            encoder.model if isinstance(encoder, KerasModel) else encoder
        )

        # Define representation size
        self.representation_size = representation_size

        # Define representation layer. This layer is used to project the
        # latent representation of the encoder to higher dimensional space,
        # and is shared between the teacher and the encoder.
        input_layer = layers.Input(shape=(encoder.output_shape[1:]))

        layer = input_layer
        for _ in range(num_projection_layers):
            layer = dense_block(representation_size)(layer)

        self.projection_head = tf.keras.Model(
            input_layer,
            layers.Dense(representation_size)(layer),
        )

        # Define the teacher model. Importantly, The teacher is
        # initialized with the same weights as the encoder.
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
                # encoder and teacher models.
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
            loss = self.dino_loss(proj_s, proj_t)

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
        # moving average (EMA) on the encoder weights.
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
            "lr": self.optimizer._hyper["learning_rate"],
            "weight_decay": self.optimizer._hyper["weight_decay"],
            "momentum": self.momentum,
            "temperature": self.temperature,
        }

    def call(self, x):
        return self.encoder(x)

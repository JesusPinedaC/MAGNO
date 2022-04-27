import tensorflow as tf


class DINOLoss(tf.keras.losses.Loss):
    def __init__(
        self, student_temp=0.1, center_momentum=0.9, temperature=0.04, **kwargs
    ):
        super().__init__(**kwargs)

        self.student_temp = student_temp
        self.center_momentum = center_momentum

        # Define temperature
        self.temperature = tf.Variable(temperature, trainable=False)

    def update_center(self, teacher_output):
        batch_center = tf.reduce_mean(teacher_output, axis=0)

        # Update batch center using an exponential
        # moving average (EMA)
        self.center.assign(
            self.center * self.center_momentum
            + (1 - self.center_momentum) * batch_center
        )

    def call(self, student_out, teacher_out):
        """
        Computes the cross-entropy between the softmax outputs of the
        teacher and student networks.
        """
        student_out /= self.student_temp
        teacher_out = tf.nn.softmax(
            (teacher_out - tf.cast(self.center, teacher_out.dtype))
            / tf.cast(self.temperature, self.center.dtype)
        )

        # Compute the cross-entropy between the teacher and student outputs
        loss = tf.map_fn(
            lambda x: tf.reduce_sum(
                -teacher_out * tf.nn.log_softmax(x, axis=-1), axis=-1
            ),
            elems=student_out,
        )
        loss = tf.reduce_mean(loss)

        # Update center
        self.update_center(teacher_out)

        return loss

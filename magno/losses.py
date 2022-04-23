import tensorflow as tf
import itertools


class DINOLoss(tf.keras.losses.Loss):
    def __init__(self, student_temp=0.1, center_momentum=0.9, **kwargs):
        super().__init__(**kwargs)

        self.student_temp = student_temp
        self.center_momentum = center_momentum

    def update_center(self, teacher_output):
        batch_center = tf.reduce_mean(teacher_output, axis=0)

        # Update batch center using an exponential
        # moving average (EMA)
        self.center = (
            self.center * self.center_momentum
            + (1 - self.center_momentum) * batch_center
        )

    def __call__(self, student_out, teacher_out, sample_weight=None):
        """
        Computes the cross-entropy between the softmax outputs of the
        teacher and student networks.
        """
        student_out /= self.student_temp
        teacher_out = tf.nn.softmax(
            (teacher_out - self.center) / self.temperature
        )

        # Compute the cross-entropy between the teacher and student outputs
        total_loss, terms = 0, 0
        for t, s in itertools.product(teacher_out, student_out):
            loss = tf.reduce_sum(-t * tf.nn.log_softmax(s, axis=-1), axis=-1)
            total_loss += tf.reduce_mean(loss)

            # Count the number of terms
            terms += 1

        # Average the loss
        total_loss /= terms

        # Update center
        self.update_center(teacher_out)

        return total_loss

import tensorflow as tf


class QuadMetric():
    def __init__(self):
        self.last_loss = tf.zeros([1])

    def reset_state(self):
        self.last_loss = tf.zeros([1])

    def update_state(self, _, outputs):
        self.last_loss = outputs

    def result(self):
        return self.last_loss

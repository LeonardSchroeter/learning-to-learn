import tensorflow as tf


# Metric object to track the accuracy of the quadratic objective function
# This simply stores the value of the last loss which corresponds to the output of the quadratic function
class QuadMetric():
    def __init__(self):
        self.last_loss = tf.zeros([1])

    def reset_state(self):
        self.last_loss = tf.zeros([1])

    def update_state(self, _, outputs):
        self.last_loss = outputs

    def result(self):
        return self.last_loss

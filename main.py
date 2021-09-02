import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

from LSTMNetworkPerParameter import LSTMNetworkPerParameter
from QuadraticFunction import QuadraticFunction

optimizer_network = LSTMNetworkPerParameter()
optimizer_optimizer = keras.optimizers.Adam()

first = True

for training_step in range(100_000):
    print("TRAINING STEP: ", training_step)
    quadratic_function = QuadraticFunction(10)
    theta = tf.random.normal([10])

    with tf.GradientTape(persistent=True) as tape:
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(theta)

            optimizer_loss = tf.zeros([1], tf.float32)

            for step in range(100):
                loss = quadratic_function(theta)

                with tape2.stop_recording():
                    gradients = tape2.gradient(loss, theta)

                optimizer_output = optimizer_network(gradients)

                theta = theta + optimizer_output

                optimizer_loss = optimizer_loss + loss

                if step % 10 == 0:
                    print("  Loss: ", loss.numpy())

    if first:
        first = False
    else:
        optimizer_gradients = tape.gradient(optimizer_loss, optimizer_network.trainable_weights)
        optimizer_optimizer.apply_gradients(zip(optimizer_gradients, optimizer_network.trainable_weights))



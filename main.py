import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import deque

import tensorflow as tf
from tensorflow import keras

from LSTMNetworkPerParameter import LSTMNetworkPerParameter
from QuadraticFunction import QuadraticFunction


def example_1():
    optimizer_network = LSTMNetworkPerParameter()
    optimizer_optimizer = keras.optimizers.Adam()

    for training_step in range(100_000):
        print("TRAINING STEP: ", training_step)
        quadratic_function = QuadraticFunction(10)
        theta = tf.random.normal([10])

        with tf.GradientTape(persistent=True) as tape:
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(theta)

                optimizer_loss = tf.zeros([1], tf.float32)

                for step in range(50):
                    loss = quadratic_function(theta)

                    with tape2.stop_recording():
                        gradients = tape2.gradient(loss, theta)

                    optimizer_output = optimizer_network(gradients)

                    theta = theta + optimizer_output

                    optimizer_loss = optimizer_loss + loss

                    if step % 10 == 0:
                        print("  Loss: ", loss.numpy())

        optimizer_gradients = tape.gradient(optimizer_loss, optimizer_network.trainable_weights)
        optimizer_optimizer.apply_gradients(zip(optimizer_gradients, optimizer_network.trainable_weights))

        optimizer_network.reset_states()

def example_2(T):
    optimizer_network = LSTMNetworkPerParameter()
    optimizer_optimizer = keras.optimizers.Adam()

    losses = deque(maxlen=T)

    for training_step in range(100_000):
        print("TRAINING STEP: ", training_step)
        quadratic_function = QuadraticFunction(10)
        theta = tf.random.normal([10])
        
        optimizer_network.reset_states()

        with tf.GradientTape(persistent=True) as tape:
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(theta)

                first = True
                for step in range(40):

                    loss = quadratic_function(theta)
                    losses.append(loss)

                    with tape2.stop_recording():
                        gradients = tape2.gradient(loss, theta)

                    optimizer_output = optimizer_network(gradients)
                    theta = theta + optimizer_output

                    optimizer_loss = tf.add_n(list(losses))

                    if step % 10 == 0:
                        print("  Loss: ", loss.numpy())
                    
                    if first:
                        first = False
                    else:
                        with tape.stop_recording():
                            optimizer_gradients = tape.gradient(optimizer_loss, optimizer_network.trainable_weights)
                            optimizer_optimizer.apply_gradients(zip(optimizer_gradients, optimizer_network.trainable_weights))

if __name__ == "__main__":
    example_2(20)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import deque

import tensorflow as tf
from LSTMNetworkPerParameter import LSTMNetworkPerParameter
from QuadraticFunction import QuadraticFunction, QuadraticFunctionLayer
from tensorflow import keras


# accumulate losses and train optimizer once for a single objective function
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

# accumulate the last T losses and train optimizer in each step of the optimization of the objective function
def example_2(T):
    optimizer_network = LSTMNetworkPerParameter()
    optimizer_optimizer = keras.optimizers.Adam()

    losses = deque(maxlen=T)

    for training_step in range(100_000):
        print("TRAINING STEP: ", training_step)
        quadratic_function = QuadraticFunction(10)
        theta = tf.random.normal([10])
        
        optimizer_network.reset_states()
        losses.clear()

        with tf.GradientTape(persistent=True) as tape:
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(theta)

                first = True
                for step in range(200):

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

# dont take the input to the optimizer network into account when computing the gradient of the loss
def example_3():
    optimizer_network = LSTMNetworkPerParameter()
    optimizer_optimizer = keras.optimizers.Adam()

    for training_step in range(100_000):
        print("TRAINING STEP: ", training_step)
        quadratic_function = QuadraticFunction(10)
        theta = tf.random.normal([10])

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(theta)

            optimizer_loss = tf.zeros([1], tf.float32)

            for step in range(50):
                loss = quadratic_function(theta)

                with tape.stop_recording():
                    gradients = tape.gradient(loss, theta)

                gradients = tf.stop_gradient(gradients)

                optimizer_output = optimizer_network(gradients)

                theta = theta + optimizer_output

                optimizer_loss = optimizer_loss + loss

                if step % 10 == 0:
                    print("  Loss: ", loss.numpy())

        optimizer_gradients = tape.gradient(optimizer_loss, optimizer_network.trainable_weights)
        optimizer_optimizer.apply_gradients(zip(optimizer_gradients, optimizer_network.trainable_weights))

        optimizer_network.reset_states()

# Apparently tf.GradientTape does not track variable assignment operations. Therefore assigning new values to theta
# in each step makes the tape forget that this new value depends on the output of the optimizer network, which results in
# the gradient of the new theta w.r.t. the weights of the optimizer network to return Null.
# We need to compute this gradient though in order to train the optimizer network.
# I tried a workaround where I assign completely new tensors to theta. This made the gradient of theta w.r.t. 
# the weights of the optimizer network to work fine, but in the next step the gradient of the loss of the
# objective function network w.r.t. its weights theta throws an error.
# Will have to look into why this happens and if I can find another workaround for this.
# Maybe try to use pytorch instead of tensorflow/keras? --> Find out whether autograd supports variable assignments first
def example_4():
    optimizer_network = LSTMNetworkPerParameter()
    optimizer_optimizer = keras.optimizers.Adam()

    for training_step in range(100_000):
        print("TRAINING STEP: ", training_step)
        quadratic_function = QuadraticFunctionLayer(10)

        with tf.GradientTape(persistent=True) as tape:

            optimizer_loss = tf.zeros([1], tf.float32)

            for step in range(50):
                loss = quadratic_function(tf.zeros([1]))

                with tape.stop_recording():
                    gradients = tape.gradient(loss, quadratic_function.trainable_weights)

                gradients = tf.stop_gradient(gradients)

                optimizer_output = optimizer_network(gradients)

                for i, (theta_t, g_t) in enumerate(zip(quadratic_function._trainable_weights, optimizer_output)):
                    quadratic_function._trainable_weights[i] = theta_t + g_t

                optimizer_loss = optimizer_loss + loss

                if step % 10 == 0:
                    print("  Loss: ", loss.numpy())

        optimizer_gradients = tape.gradient(optimizer_loss, optimizer_network.trainable_weights)
        optimizer_optimizer.apply_gradients(zip(optimizer_gradients, optimizer_network.trainable_weights))

        optimizer_network.reset_states()

if __name__ == "__main__":
    example_3()

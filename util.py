import math
import time

import tensorflow as tf


class Util():
    def __init__(self):
        self.dict_structure = None
        self.sizes_shapes = None

    def to_1d(self, weight_dict):
        sizes_shapes = []
        weights_1d = []
        dict_structure = {}
        for layer_name, layer_weights in weight_dict.items():
            dict_structure[layer_name] = {}
            for weight_name, weights in layer_weights.items():
                dict_structure[layer_name][weight_name] = None
                size = tf.size(weights).numpy()
                shape = tf.shape(weights).numpy()
                sizes_shapes.append((size, shape))
                weights_1d.append(tf.reshape(weights, size))
        all_weights_1d = tf.concat(weights_1d, 0)

        self.dict_structure = dict_structure
        self.sizes_shapes = sizes_shapes

        return all_weights_1d

    def from_1d(self, tensor_1d):
        if not self.dict_structure or not self.sizes_shapes:
            raise Exception("You have to call to_1d before transforming back to dict!")

        result_dict = self.dict_structure
        sizes, shapes = zip(*self.sizes_shapes)
        sizes, shapes = list(sizes), list(shapes)
        tensors_split = tf.split(tensor_1d, sizes, 0)
        tensors = [tf.reshape(tensor, shape) for tensor, shape in zip(tensors_split, shapes)]
        i = 0
        for layer_name in result_dict.keys():
            for weight_name in result_dict[layer_name].keys():
                result_dict[layer_name][weight_name] = tensors[i]
                i += 1
        return result_dict

    # Runtime of this is way to high to use in training
    # Possible solution: implement using only tf functions
    def preprocess_gradients(self, gradients, p):
        def scale(parameter):
            value = parameter.numpy()
            if value >= math.e ** -p:
                res = tf.math.log(parameter) / tf.constant(p, dtype=tf.float32)
            elif value <= -math.e ** -p:
                res = tf.math.log(tf.constant(-1, dtype=tf.float32) * parameter) / tf.constant(p, dtype=tf.float32)
            else:
                res = tf.constant(-math.e ** p) * parameter
            return res
        return tf.map_fn(scale, gradients)

if __name__ == "__main__":
    util = Util()

    grads = tf.constant([-1e-4])
    print(grads.numpy())
    grads = util.preprocess_gradients(grads, 10)
    print(grads.numpy())
    
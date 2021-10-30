import math
import time

import matplotlib.pyplot as plt
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

    # Runtime of this is pretty high (34826 parameters take ~2.5h per epoch)
    # Should therefore only be used in examples with way less objective parameters
    @tf.function
    def preprocess_gradients(self, gradients, p):
        tf_p = tf.constant(p, dtype=tf.float32)
        tf_e = tf.exp(tf.constant(1.0))

        def scale(param):
            cond1 = tf.greater_equal(param, tf.pow(tf_e, -1 * tf_p))
            cond2 = tf.less_equal(param, -1 * tf.pow(tf_e, -1 * tf_p))

            func1 = lambda: tf.math.log(param) / tf_p
            func2 = lambda: -1 * tf.math.log(-1 * param) / tf_p
            func3 = lambda: -1 * tf.pow(tf_e, tf_p) * param

            return tf.case([(cond1, func1), (cond2, func2)], default=func3)

        return tf.map_fn(scale, gradients)

if __name__ == "__main__":
    util = Util()

    t1 = time.time()
    a = tf.random.normal([34826])
    b = util.preprocess_gradients(a, 10)
    t2 = time.time()
    print(t2-t1)
    # plt.plot(a.numpy(), b.numpy())
    # plt.show()


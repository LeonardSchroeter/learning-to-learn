import matplotlib.pyplot as plt
import tensorflow as tf


class Util():
    def __init__(self):
        self.dict_structure = None
        self.sizes_shapes = None

    # converts a nested weights dict to a flat tensor
    # stores the dict structure and the shapes of the tensors in self.dict_structure and self.sizes_shapes to be able to convert back to dict
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

    # converts a flat tensor back to a nested weights dict
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

    # converts a nested weights dict to one flat tensor for each layer
    # stores the dict structure and the shapes of the tensors in self.dict_structure_per and self.sizes_shapes_per to be able to convert back to dict
    def to_1d_per_layer(self, weight_dict):
        dict_structure = {}
        sizes_shapes = []
        result = []
        for layer_name, layer_weights in weight_dict.items():
            weights_1d = []
            dict_structure[layer_name] = {}

            for weight_name, weights in layer_weights.items():
                dict_structure[layer_name][weight_name] = None

                size = tf.size(weights).numpy()
                shape = tf.shape(weights).numpy()
                sizes_shapes.append((size, shape))

                weights_1d.append(tf.reshape(weights, size))

            all_weights_1d = tf.concat(weights_1d, 0)
            result.append(all_weights_1d)

        self.dict_structure_per = dict_structure
        self.sizes_shapes_per = sizes_shapes

        return result

    # converts one flat tensor per layer back to a nested weights dict
    def from_1d_per_layer(self, tensor_arr):
        if not self.dict_structure_per or not self.sizes_shapes_per:
            raise Exception("You have to call to_1d before transforming back to dict!")

        tensor_1d = tf.concat(tensor_arr, 0)
        result_dict = self.dict_structure_per

        sizes, shapes = zip(*self.sizes_shapes_per)
        sizes, shapes = list(sizes), list(shapes)

        tensors_split = tf.split(tensor_1d, sizes, 0)
        tensors = [tf.reshape(tensor, shape) for tensor, shape in zip(tensors_split, shapes)]
        i = 0
        for layer_name in result_dict.keys():
            for weight_name in result_dict[layer_name].keys():
                result_dict[layer_name][weight_name] = tensors[i]
                i += 1
        return result_dict

# preprocessing function for the gradients of the objective function
@tf.function
def preprocess_gradients(gradients, p):
    tf_p = tf.constant(p, dtype=tf.float32)
    tf_e = tf.exp(tf.constant(1.0))

    cond = tf.greater_equal(tf.abs(gradients), tf.pow(tf_e, -1 * tf_p))

    x = tf.sign(gradients) * tf.math.log(tf.abs(gradients)) / tf_p
    y = -1 * tf.pow(tf_e, tf_p) * gradients

    return tf.where(cond, x=x, y=y)

# inverse preprocessing function
# used to convert the gradients of the objective function back to the original space
# we tried to use it to get a suitable pretraining domain but setting a normal distibution lead to better results
@tf.function
def preprocess_gradients_inverse(gradients, p):
    tf_p = tf.constant(p, dtype=tf.float32)
    tf_e = tf.exp(tf.constant(1.0))

    return -1 * tf.sign(gradients) * tf.pow(tf_e, -1 * tf_p * tf.abs(gradients))

# add a list of tensors weighted by the given weights
# used to sum the timesteps of the objective function in the experiments testing different weighting schemes
def weighted_sum(tensor_list, beta):
    num = len(tensor_list)
    weights = tf.linspace(1.0, beta * num + 1.0 - beta, num)

    return tf.reduce_sum(tf.stack(tensor_list, axis=0) * weights, axis=0)

if __name__ == "__main__":
    pass

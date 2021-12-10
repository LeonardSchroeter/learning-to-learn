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

@tf.function
def preprocess_gradients(gradients, p):
    tf_p = tf.constant(p, dtype=tf.float32)
    tf_e = tf.exp(tf.constant(1.0))

    cond = tf.greater_equal(tf.abs(gradients), tf.pow(tf_e, -1 * tf_p))

    x = tf.sign(gradients) * tf.math.log(tf.abs(gradients)) / tf_p
    y = -1 * tf.pow(tf_e, tf_p) * gradients

    return tf.where(cond, x=x, y=y)

@tf.function
def preprocess_gradients_inverse(gradients, p):
    tf_p = tf.constant(p, dtype=tf.float32)
    tf_e = tf.exp(tf.constant(1.0))

    return -1 * tf.sign(gradients) * tf.pow(tf_e, -1 * tf_p * tf.abs(gradients))

if __name__ == "__main__":
    inp = tf.random.normal([10], stddev=0.01)
    print(inp.numpy())
    out = preprocess_gradients(inp, 10)
    print(out.numpy())
    inp2 = preprocess_gradients_inverse(out, 10)
    print(inp2.numpy())
    print(inp.numpy() == inp2.numpy())

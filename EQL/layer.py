import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras import regularizers, initializers


def identity(out, index):
    return tf.identity(tf.gather(out, [index], axis=1), name='identity_output')


def sin(out, index):
    return tf.sin(tf.gather(out, [index], axis=1), name='sin_output')


def cos(out, index):
    return tf.cos(tf.gather(out, [index], axis=1), name='cos_output')
    
def div(out, index):
    sum1 = tf.gather(out, [index], axis=1)
    return tf.divide(tf.convert_to_tensor(1, dtype=tf.float32), sum1, name='div_output')
    
def mult(out, index):
    sum1 = tf.gather(out, [index], axis=1)
    sum2 = tf.gather(out, [index + 1], axis=1)
    return tf.multiply(sum1, sum2, name='mult_output')

def sphere(out, index):
    sum1 = tf.gather(out, [index], axis=1)
    sin = tf.sin(sum1)
    cos = tf.multiply(sum1, tf.cos(sum1))
    numerator = tf.add(sin,cos)
    numerator = tf.multiply(numerator, tf.convert_to_tensor(3, dtype = tf.float32))
    denom = tf.pow(sum1, tf.convert_to_tensor(3, dtype = tf.float32))

    frac = tf.divide(numerator, denom)
    
    return tf.pow(frac, tf.convert_to_tensor(2, dtype = tf.float32), name = 'sphere_output')
    
def lorentz(out, index):
    sum1 = tf.gather(out, [index], axis=1)
    denom = tf.add(tf.convert_to_tensor(1, dtype = tf.float32), tf.pow(sum1, tf.convert_to_tensor(2, dtype = tf.float32)))
    return tf.divide(tf.convert_to_tensor(1, dtype = tf.float32), denom, name = 'lorentz_output')
def oz(out, index):
    sum1 = tf.gather(out, [index], axis=1)
    denom = tf.add(tf.convert_to_tensor(1, dtype = tf.float32), tf.pow(sum1, tf.convert_to_tensor(2, dtype = tf.float32)))
    return tf.divide(tf.convert_to_tensor(1, dtype = tf.float32), denom, name = 'oz_output')
def ts(out, index):
    sum1 = tf.gather(out, [index], axis=1)
    sum2 = tf.gather(out, [index+1], axis=1)
    num = tf.convert_to_tensor(8* (math.pi), dtype = tf.float32)
    term1 = tf.pow(sum1, tf.convert_to_tensor(2, dtype = tf.float32))
    term2 = tf.pow(sum2, tf.convert_to_tensor(2, dtype = tf.float32))
    denom = tf.multiply(term1, term2)
    return tf.divide(num, denom, name = 'ts_output')
class EqlLayer(keras.layers.Layer):
    def __init__(self, w_initializer, b_initializer, v, lmbda=0, mask=None, exclude=None):
        super(EqlLayer, self).__init__()
        if exclude is None:
            exclude = []
        self.regularizer = regularizers.L1(l1=lmbda)
        self.w_initializer = initializers.get(w_initializer)
        self.b_initializer = initializers.get(b_initializer)
        self.mask = mask
        self.v = v
        self.activations = [identity, sin, div, cos, mult, sphere, lorentz, oz, ts]

        self.exclusion = 0
        if 'id' in exclude:
            self.exclusion += 1
            self.activations.remove(identity)
        if 'sin' in exclude:
            self.exclusion += 1
            self.activations.remove(sin)
        if 'cos' in exclude:
            self.exclusion += 1
            self.activations.remove(cos)
        if 'mult' in exclude:
            self.exclusion += 2
            self.activations.remove(mult)
        if 'div' in exclude:
            self.exclusion += 1
            self.activations.remove(div)
        if 'sphere' in exclude:
            self.exclusion += 1
            self.activations.remove(sphere)
        if 'lorentz' in exclude:
            self.exclusion += 1
            self.activations.remove(lorentz)
        if 'oz' in exclude:
            self.exclusion += 1
            self.activations.remove(oz)
        if 'ts' in exclude:
            self.exclusion += 2
            self.activations.remove(ts)

    def _mask(self):
        for i in range(self.w.shape[0]):
            w_mask = tf.matmul([self.w[i]], self.mask[0][i])[0]
            self.w[i].assign(w_mask)
        b_mask = tf.matmul([self.b], self.mask[1])[0]
        self.b.assign(b_mask)

    def build(self, input_shape):
        isZeroes = True
        self.w = self.add_weight(
            shape=(input_shape[-1], 11 * self.v - self.v * self.exclusion),
            initializer=self.w_initializer,
            trainable=True, regularizer=self.regularizer
        )
        if self.b_initializer == 'zeros':
            isZeroes = False
        self.b = self.add_weight(
            shape=(11 * self.v - self.v * self.exclusion,), initializer=self.b_initializer,
            trainable=isZeroes, regularizer=self.regularizer
        )

    def call(self, inputs):
        if self.mask:
            for i in range(self.w.shape[0]):
                w_mask = tf.matmul([self.w[i]], self.mask[0][i])[0]
                self.w[i].assign(w_mask)
            b_mask = tf.matmul([self.b], self.mask[1])[0]
            self.b.assign(b_mask)

        
        output_batches = []
        for i in range(self.v):
            v = (11 - self.exclusion) * i
            for a in range(len(self.activations)):
                act = str(self.activations[a])

                check3 = act.find("ts")
                if check3 != -1:
                    new_inputs = tf.pow(inputs, tf.convert_to_tensor(2, dtype = tf.float32))
                    out = tf.matmul(new_inputs, self.w) + self.b
                    activation = self.activations[a](out, a + v)
                else:
                    out = tf.matmul(inputs, self.w) + self.b
                    activation = self.activations[a](out, a + v)
                output_batches.append(activation)
        output = tf.concat(output_batches, axis=1)
        return output


class DenseLayer(keras.layers.Layer):
    def __init__(self, w_initializer, b_initializer, lmbda=0, mask=None):
        super(DenseLayer, self).__init__()
        self.regularizer = regularizers.L1(l1=lmbda)
        self.w_initializer = initializers.get(w_initializer)
        self.b_initializer = initializers.get(b_initializer)
        self.mask = mask


    def _mask(self):
        for i in range(self.w.shape[0]):
            w_mask = tf.matmul([self.w[i]], self.mask[0][i])[0]
            self.w[i].assign(w_mask)
        b_mask = tf.matmul([self.b], self.mask[1])[0]
        self.b.assign(b_mask)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 1), #TODO: Output of dense layer is 1, maybe change this for multi-dimensionality
            initializer=self.w_initializer,
            trainable=True, regularizer=self.regularizer
        )
        self.b = self.add_weight(
            shape=(1,), initializer=self.b_initializer, trainable=True, regularizer=self.regularizer
        )

    def call(self, inputs):
        if self.mask:
            for i in range(self.w.shape[0]):
                w_mask = tf.matmul([self.w[i]], self.mask[0][i])[0]
                self.w[i].assign(w_mask)
            b_mask = tf.matmul([self.b], self.mask[1])[0]
            self.b.assign(b_mask)
        out = tf.matmul(inputs, self.w) + self.b
        return out

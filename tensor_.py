#%%

from tensorflow import keras

import tensorflow as tf


def Initialize_One_Variable(units):

  w_init = tf.random_uniform_initializer()

  R_kernal = tf.Variable(initial_value=w_init(shape=(units, units)),trainable=True,)

  return R_kernal


def Initialize_Variable(input_dim, units,):

  w_init = tf.random_normal_initializer()
  b_init = tf.zeros_initializer()

  w_0 = tf.Variable(initial_value=w_init(shape=(input_dim, units)), trainable=True,)

  b_0 = tf.Variable(initial_value=b_init(shape=(units)), trainable=True)

  return w_0, b_0


class Custom_Layer(keras.layers.Layer):

  def __init__(self, input_tuple, **kwargs):

    super(Custom_Layer, self).__init__()
    input_shape, units = input_tuple

    self.Hidden_Size = (int)(input_shape * 0.5)
    self.inputshape = input_shape
    self.units = units


    self.Uz = Initialize_One_Variable(self.Hidden_Size)

    self.Ur = Initialize_One_Variable(self.Hidden_Size)

    self.Uh = Initialize_One_Variable(self.Hidden_Size)

    self.wz, self.bz = Initialize_Variable(self.inputshape,self.Hidden_Size)

    self.wr, self.br = Initialize_Variable(self.inputshape,self.Hidden_Size)

    self.wh, self.bh = Initialize_Variable(self.inputshape,self.Hidden_Size)

    self.w_out, self.b_out = Initialize_Variable(self.Hidden_Size,self.units)


  def get_config(self):

    cfg = super().get_config()

    return cfg


  def Custom_Method(self, step_input, step_state, training):

    r = tf.sigmoid(tf.matmul(step_input,self.wr) + tf.matmul(step_state, self.Ur) + self.br)

    z = tf.sigmoid(tf.matmul(step_input,self.wz) + tf.matmul(step_state, self.Uz) + self.bz)

    h__ = tf.nn.relu(tf.matmul(step_input, self.wh) + tf.matmul(tf.multiply(r, step_state),self.Uh) + self.bh)

    h = (1-z) * h__ + z * step_state

    output__ = tf.nn.relu(tf.matmul(h, self.w_out) + self.b_out)
    return output__, h


  def call(self, inputs, training=False):

    unstack = tf.unstack(inputs, axis=1)

    out1, hiddd = self.Custom_Method(unstack[0], tf.zeros_like(unstack[0][:,0:self.Hidden_Size]),training=training)

    out2, hiddd = self.Custom_Method(unstack[1], hiddd,training=training)

    out3, hiddd = self.Custom_Method(unstack[2], hiddd,training=training)

    out4, hiddd = self.Custom_Method(unstack[3], hiddd,training=training)

    return out4

  Layer___ = Custom_Layer((12,9))

  randomt = tf.random.uniform(shape=(64,4,7))

  Layer___(randomt)
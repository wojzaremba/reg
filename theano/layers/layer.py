import numpy as np
import theano
import theano.tensor as T

from theano.tensor.nnet import conv

#class ActL(Layer):
#class Bundle(object):
#class LogRegB(Bundle):


class Layer(object):
  def __init__(self):
    self.output = None

class Cost(object):
  def __init__(self):
    self.output = None
    self.prob = None
    self.pred = None

class FCL(Layer):
  def __init__(self, n_in, n_out):
    super(FCL, self).__init__()
    self.W = theano.shared(value=np.zeros((n_in, n_out),
                         dtype=theano.config.floatX),
                 name='W', borrow=True)
    self.params = [self.W]

  def fp(self, x):
    self.output = T.dot(x, self.W)

class ConvL(Layer):
  def __init__(self, rng, filter_shape, image_shape):
    super(ConvL, self).__init__()
    assert image_shape[1] == filter_shape[1]
    fan_in = np.prod(filter_shape[1:])
    self.filter_shape = filter_shape
    self.image_shape = image_shape
    W_values = np.asarray(rng.uniform(
      low=-np.sqrt(3./fan_in),
      high=np.sqrt(3./fan_in),
      size=filter_shape), dtype=theano.config.floatX)
    self.W = theano.shared(value=W_values, name='W')
    b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
    self.b = theano.shared(value=b_values, name='b')
    self.params = [self.W, self.b]

  def fp(self, x):
    self.output = conv.conv2d(x, self.W,
      filter_shape=self.filter_shape, image_shape=self.image_shape)

class SoftmaxC(Cost):
  def __init__(self):
    super(SoftmaxC, self).__init__()
    self.params = []

  def fp(self, x, y):
    self.prob = T.nnet.softmax(x)
    self.pred = T.argmax(self.prob, axis=1)
    self.output = -T.mean(T.log(self.prob)[T.arange(y.shape[0]), y])

  def errors(self, y):
    if y.ndim != self.pred.ndim:
      raise TypeError('y should have the same shape as self.pred',
        ('y', y.type, 'pred', self.pred.type))
    if y.dtype.startswith('int'):
      return T.mean(T.neq(self.pred, y))
    else:
      raise NotImplementedError()

class BiasL(Layer):
  def __init__(self, n_out):
    super(BiasL, self).__init__()
    self.b = theano.shared(value=np.zeros((n_out,),
                         dtype=theano.config.floatX),
                 name='b', borrow=True)
    self.params = [self.b]

  def fp(self, x):
    self.output = x + self.b


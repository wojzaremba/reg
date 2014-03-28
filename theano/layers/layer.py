import numpy as np
import theano
import theano.tensor as T
from numpy import random
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

  def pred(self):
    return T.argmax(self.prob, axis=1)

  def errors(self, y):
    pred = self.pred()
    if y.ndim != pred.ndim:
      raise TypeError('y should have the same shape as self.pred()',
        ('y', y.type, 'pred', pred.type))
    if y.dtype.startswith('int'):
      return T.mean(T.neq(pred, y))
    else:
      raise NotImplementedError()

class FCL(Layer):
  def __init__(self, n_in, n_out):
    super(FCL, self).__init__()
    self.W = theano.shared(value=np.zeros((n_in, n_out),
                         dtype=theano.config.floatX),
                 name='W', borrow=True)
    self.params = [self.W]

  def fp(self, x, _):
    x_tmp = T.flatten(x, 2)
    self.output = T.dot(x_tmp, self.W)

class ConvL(Layer):
  def __init__(self, filter_shape, image_shape,
               subsample=(1, 1), border_mode='full'):
    super(ConvL, self).__init__()
    assert image_shape[1] == filter_shape[1]
    fan_in = np.prod(filter_shape[1:])
    self.filter_shape = filter_shape
    self.image_shape = image_shape
    W_values = np.asarray(random.uniform(
      low=-np.sqrt(3./fan_in),
      high=np.sqrt(3./fan_in),
      size=filter_shape), dtype=theano.config.floatX)
    self.W = theano.shared(value=W_values, name='W')
    self.subsample = subsample
    self.border_mode = border_mode
    self.params = [self.W]

  def fp(self, x, _):
    self.output = conv.conv2d(x, self.W,
      filter_shape=self.filter_shape, image_shape=self.image_shape,
      subsample=self.subsample, border_mode=self.border_mode)

class SoftmaxC(Cost):
  def __init__(self):
    super(SoftmaxC, self).__init__()
    self.params = []

  def fp(self, x, y):
    self.prob = T.nnet.softmax(x)
    self.output = -T.mean(T.log(self.prob)[T.arange(y.shape[0]), y])

class BiasL(Layer):
  def __init__(self, n_out):
    super(BiasL, self).__init__()
    self.b = theano.shared(value=np.zeros((n_out,),
                         dtype=theano.config.floatX),
                 name='b', borrow=True)
    self.params = [self.b]

  def fp(self, x, _):
    self.output = x + self.b


class ActL(Layer):
  def __init__(self, f):
    super(ActL, self).__init__()
    self.f = f
    self.params = []

  def fp(self, x, _):
    self.output = self.f(x)

class ReluL(ActL):
  def __init__(self):
    relu = lambda x: T.maximum(x, 0)
    ActL.__init__(self, relu)


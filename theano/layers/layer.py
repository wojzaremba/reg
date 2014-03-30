import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from numpy import random
from theano.tensor.nnet import conv
import cache
from os.path import expanduser

class Layer(object):
  def __init__(self, in_shape):
    self.output = None
    self.in_shape = in_shape

class Cost(Layer):
  def __init__(self, in_shape):
    Layer.__init__(self, in_shape)
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
  def __init__(self, out_len, in_shape=None):
    Layer.__init__(self, in_shape)
    in_len = reduce(lambda x, y: x * y, list(self.in_shape)[1:])
    self.W = theano.shared(value=np.zeros((in_len, out_len),
                           dtype=theano.config.floatX),
                           name='W', borrow=True)
    self.out_shape = (in_shape[0], out_len)
    self.params = [self.W]

  def fp(self, x, _):
    x_tmp = T.flatten(x, 2)
    self.output = T.dot(x_tmp, self.W)

class ConvL(Layer):
  def __init__(self, filter_shape, subsample=(1, 1),
               border_mode='full', in_shape=None):
    Layer.__init__(self, in_shape)
    assert in_shape[1] == filter_shape[1]
    fan_in = np.prod(filter_shape[1:])
    self.filter_shape = filter_shape
    self.in_shape = in_shape
    W_values = np.asarray(random.uniform(
      low=-np.sqrt(3./fan_in),
      high=np.sqrt(3./fan_in),
      size=filter_shape), dtype=theano.config.floatX)
    self.W = theano.shared(value=W_values, name='W')
    self.subsample = subsample
    self.border_mode = border_mode
    # TODO: Doesn't care about border.
    self.out_shape = (in_shape[0],
                      filter_shape[0],
                      (in_shape[2] - filter_shape[2]) / subsample[0] + 1,
                      (in_shape[3] - filter_shape[3]) / subsample[1] + 1)
    self.params = [self.W]

  def fp(self, x, _):
    self.output = conv.conv2d(x, self.W,
      filter_shape=self.filter_shape, image_shape=self.in_shape,
      subsample=self.subsample, border_mode=self.border_mode)

class SoftmaxC(Cost):
  def __init__(self, in_shape=None):
    Cost.__init__(self, in_shape)
    self.out_shape = self.in_shape
    self.params = []

  def fp(self, x, y):
    self.prob = T.nnet.softmax(x)
    self.output = -T.mean(T.log(self.prob)[T.arange(y.shape[0]), y])

class BiasL(Layer):
  def __init__(self, in_shape=None):
    Layer.__init__(self, in_shape)
    self.b = theano.shared(value=np.zeros((self.in_shape[1],),
                           dtype=theano.config.floatX),
                           name='b', borrow=True)
    self.out_shape = self.in_shape
    self.params = [self.b]

  def fp(self, x, _):
    self.output = x + self.b


class ActL(Layer):
  def __init__(self, f, in_shape):
    Layer.__init__(self, in_shape)
    self.f = f
    self.out_shape = self.in_shape
    self.params = []

  def fp(self, x, _):
    self.output = self.f(x)

class ReluL(ActL):
  def __init__(self, in_shape):
    relu = lambda x: T.maximum(x, 0)
    ActL.__init__(self, relu, in_shape)

class MaxpoolL(Layer):
  def __init__(self, pool_shape, ignore_border=True, in_shape=None):
    Layer.__init__(self, in_shape)
    self.pool_shape = pool_shape
    self.ignore_border = ignore_border
    self.out_shape = (self.in_shape[0], self.in_shape[1],
                      self.in_shape[2] / pool_shape[0],
                      self.in_shape[3] / pool_shape[1])
    self.params = []

  def fp(self, x, _):
    self.output = downsample.max_pool_2d(x, self.pool_shape, self.ignore_border)

class Source(object):
  def __init__(self, dataset, batch_size):
    self.dataset = dataset
    self.train_x, self.train_y, self.test_x, self.test_y = self.load_data()
    data_shape = self.train_x.get_value(borrow=True).shape
    self.out_shape = (batch_size, data_shape[1], data_shape[2], data_shape[3])
    self.n_train_batches = \
      self.train_x.get_value(borrow=True).shape[0] / batch_size
    self.n_test_batches = \
      self.test_x.get_value(borrow=True).shape[0] / batch_size

  def load_data(self):
    floatX = theano.config.floatX
    print '\tloading data', self.dataset
    if cache.DATA.has_key(self.dataset):
      print '\tloading dataset from cache'
      rval = cache.DATA[self.dataset]
    else:
      home = expanduser("~")
      fname = "%s/data/%s/data.pkl.gz" % (home, self.dataset)
      f = gzip.open(fname, 'rb')
      train_set, test_set = cPickle.load(f)
      f.close()
      def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        data_array_x = np.asarray(data_x, dtype=floatX)
        data_array_y = np.asarray(data_y, dtype=floatX)
        shared_x = theano.shared(data_array_x, borrow=borrow)
        shared_y = theano.shared(data_array_y, borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
      test_x, test_y = shared_dataset(test_set)
      train_x, train_y = shared_dataset(train_set)
      rval = (train_x, train_y, test_x, test_y)
      cache.DATA[self.dataset] = rval
    print "\ttrain set size:", rval[0].get_value(borrow=True).shape
    print "\ttest set size:", rval[2].get_value(borrow=True).shape
    return rval

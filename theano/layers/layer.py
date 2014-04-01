import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from numpy import random
from theano.tensor.nnet import conv
import config

class Layer(object):
  def __init__(self, prev_layer):
    self.output = None
    self.prev = []
    self.in_shape = None
    self.succ = []
    if prev_layer is not None:
      self.in_shape = prev_layer.out_shape
      self.prev.append(prev_layer)
      prev_layer.succ.append(self)

# Some layers behave differently during testing
# than training (e.g. Dropout).
  def fptest(self, x, y):
    return self.fp(x, y)

  def attach(self, layer, params):
    if layer.__class__.__name__ == "function":
      print "Bundle %s" % layer.__name__
    params['prev_layer'] = self
    l = layer(**params)
    if layer.__class__.__name__ != "function":
      print "Layer %s, input shape %s" % (layer.__name__, str(l.in_shape))
    return l

  def get_updates(self, lr, cost):
    updates = []
    for l in self.succ:
      updates += l.get_updates(lr, cost)
    for p in self.params:
      g = T.grad(cost=cost, wrt=p)
      updates.append((p, p - lr * g))
    return updates

  def get_costs(self, (x, y), train):
    # Due to circular dependency.
    from layers.cost import Cost
    if isinstance(self, Cost):
      costs = [self]
    else:
      costs = []
    for l in self.succ:
      if train:
        l.fp(x, y)
      else:
        l.fptest(x, y)
      costs = costs + l.get_costs((l.output, y), train)
    unique = []
    for item in costs:
      if item not in unique:
        unique.append(item)
    return unique

  def dump(self):
    params = []
    for p in self.params:
      params.append(p.get_value(borrow=True))
    parent_params = []
    for l in self.succ:
      parent_params.append(l.dump())
    return (params, parent_params)

  def load(self, (params, parent_params)):
    for p_idx in xrange(len(self.params)):
      self.params[p_idx].set_value(params[p_idx])
    for l_idx in xrange(len(self.succ)):
      self.succ[l_idx].load(parent_params[l_idx])

class DropoutL(Layer):
  def __init__(self, p=0.5, prev_layer=None):
    Layer.__init__(self, prev_layer)
    self.p = p
    self.out_shape = self.in_shape
    self.rng = np.random.RandomState(1)
    self.params = []

  def fp(self, x, _):
    srng = T.shared_randomstreams.RandomStreams(self.rng.randint(2))
    mask = srng.binomial(n=1, p=1-self.p, size=self.in_shape)
    self.output = x * T.cast(mask, theano.config.floatX)

  def fptest(self, x, _):
    self.output = x * self.p

class FCL(Layer):
  def __init__(self, out_len, prev_layer=None):
    Layer.__init__(self, prev_layer)
    in_len = reduce(lambda x, y: x * y, list(self.in_shape)[1:])
    self.W = theano.shared(value=0.1 * np.random.randn(in_len, out_len),
                           name='W', borrow=True)
    self.out_shape = (self.in_shape[0], out_len)
    self.params = [self.W]

  def fp(self, x, _):
    x_tmp = T.flatten(x, 2)
    self.output = T.dot(x_tmp, self.W)

class ConvL(Layer):
  def __init__(self, filter_shape, subsample=(1, 1),
               border_mode='full', prev_layer=None):
    Layer.__init__(self, prev_layer)
    assert self.in_shape[1] == filter_shape[1]
    fan_in = np.prod(filter_shape[1:])
    self.filter_shape = filter_shape
    W_values = np.asarray(random.uniform(
      low=-np.sqrt(3./fan_in),
      high=np.sqrt(3./fan_in),
      size=filter_shape), dtype=theano.config.floatX)
    self.W = theano.shared(value=W_values, name='W')
    self.subsample = subsample
    self.border_mode = border_mode
    # TODO: Doesn't care about border.
    self.out_shape = (self.in_shape[0],
                      filter_shape[0],
                      (self.in_shape[2] - filter_shape[2]) / subsample[0] + 1,
                      (self.in_shape[3] - filter_shape[3]) / subsample[1] + 1)
    self.params = [self.W]

  def fp(self, x, _):
    self.output = conv.conv2d(x, self.W,
      filter_shape=self.filter_shape, image_shape=self.in_shape,
      subsample=self.subsample, border_mode=self.border_mode)

class BiasL(Layer):
  def __init__(self, prev_layer=None):
    Layer.__init__(self, prev_layer)
    self.b = theano.shared(value=np.zeros((self.in_shape[1],),
                           dtype=theano.config.floatX),
                           name='b', borrow=True)
    self.out_shape = self.in_shape
    self.params = [self.b]

  def fp(self, x, _):
    if len(self.in_shape) == 4:
      self.output = x + self.b.dimshuffle('x', 0, 'x', 'x')
    elif len(self.in_shape) == 2:
      self.output = x + self.b
    else:
      assert False

class ActL(Layer):
  def __init__(self, f, prev_layer=None):
    Layer.__init__(self, prev_layer)
    self.f = f
    self.out_shape = self.in_shape
    self.params = []

  def fp(self, x, _):
    self.output = self.f(x)

class ReluL(ActL):
  def __init__(self, prev_layer=None):
    relu = lambda x: T.maximum(x, 0)
    ActL.__init__(self, relu, prev_layer)


class MaxpoolL(Layer):
  def __init__(self, pool_shape, ignore_border=True, prev_layer=None):
    Layer.__init__(self, prev_layer)
    self.pool_shape = pool_shape
    self.ignore_border = ignore_border
    self.out_shape = (self.in_shape[0], self.in_shape[1],
                      self.in_shape[2] / pool_shape[0],
                      self.in_shape[3] / pool_shape[1])
    self.params = []

  def fp(self, x, _):
    self.output = downsample.max_pool_2d(x, self.pool_shape, self.ignore_border)

class LRCrossmapL(Layer):
  def __init__(self, size, scale=0.001, power=0.75, prev_layer=None):
    Layer.__init__(self, prev_layer)
    self.out_shape = self.in_shape
    self.scale = scale
    self.power = power
    self.size = size
    self.params = []

  def fp(self, x, _):
    N = self.in_shape[1]
    output = x
    for f in range(0, N):
      T.set_subtensor(output[:, f, :, :],
		x[:, f, :, :] /
		(1 + (self.scale / self.size)
		* T.sqr(x[:, max(0, f - self.size/2) : min(N, f + self.size/2),
				  :, :]).sum(axis=1))**self.power)
    self.output = output

class LRSpatialL(Layer):
  def __init__(self, size, scale=0.001, power=0.75, prev_layer=None):
    Layer.__init__(self, prev_layer)
    self.out_shape = self.in_shape
    self.scale = scale
    self.power = power
    self.size = size
    self.params = []

  def fp(self, x, _):
    X, Y = self.in_shape[2:]
    output = x
    for i in range(0, X):
      for j in range(0, Y):
        T.set_subtensor(output[:, :, i, j],
		  x[:, :, i, j] /
		  (1 + (self.scale / self.size**2)
		  * T.sqr(x[:, :, max(0, i - self.size/2) : min(X, i + self.size/2),
				    max(0, j - self.size/2) : min(Y, j + self.size/2)])
				.sum(axis=(2, 3)))**self.power)
    self.output = output

class Source(Layer):
  def __init__(self, dataset, batch_size):
    Layer.__init__(self, None)
    self.dataset = dataset
    self.train_x, self.train_y, self.test_x, self.test_y = self.load_data()
    data_shape = self.train_x.get_value(borrow=True).shape
    self.out_shape = (batch_size, data_shape[1], data_shape[2], data_shape[3])
    self.n_train_batches = \
      self.train_x.get_value(borrow=True).shape[0] / batch_size
    self.n_test_batches = \
      self.test_x.get_value(borrow=True).shape[0] / batch_size
    self.params = []

  def fp(self, x, _):
    self.output = x

  def load_data(self):
    floatX = theano.config.floatX
    fname = "%s/%s/data.pkl.gz" % (config.DATA_DIR, self.dataset)
    print '\tloading data %s from %s' % (self.dataset, fname)
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
    print "\ttrain set size:", train_x.get_value(borrow=False).shape
    print "\ttest set size:", test_x.get_value(borrow=False).shape
    print ""
    return rval

import theano.tensor as T
from layers.layer import Layer

class Cost(Layer):
  def __init__(self, is_classifier, prev_layer):
    Layer.__init__(self, prev_layer)
    self.is_classifier = is_classifier
    self.out_shape = (1,)
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

class SoftmaxC(Cost):
  def __init__(self, prev_layer=None):
    Cost.__init__(self, True, prev_layer)
    self.params = []

  def fp(self, x, y):
    self.prob = T.nnet.softmax(x)
    self.output = -T.mean(T.log(self.prob)[T.arange(y.shape[0]), y])


class WL2C(Cost):
  def __init__(self, alpha=1, prev_layer=None):
    Cost.__init__(self, False, prev_layer)
    self.alpha = alpha
    self.params = []

  def fp(self, *_):
    self.output = self.alpha * (self.prev[0].W ** 2).sum()




from layers.layer import Layer, FCL, BiasL, ReluL, SoftmaxC, Cost, ConvL

class Bundle(Layer):
  def __init__(self, is_cost=False, in_shape=None):
    Layer.__init__(self, in_shape)
    self.is_cost = is_cost
    self.out_shape = None
    self.prob = None
    self.bundle = []
    self.params = []

  def fp(self, x, y=None):
    x_out = x
    for b in self.bundle:
      b.fp(x_out, y)
      x_out = b.output
    self.output = x_out
    if self.is_cost:
      self.prob = self.bundle[-1].prob

  def append(self, layer, params):
    print "\tAdding sublayer", layer.__name__
    in_shape = None
    in_shape = self.in_shape
    if len(self.bundle) > 0:
      in_shape = self.bundle[-1].out_shape
    params['in_shape'] = in_shape
    l = layer(**params)
    self.out_shape = l.out_shape
    self.bundle.append(l)
    for p in l.params:
      self.params.append(p)
    print "\t\tinput shape: %s, output shape: %s" % \
      (str(in_shape), str(l.out_shape))

class FCB(Bundle):
  def __init__(self, out_len, in_shape=None):
    Bundle.__init__(self, False, in_shape)
    self.append(FCL, {'out_len': out_len})
    self.append(BiasL, {})
    self.append(ReluL, {})

class ConvB(Bundle):
  def __init__(self, filter_shape,
               subsample=(1, 1), border_mode='full', in_shape=None):
    Bundle.__init__(self, False, in_shape)
    self.append(ConvL, {'filter_shape':filter_shape,
                        'subsample':subsample,
                        'border_mode':border_mode})
    self.append(BiasL, {})
    self.append(ReluL, {})

# XXX : Somehow should know about number of labels.
class SoftmaxBC(Cost, Bundle):
  def __init__(self, out_len=10, in_shape=None):
    Cost.__init__(self, in_shape)
    Bundle.__init__(self, True, in_shape)
    self.append(FCL, {'out_len': out_len})
    self.append(BiasL, {})
    self.append(SoftmaxC, {})

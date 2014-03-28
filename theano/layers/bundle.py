from layers.layer import Layer, FCL, BiasL, ReluL, SoftmaxC, Cost

# XXX: act missing.

class Bundle(Layer):
  def __init__(self, bundle, is_cost=False):
    super(Bundle, self).__init__()
    self.bundle = bundle
    assert len(self.bundle) >= 2
    self.is_cost = is_cost
    self.prob = None
    self.params = []
    for b in self.bundle:
      for p in b.params:
        self.params.append(p)

  def fp(self, x, y=None):
    x_out = x
    for b in self.bundle:
      b.fp(x_out, y)
      x_out = b.output
    self.output = x_out
    if self.is_cost:
      self.prob = self.bundle[-1].prob

class FCB(Bundle):
  def __init__(self, n_in, n_out):
    super(FCB, self).__init__([FCL(n_in, n_out),
                               BiasL(n_out),
                               ReluL()])

class SoftmaxBC(Bundle, Cost):
  def __init__(self, n_in, n_labels):
    Bundle.__init__(self, [FCL(n_in, n_labels),
                           BiasL(n_labels),
                           SoftmaxC()],
                    True)
    Cost.__init__(self)

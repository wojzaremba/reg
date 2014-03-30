import time

import numpy as np

import theano
import theano.tensor as T

class Model(object):

  def __init__(self, lr=0.13, n_epochs=100, batch_size=100):
    print "_" * 100
    print "Creating model lr=%f, n_epochs=%d, batch_size=%d" % \
      (lr, n_epochs, batch_size)
    self.lr = lr
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.out_shape = None
    self.source = None
    self.layers = []
    # XXX: if something is a cost add it here. Sum up all costs at the end.
    self.costs = []

  def set_source(self, source, params):
    params['batch_size'] = self.batch_size
    self.source = source(**params)

  def append(self, layer, params):
    print "Adding layer", layer.__name__
    if len(self.layers) > 0:
      in_shape = self.layers[-1].out_shape
    else:
      in_shape = self.source.out_shape
    params['in_shape'] = in_shape
    l = layer(**params)
    self.out_shape = l.out_shape
    self.layers.append(l)
    print "\tinput shape: %s, output shape: %s" % \
      (str(in_shape), str(l.out_shape))

  def build_model(self):
    print '\n... building the model'
    idx = T.lscalar()
    bs = self.batch_size
    x = T.tensor4('x')
    y = T.ivector('y')
    x_out = x
    for i in range(0, len(self.layers)):
      l = self.layers[i]
      l.fp(x_out, y)
      x_out = l.output
    costL = self.layers[-1]
    outputs = [costL.errors(y), costL.output]
    test_model = theano.function(inputs=[idx], outputs=outputs,
                                 givens={
      x: self.source.test_x[idx * bs: (idx + 1) * bs],
      y: self.source.test_y[idx * bs: (idx + 1) * bs]})
    updates = []
    for l in self.layers:
      for p in l.params:
        g = T.grad(cost=costL.output, wrt=p)
        updates.append((p, p - self.lr * g))

    train_model = theano.function(inputs=[idx],
      outputs=outputs,
      updates=updates,
      givens={
        x: self.source.train_x[idx * bs:(idx + 1) * bs],
        y: self.source.train_y[idx * bs:(idx + 1) * bs]})
    return (train_model, test_model)

  def train(self):
    train_model, test_model = self.build_model()
    print '... training the model'
    test_frequency = self.source.n_train_batches / 5

    for epoch in xrange(self.n_epochs):
      start_time = time.clock()
      train_res = []
      for minibatch_index in xrange(self.source.n_train_batches):
        train_res.append(train_model(minibatch_index))
        it = (epoch - 1) * self.source.n_train_batches + minibatch_index
        if (it + 1) % test_frequency == 0:
          test_res = \
            zip(*[test_model(i) for i in xrange(self.source.n_test_batches)])

          train_error = np.mean(zip(*train_res)[0]) * 100.
          train_loss = np.mean(zip(*train_res)[1])
          test_error = np.mean(test_res[0]) * 100.
          test_loss = np.mean(test_res[1])
          print '\ttrain err=%.2f%%,loss=%.5f;  test err=%.2f%%,loss=%.5f''' % \
                     (train_error, train_loss, test_error, test_loss)
      end_time = time.clock()
      print "Epoch %d took %.1fs" % (epoch, end_time - start_time)

    print "Training finished !"


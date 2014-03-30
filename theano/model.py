import os
import os.path
import sys
import time

import numpy as np

import theano
import theano.tensor as T

class Model(object):

  def __init__(self, lr=0.13, n_epochs=1000, batch_size=100):
    print "_" * 100
    print "Creating model lr = %f, batch_size = %d" % (lr, batch_size)
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

    test_model = theano.function(inputs=[idx], outputs=costL.errors(y),
                                 givens={
      x: self.source.test_x[idx * bs: (idx + 1) * bs],
      y: self.source.test_y[idx * bs: (idx + 1) * bs]})
    updates = []
    for l in self.layers:
      for p in l.params:
        g = T.grad(cost=costL.output, wrt=p)
        updates.append((p, p - self.lr * g))

    train_model = theano.function(inputs=[idx],
      outputs=costL.output,
      updates=updates,
      givens={
        x: self.source.train_x[idx * bs:(idx + 1) * bs],
        y: self.source.train_y[idx * bs:(idx + 1) * bs]})
    return (train_model, test_model)

# XXX: Don't apply fp to train data, but just look on proper output
# (relie on polymorpism of cost.errors).
  def train(self):
    train_model, test_model = self.build_model()
    print '... training the model'
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    test_frequency = 10#min(self.source.n_train_batches, patience / 2)

    best_test_loss = np.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < self.n_epochs) and (not done_looping):
      epoch = epoch + 1
      for minibatch_index in xrange(self.source.n_train_batches):
        train_model(minibatch_index)
        it = (epoch - 1) * self.source.n_train_batches + minibatch_index

        if (it + 1) % test_frequency == 0:
          test_losses = \
            [test_model(i) for i in xrange(self.source.n_test_batches)]
          this_test_loss = np.mean(test_losses)

          print('epoch %i, minibatch %i/%i, test error %f %%' % \
            (epoch, minibatch_index + 1, self.source.n_train_batches,
            this_test_loss * 100.))

          if this_test_loss < best_test_loss:
            if this_test_loss < best_test_loss * \
               improvement_threshold:
              patience = max(patience, it * patience_increase)
            best_test_loss = this_test_loss
            test_losses = \
              [test_model(i) for i in xrange(self.source.n_test_batches)]
            test_score = np.mean(test_losses)

            print(('\tepoch %i, minibatch %i/%i, test error of best'
               ' model %f %%') %
              (epoch, minibatch_index + 1, self.source.n_train_batches,
               test_score * 100.))

        if patience <= it:
          done_looping = True
          break

    end_time = time.clock()
    print 'Optimization complete with best test score of %f' \
      % (best_test_loss * 100.)
    print 'The code run for %d epochs, with %f epochs/sec' % (
      epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                os.path.split(__file__)[1] +
                ' ran for %.1fs' % ((end_time - start_time)))


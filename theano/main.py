import cPickle
import gzip
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from os.path import expanduser
from layers.layer import FCL, BiasL, SoftmaxC

floatX = theano.config.floatX

def load_data(dataset):
  print '... loading data'
  home = expanduser("~")
  f = gzip.open(home + '/data/' + dataset + '/data.pkl.gz', 'rb')
  train_set, valid_set, test_set = cPickle.load(f)
  f.close()

  def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

  test_set_x, test_set_y = shared_dataset(test_set)
  valid_set_x, valid_set_y = shared_dataset(valid_set)
  train_set_x, train_set_y = shared_dataset(train_set)

  rval = [(train_set_x, train_set_y),
          (valid_set_x, valid_set_y),
          (test_set_x, test_set_y)]
  return rval


def sgd(model, lr=0.13, n_epochs=1000, dataset='mnist', batch_size=600):
  datasets = load_data(dataset)

  train_set_x, train_set_y = datasets[0]
  valid_set_x, valid_set_y = datasets[1]
  test_set_x, test_set_y = datasets[2]

  n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
  n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
  n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

  print '... building the model'

  index = T.lscalar()
  x = T.matrix('x')
  y = T.ivector('y')
  x_out = x
  for i in range(0, len(model) - 1):
    l = model[i]
    l.fp(x_out)
    x_out = l.output
  costL = model[-1]
  costL.fp(x_out, y)
  cost = costL.output

  test_model = theano.function(inputs=[index], outputs=costL.errors(y),
    givens={
      x: test_set_x[index * batch_size: (index + 1) * batch_size],
      y: test_set_y[index * batch_size: (index + 1) * batch_size]})

  validate_model = theano.function(inputs=[index], outputs=costL.errors(y),
    givens={
      x: valid_set_x[index * batch_size:(index + 1) * batch_size],
      y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

  updates = []

  for l in model:
    for p in l.params:
      g = T.grad(cost=cost, wrt=p)
      updates.append((p, p - lr * g))

  train_model = theano.function(inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
      x: train_set_x[index * batch_size:(index + 1) * batch_size],
      y: train_set_y[index * batch_size:(index + 1) * batch_size]})

  ###############
  # TRAIN MODEL #
  ###############
  print '... training the model'
  patience = 5000
  patience_increase = 2
  improvement_threshold = 0.995
  validation_frequency = min(n_train_batches, patience / 2)

  best_validation_loss = np.inf
  test_score = 0.
  start_time = time.clock()

  done_looping = False
  epoch = 0
  while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):

      train_model(minibatch_index)
      it = (epoch - 1) * n_train_batches + minibatch_index

      if (it + 1) % validation_frequency == 0:
        validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        this_validation_loss = np.mean(validation_losses)

        print('epoch %i, minibatch %i/%i, validation error %f %%' % \
          (epoch, minibatch_index + 1, n_train_batches,
          this_validation_loss * 100.))

        if this_validation_loss < best_validation_loss:
          if this_validation_loss < best_validation_loss *  \
             improvement_threshold:
            patience = max(patience, it * patience_increase)

          best_validation_loss = this_validation_loss

          test_losses = [test_model(i) for i in xrange(n_test_batches)]
          test_score = np.mean(test_losses)

          print(('   epoch %i, minibatch %i/%i, test error of best'
             ' model %f %%') %
            (epoch, minibatch_index + 1, n_train_batches,
             test_score * 100.))

      if patience <= it:
        done_looping = True
        break

  end_time = time.clock()
  print(('Optimization complete with best validation score of %f %%,'
       'with test performance %f %%') %
         (best_validation_loss * 100., test_score * 100.))
  print 'The code run for %d epochs, with %f epochs/sec' % (
    epoch, 1. * epoch / (end_time - start_time))
  print >> sys.stderr, ('The code for file ' +
              os.path.split(__file__)[1] +
              ' ran for %.1fs' % ((end_time - start_time)))

def main():
  model = [FCL(28 * 28, 10), BiasL(10), SoftmaxC()]
  sgd(model)

if __name__ == '__main__':
  main()

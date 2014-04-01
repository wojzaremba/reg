import time
import cPickle
import numpy as np

import theano
import theano.tensor as T
import os
import config

class Model(object):

  def __init__(self, name, lr=0.13, n_epochs=100, batch_size=100):
    print "_" * 100
    print "Creating model lr=%f, n_epochs=%d, batch_size=%d" % \
      (lr, n_epochs, batch_size)
    self.lr = lr
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.name = name
    self.source = None

  def set_source(self, source, params):
    params['batch_size'] = self.batch_size
    self.source = source(**params)
    return self.source

  def build_model(self):
    print '\n... building the model'
    idx = T.lscalar()
    bs = self.batch_size
    x = T.tensor4('x')
    y = T.ivector('y')
    s = self.source
    # XXX : Here should be sum over costs.
    cost_test = s.get_costs((x, y), False)[0]

    test_model = theano.function(inputs=[idx],
      outputs=[cost_test.errors(y), cost_test.output],
      givens={
        x: s.test_x[idx * bs: (idx + 1) * bs],
        y: s.test_y[idx * bs: (idx + 1) * bs]})

    cost_train = s.get_costs((x, y), True)[0]
    updates = s.get_updates(self.lr, cost_train.output)

    train_model = theano.function(inputs=[idx],
      outputs=[cost_train.errors(y), cost_train.output],
      updates=updates,
      givens={
        x: s.train_x[idx * bs:(idx + 1) * bs],
        y: s.train_y[idx * bs:(idx + 1) * bs]})
    return (train_model, test_model)

  def save(self, epoch):
    if not os.path.isdir(config.DUMP_DIR):
      os.makedirs(config.DUMP_DIR)
    dname = config.DUMP_DIR + self.name
    if not os.path.isdir(dname):
      os.makedirs(dname)
    fname = "%s/%s_%d" % (dname, self.name, epoch)
    f = open(fname, 'w')
    print "Saving weights %s" % (fname)
    cPickle.dump(self.source.dump(), f)

  def load(self):
    dname = config.DUMP_DIR + self.name
    if not os.path.isdir(dname):
      return 0
    epochs = [int(f[len(self.name) + 1:]) for f in os.listdir(dname)]
    if len(epochs) == 0:
      return 0
    epoch = max(epochs)
    fname = "%s/%s_%d" % (dname, self.name, epoch)
    res = ''
    while res != "y" and res != "Y":
      res = raw_input("Resume %s (y), or start from scratch (n) ? : " % (fname))
      if res == "n" or res == "N":
        print "Starting training from beginning"
        return 0
    print "Loading weights from %s" % (fname)
    f = open(fname, 'rb')
    self.source.load(cPickle.load(f))
    return epoch + 1

  def train(self):
    train_model, test_model = self.build_model()
    start_epoch = self.load()
    print '... training the model'
    test_freq = self.source.n_train_batches / 5
    save_freq = 5
    for epoch in range(start_epoch, self.n_epochs):
      start_time = time.clock()
      train_res = []
      for minibatch_index in xrange(self.source.n_train_batches):
        train_res.append(train_model(minibatch_index))
        it = (epoch - 1) * self.source.n_train_batches + minibatch_index
        if it % test_freq == 0 and it > 0:
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
      if epoch % save_freq == 0 and epoch > 0:
        self.save(epoch)
    self.save(self.n_epochs - 1)
    print "Training finished !"

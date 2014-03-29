import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
from os.path import expanduser

floatX = theano.config.floatX

def load_data(dataset):
  print '... loading data'
  home = expanduser("~")
  f = gzip.open(home + '/data/' + dataset + '/data.pkl.gz', 'rb')
  train_set, test_set = cPickle.load(f)
  f.close()

  def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

  test_set_x, test_set_y = shared_dataset(test_set)
  train_set_x, train_set_y = shared_dataset(train_set)

  rval = [(train_set_x, train_set_y),
          (test_set_x, test_set_y)]
  return rval

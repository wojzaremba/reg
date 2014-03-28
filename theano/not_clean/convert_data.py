import cPickle
import gzip
import os
import sys
import time
import numpy as np
from os.path import expanduser
import matplotlib.pyplot as plt
from scipy import misc

def main():
  print '... loading data'
  home = expanduser("~")
  f = gzip.open(home + '/data/mnist/data_old.pkl.gz', 'rb')
  train_set, valid_set, test_set = cPickle.load(f)
  f.close()  

  print "train size: ",  train_set[0].shape
  print "val size: ",  valid_set[0].shape

  x = np.concatenate((train_set[0], valid_set[0]), axis=0) 
  y = np.concatenate((train_set[1], valid_set[1]), axis=0) 
  print "train size: ",  x.shape
  x = np.reshape(x, (60000, 1, 28, 28))
  train_set = (x, y)
  x_test = np.reshape(test_set[0], (10000, 1, 28, 28))
  test_set = (x_test, test_set[1])
  print "test size: ",  test_set[0].shape
  l = x[1, 0, :, :]
  #plt.imshow(l)
  #plt.show()
  p = home + '/data/mnist/data.pkl'
  fid = open(p, 'wb')
  cPickle.dump((train_set, test_set), fid)
  fid.close()

if __name__ == '__main__':
  main()

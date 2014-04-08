import cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt

def unpickle(name):
  fname = '/Users/Denton/Downloads/cifar-10-batches-py/' + name
  fo = open(fname, 'rb')  
  dict = cPickle.load(fo)
  fo.close()
  return dict
if __name__ == '__main__':
  all_data = np.zeros((40000, 3, 32, 32))
  all_labels = np.zeros((40000))
  for b in range(0, 4):
    dict1 = unpickle('data_batch_' + str(b + 1))
    data = dict1['data']
    data = data.reshape((10000, 3, 32, 32)) 
    labels = dict1['labels']
    all_data[b * 10000 : (b + 1) * 10000, :, :, :] = data
    all_labels[b * 10000 : (b + 1) * 10000] = labels

  train_set = (all_data, all_labels)

  dict1 = unpickle('data_batch_5')
  data = dict1['data']
  data = data.reshape((10000, 3, 32, 32)) 
  labels = dict1['labels']
  test_set = (data, labels) 
 
  f = open('/Users/Denton/data/cifar10/data.pkl', 'wb')
  cPickle.dump( (train_set, test_set), f)

  dict1 = unpickle('test_batch')
  data = dict1['data']
  labels = dict1['labels']
  data = data.reshape((10000, 3, 32, 32)) 
  test_set = (data, labels) 
  f = open('/Users/Denton/data/cifar10/data_test.pkl', 'wb')
  cPickle.dump(test_set, f)

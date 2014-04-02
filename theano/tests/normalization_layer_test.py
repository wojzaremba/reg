import unittest
import numpy.random as r
import numpy as np
import theano
from layers.layer import EmptyL, LRCrossmapL

class NormalizationTests(unittest.TestCase):

  def testLRCrossmap(self):
    size = 5
    scale = 1.
    power = 0.5
    N = 5
    shape = [10, N, 5, 5]
    layer1 = EmptyL(in_shape=shape)
    layer1.attach(LRCrossmapL, {'size': size, 'scale': scale, 'power': power})
    x = r.rand(shape[0], shape[1], shape[2], shape[3])
    x[0, 0:3, :, :] = 2

    true = np.ones(shape)
    for f in range(0, N):
      norm_const = (1 + (scale / size)
		* (x[:, max(0, f - size/2) : min(N, 1 + f + size/2),
			 :, :]**2).sum(axis=1))**power
      true[:, f, :, :] = x[:, f, :, :] / norm_const
    x = theano.shared(x)
    y = 1 #not used
    layer1.succ[0].fp(x, y)
    self.assertTrue((true == layer1.succ[0].output.eval()).all())

def main():
  unittest.main()

if __name__ == '__main__':
  main()

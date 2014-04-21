import numpy as np
import numpy.linalg as linalg

def fc_svd(W, rank):
  U, S, V = linalg.svd(W)
  SS = np.zeros((U.shape[0], V.shape[0]))
  np.fill_diagonal(SS, S)
  Wapprox = np.dot(U[:, 0:rank], np.dot(SS[0:rank], V[:, 0:rank]))
  return Wapprox

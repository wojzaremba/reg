import theano
import numpy as np
import theano.tensor as T

def main():
  x=T.scalar()
  y=T.scalar()
  z=x+y*y
  f=theano.function([x,y], z)
  #import pdb
  #pdb.set_trace()
  fy=T.grad(z,y)
  fy2=theano.function([x,y],fy)
  print fy2(4,3)


if __name__ == '__main__':
  main()


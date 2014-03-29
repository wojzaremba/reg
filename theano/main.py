from layers.bundle import SoftmaxBC
from layers.layer import ReluL, ConvL, MaxpoolL
from train import sgd

# XXX: Test conv (separate test).
# XXX: write bundles.

def main():
  filter_shape = (96, 1, 5, 5)
  image_shape = (600, 1, 28, 28)
  subsample = (4, 4)
  model = [ConvL(filter_shape, image_shape, subsample, border_mode='valid'),
           ReluL(),
		   MaxpoolL( (2, 2)), 
		   SoftmaxBC(96 * 3 * 3, 10)]
  sgd(model)


if __name__ == '__main__':
  main()

from layers.bundle import SoftmaxBC
from layers.layer import ConvL
from train import sgd

# XXX: Test conv (separate test).
# XXX: write bundles.

def main():
  filter_shape = (96, 1, 5, 5)
  image_shape = (600, 1, 28, 28)
  subsample = (1, 1)
  model = [ConvL(filter_shape, image_shape, subsample),
           SoftmaxBC(96 * 28 * 28, 10)]
  sgd(model)


if __name__ == '__main__':
  main()

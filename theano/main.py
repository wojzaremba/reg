from layers.bundle import SoftmaxBC, ConvB
from layers.layer import MaxpoolL, Source
from model import Model

# XXX: Test conv (separate test).

def conv_mnist():
  model = Model()
  model.set_source(Source, {'dataset': 'mnist'})
  model.append(ConvB, {'filter_shape':(96, 1, 5, 5),
                       'subsample':(4, 4),
                       'border_mode':'valid'})
  model.append(MaxpoolL, {'pool_shape':(2, 2)})
  model.append(SoftmaxBC, {})
  return model

def fully_connected_mnist():
  model = Model()
  model.set_source(Source, {'dataset': 'mnist'})
  model.append(SoftmaxBC, {})
  return model

def main():
  #model = conv_mnist()
  model = fully_connected_mnist()
  model.train()

if __name__ == '__main__':
  main()

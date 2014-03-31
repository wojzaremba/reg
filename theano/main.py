from layers.bundle import SoftmaxBC, ConvB, FCB
from layers.layer import MaxpoolL, DropoutL, LRCrossmapL, Source
from model import Model
import sys

def conv_mnist():
  model = Model()
  model.set_source(Source, {'dataset': 'mnist'})
  model.append(ConvB, {'filter_shape': (96, 1, 5, 5),
                       'subsample': (4, 4),
                       'border_mode': 'valid'})
  model.append(LRCrossmapL, {'size': 5,
							 'scale':0.001,
							 'power':0.75})
  model.append(MaxpoolL, {'pool_shape': (2, 2)})
  model.append(SoftmaxBC, {'out_len': 10})
  return model

def fc_mnist():
  model = Model()
  model.set_source(Source, {'dataset': 'mnist'})
  model.append(FCB, {'out_len': 200})
  model.append(SoftmaxBC, {'out_len': 10})
  return model

def fc_do_mnist():
  model = Model()
  model.set_source(Source, {'dataset': 'mnist'})
  model.append(FCB, {'out_len': 200})
  model.append(DropoutL, {})
  model.append(SoftmaxBC, {'out_len': 10})
  return model

def main():
  fun = 'conv_mnist'
  if len(sys.argv) > 1:
    fun = sys.argv[1]
  model = eval(fun + '()')
  model.train()

if __name__ == '__main__':
  main()

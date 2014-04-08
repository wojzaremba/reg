from layers.bundle import SoftmaxBC, ConvB, FCB
from layers.layer import MaxpoolL, DropoutL, LRSpatialL
from layers.layer import  Source, FCL, BiasL, LRCrossmapL
from layers.cost import WL2C
from model import Model
import sys

def conv_cifar(model):
  model.set_source(Source, {'dataset': 'cifar10'}) \
  .attach(ConvB, {'filter_shape': (64, 3, 5, 5),
				  'subsample': (2, 2),
				  'border_mode': 'valid'})\
  .attach(LRCrossmapL, {'size': 5})\
  .attach(FCB, {'out_len': 64})\
  .attach(SoftmaxBC, {'out_len': 10})
  return model

def conv_mnist(model):
  model.set_source(Source, {'dataset': 'mnist'})\
  .attach(ConvB, {'filter_shape': (96, 1, 5, 5),
                  'subsample': (4, 4),
                  'border_mode': 'valid'})\
  .attach(LRSpatialL, {'size': 5,
							 'scale':0.001,
							 'power':0.75})\
  .attach(MaxpoolL, {'pool_shape': (2, 2)})\
  .attach(SoftmaxBC, {'out_len': 10})
  return model

def fc_reg_mnist(model):
  fc = model.set_source(Source, {'dataset': 'mnist'})\
  .attach(FCL, {'out_len': 200})
  fc.attach(WL2C, {'alpha': 0.01})
  fc.attach(BiasL, {})\
  .attach(SoftmaxBC, {'out_len': 10})
  return model

def fc_mnist(model):
  model.set_source(Source, {'dataset': 'mnist'})\
  .attach(FCB, {'out_len': 200})\
  .attach(SoftmaxBC, {'out_len': 10})
  return model

def fc_do_mnist(model):
  model.set_source(Source, {'dataset': 'mnist'})\
  .attach(FCB, {'out_len': 200})\
  .attach(DropoutL, {})\
  .attach(SoftmaxBC, {'out_len': 10})
  return model

def main():
  fun = 'conv_mnist'
  if len(sys.argv) > 1:
    fun = sys.argv[1]
  model = Model(name=fun)
  model = eval(fun + '(model)')
  model.train()

if __name__ == '__main__':
  main()

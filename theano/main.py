from layers.bundle import SoftmaxBC, ConvB, FCB
from layers.layer import MaxpoolL, DropoutL, DropinL, LRSpatialL
from layers.layer import  Source, FCL, BiasL, LRCrossmapL
from layers.cost import WL2C
from model import Model
import sys



def fc_cifar(model):
  model.set_source(Source, {'dataset': 'cifar10'})\
  .attach(ConvB, {'filter_shape': (64, 3, 5, 5),
				  'subsample': (1, 1),
				  'border_mode': 'full',
          'on_gpu': model.on_gpu})\
  .attach(SoftmaxBC, {'out_len': 10})
  return model

def conv_cifar(model):
  model.set_source(Source, {'dataset': 'cifar10'}) \
  .attach(ConvB, {'filter_shape': (64, 3, 5, 5),
				  'subsample': (1, 1),
				  'border_mode': 'full',
          'on_gpu': model.on_gpu})\
  .attach(MaxpoolL, {'pool_shape': (3, 3),
                     'stride': (2, 2),
                     'on_gpu': model.on_gpu})\
  .attach(LRCrossmapL, {'size': 9})\
  .attach(FCB, {'out_len': 64})\
  .attach(SoftmaxBC, {'out_len': 10})
  return model

def conv_mnist(model):
  model.set_source(Source, {'dataset': 'mnist'})\
  .attach(ConvB, {'filter_shape': (96, 1, 5, 5),
                  'subsample': (4, 4),
                  'border_mode': 'full',
                  'on_gpu': model.on_gpu})\
  .attach(LRSpatialL, {'size': 5,
							 'scale':0.001,
							 'power':0.75})\
  .attach(MaxpoolL, {'pool_shape': (2, 2),
                     'stride': (1, 1),
                     'on_gpu': model.on_gpu})\
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
  .attach(FCB, {'out_len': 800})\
  .attach(FCB, {'out_len': 800})\
  .attach(SoftmaxBC, {'out_len': 10})
  return model

def fc_do_mnist(model):
  model.set_source(Source, {'dataset': 'mnist'})\
  .attach(FCB, {'out_len': 800})\
  .attach(DropoutL, {})\
  .attach(SoftmaxBC, {'out_len': 10})
  return model

def fc_di_mnist(model):
  model.set_source(Source, {'dataset': 'mnist'})\
  .attach(FCB, {'out_len': 800})\
  .attach(DropinL, {'p': 1})\
  .attach(SoftmaxBC, {'out_len': 10})
  return model

def main():
  fun = 'fc_cifar'
  if len(sys.argv) > 1:
    fun = sys.argv[1]
  model = Model(name=fun, lr=0.1, n_epochs=500, on_gpu=True)
  model = eval(fun + '(model)')
  model.train()

if __name__ == '__main__':
  main()

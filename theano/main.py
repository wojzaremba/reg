from layers.bundle import SoftmaxBC
from train import sgd

# XXX: Test conv (separate test).
# XXX: write bundles.

def main():
  model = [SoftmaxBC(28 * 28, 10)]
  sgd(model)

if __name__ == '__main__':
  main()

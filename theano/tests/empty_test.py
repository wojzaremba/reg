import unittest

class EmptyTests(unittest.TestCase):

  def testFirst(self):
    self.assertEquals(1, 1)

def main():
  unittest.main()

if __name__ == '__main__':
  main()

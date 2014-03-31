Reg
===

All the weights, and many training sets are available under http://www.cs.nyu.edu/~zaremba/data .

Setting up hooks
================
    cd .git/hooks
    .git/hooks$ ln -s ../../theano/PRESUBMIT.py pre-commit
    chmod a+x ./pre-commit

**TODO**
========
- Implement normalizations.
- Import CIFAR-10.
- Create multiple costs.
- More parameters for starting execution.
- Save mnist in float8.
- Mean subtraction layer, division layer, subtraction layer. Combine them to input layer.
- Add data_augmentation layers (like random view).
- Write script to visulalize weights from the first layer.
- Common initialization mechanism.
- Import Pierre's model.
- Write loss (replace cost), write L2 loss, and cross-entropy.
- add some tests !!

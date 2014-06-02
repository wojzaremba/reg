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
Wojciech:
- Write loss (replace cost), write L2 loss, and cross-entropy. Place a L2 penalty on weights.

Emily:
- Implement normalizations.
- Import CIFAR-10.
- Get baseline on cifar-10
- get a simple regularization through approx (schedule it with saving, and testing)

Joan:
- Get nuclear norm to work

Potential:
- More parameters for starting execution (there is special lib in python to parse it arg-something).
- Save mnist in float8.
- Mean subtraction layer, division layer, subtraction layer. Combine them to input layer.
- Add data_augmentation layers (like random view).
- Write script to visulalize weights from the first layer.
- Common initialization mechanism.
- Import Pierre's model.
- add some tests (also end2end test) !! Move end2end from other project.
- model should save its model configuration as it runs

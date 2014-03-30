Reg
===

All the weights, and many training sets are available under http://www.cs.nyu.edu/~zaremba/data .

Setting up hooks
================
cd .git/hooks
.git/hooks$ ln -s ../../theano/PRESUBMIT.py pre-commit
chmod a+x ./pre-commit


TODO: 
1. make clean in PRESUBMIT
2. Figure out how to make cache working.
3. Implement dropout (tell Karol).
4. Implement normalizations.
5. Import CIFAR-10.
6. Create multiple costs.
7. Fix softmax to know about number of labels.
8. Get some faster impl. on CPU.


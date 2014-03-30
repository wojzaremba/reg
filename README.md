Reg
===

All the weights, and many training sets are available under http://www.cs.nyu.edu/~zaremba/data .

Setting up hooks
================
cd .git/hooks
.git/hooks$ ln -s ../../theano/PRESUBMIT.py pre-commit
chmod a+x ./pre-commit


TODO: 
4. Implement normalizations.

5. Import CIFAR-10.

6. Create multiple costs.

8. Get some faster impl. of conv on CPU.

9. Save model automatically.

10. Mean subtraction layer, division layer, subtraction layer. Combine them to input layer.

11. Add data_augmentation layers (like random view).

12. Write script to visulalize weights from the first layer.

13. Store mnist as int in 0-255 ? 

14. Common initialization mechanism.

15. Fix dropout.

16. Import Pierre's model.

17. Write L2 loss.


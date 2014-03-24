% 1. Fix GPU compilation.
global plan
addpath(genpath('.'));
json = ParseJSON('plans/cifar_conv.txt');
Plan(json, [], 0);
Run();
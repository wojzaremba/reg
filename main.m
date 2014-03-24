global plan
addpath(genpath('.'));
json = ParseJSON('plans/cifar_conv.txt');
Plan(json);
Run();
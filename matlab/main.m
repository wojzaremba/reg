

global plan
addpath(genpath('.'));
json = ParseJSON('plans/mnist_conv.txt');
Plan(json);

plan.regu.regepoch=0;
plan.regu.betareg=4;

Run();

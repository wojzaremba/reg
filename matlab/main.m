global plan
addpath(genpath('.'));
json = ParseJSON('plans/cifar_conv.txt');
Plan(json);

plan.regu.regepoch=1;
plan.regu.betareg=1;

Run();

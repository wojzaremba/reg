C_(CleanGPU);
clear all;
C_(SetDevice, 1);
global plan;
randn('seed', 1);
load_imagenet_model('matthew', 128);

% Forward prop
nbatches = 390;
error = 0;
plan.input.step = 1;
for i = 1:nbatches
    plan.input.GetImage(0);
    ForwardPass(); 
    e = plan.classifier.GetScore(5);
    error = error + e;
    fprintf('(%d) %d / %d = %f     (%d / %d = %f)\n', i, e,  plan.input.batch_size, e /  plan.input.batch_size, error, i * plan.input.batch_size, error / (i * plan.input.batch_size));
end

% 8866 / 49920 = 0.177604
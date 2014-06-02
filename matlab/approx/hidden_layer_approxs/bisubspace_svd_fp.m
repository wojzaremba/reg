C_(CleanGPU);
clear all;
C_(SetDevice, 1);
global plan;
randn('seed', 1);
load_imagenet_model;

W = plan.layer{5}.cpu.vars.W;

fprintf('||W|| = %f \n', norm(W(:)));

iclust = 2;
oclust = 2;
oratio = 0.4; % (0.6 --> 76), (0.5 --> 64)
iratio = 0.3;  % (0.6 --> 78), (0.5 --> 24), (0.4 --> 19)
odegree = floor(size(W, 1) * oratio / oclust);
idegree = floor(size(W, 4) * iratio / iclust);

code = sprintf('in%d_out%d', idegree, odegree);

in_s = 55;
out_s = 51;
[Wapprox, C, Z, F, idx_input, idx_output] = bispace_svd(W, iclust, iratio, oclust, oratio, 0, in_s, out_s);
if plan.layer{5}.on_gpu
    C_(CopyToGPU, plan.layer{5}.gpu.vars.W, single(Wapprox));
else
    plan.layer{5}.cpu.vars.W = single(Wapprox);
end

% Get error
error = 0;
plan.input.step = 1;
nbatches = 390;
for i = 1:nbatches
    plan.input.GetImage(0);
    ForwardPass(); 
    e = plan.classifier.GetScore(5);
    error = error + e;
    fprintf('(%d) %d / %d = %f     (%d / %d = %f)\n', i, e,  plan.input.batch_size, e /  plan.input.batch_size, error, i * plan.input.batch_size, error / (i * plan.input.batch_size));
end

load 'generated_mats/layer2_bisubspace_svd_2_2_error.mat'
idx = find(ismember(rank_codes, code));
if isempty(idx)
    rank_codes{end+1} = code;
    errors(end+1) = error / (i * plan.input.batch_size);
else
    errors(idx) = error / (i * plan.input.batch_size);
end

save('generated_mats/layer2_bisubspace_svd_2_2_error.mat', 'errors', 'rank_codes');


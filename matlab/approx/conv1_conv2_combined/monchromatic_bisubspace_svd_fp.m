C_(CleanGPU);
clear all;
C_(SetDevice, 0);
global plan;
randn('seed', 1);
load_imagenet_model;

num_colors = 6;

W = plan.layer{5}.cpu.vars.W;

iclust = 2;
oclust = 2;
oratio = 0.4; % (0.6 --> 76), (0.5 --> 64)
iratio = 0.4;  % (0.6 --> 78), (0.5 --> 24), (0.4 --> 19)
odegree = floor(size(W, 1) * oratio / oclust);
idegree = floor(size(W, 4) * iratio / iclust);
code = sprintf('in%d_out%d', idegree, odegree);

if 1
    fname = sprintf('/misc/vlgscratch3/FergusGroup/denton/monochromatic%d_finetuneall', num_colors);
    load_weights(fname, 1);
    fprintf('\nLoading weights from %s\n\n', fname);
    fname = sprintf('monochromatic%d_layer2_svd_2_2_%d_%d_finetuneall', num_colors, idegree, odegree);
    load_weights(fname, 2);
    fprintf('\nLoading weights from %s\n\n', fname);
end

% Compute first layer approximation
W = plan.layer{2}.cpu.vars.W;
fprintf('Conv 1: ||W|| = %f \n', norm(W(:)));
args.num_colors = num_colors;
args.even = 1;
[Wapprox, Wmono, colors, perm] = monochromatic_approx(double(W), args);
L2_err = norm(W(:) - Wapprox(:)) / norm(W(:));
fprintf('Conv 1: ||W - Wapprox|| / ||W|| = %f\n', L2_err);

% Replace first convolutional layer weights with approximated weights.
if plan.layer{2}.on_gpu
    C_(CopyToGPU, plan.layer{2}.gpu.vars.W, single(Wapprox));
else
    plan.layer{2}.cpu.vars.W = single(Wapprox);
end

% Compute first layer approximation
W = plan.layer{5}.cpu.vars.W;
fprintf('Conv 2: ||W|| = %f \n', norm(W(:)));

in_s = 55;
out_s = 51;

[Wapprox, C, Z, F, idx_input, idx_output] = bispace_svd(W, iclust, iratio, oclust, oratio, 0, in_s, out_s);

L2_err = norm(W(:) - Wapprox(:)) / norm(W(:));
fprintf('Conv 2: ||W - Wapprox|| / ||W|| = %f\n', L2_err);

% Replace second convolutional layer weights with approximated weights
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
% % 
% load generated_mats/monochromatic6_layer2_svd_finetuned_error.mat
% idx = find(ismember(rank_codes, code));
% if isempty(idx)
%     rank_codes{end+1} = code;
%     errors(end+1) = error / (i * plan.input.batch_size);
% else
%     errors(idx) = error / (i * plan.input.batch_size);
% end
% save('generated_mats/monochromatic6_layer2_svd_finetuned_error.mat', 'errors', 'rank_codes');


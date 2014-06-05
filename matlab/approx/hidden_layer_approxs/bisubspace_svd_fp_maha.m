C_(CleanGPU);
clear all;
C_(SetDevice, 0);
global plan;
randn('seed', 1);
load_imagenet_model;

W = plan.layer{5}.cpu.vars.W;

iclust = 2;
oclust = 2;
oratio = 0.5; % (0.6 --> 76), (0.5 --> 64)
iratio = 0.4;  % (0.6 --> 78), (0.5 --> 24), (0.4 --> 19)
odegree = floor(size(W, 1) * oratio / oclust);
idegree = floor(size(W, 4) * iratio / iclust);
code = sprintf('in%d_out%d', idegree, odegree);
fname = sprintf('layer2_svd_%d_%d_%d_%d_finetuneall', iclust, oclust, idegree, odegree);

if 0
  load_weights(fname, 2);
  fprintf('\nLoading weights from %s\n\n', fname);
end

W = plan.layer{5}.cpu.vars.W;

metric = load('/misc/vlgscratch3/FergusGroup/denton/mahalanobis_distance_approx.mat');
epsilon = 0.5 ;
sigmas = epsilon + metric.conv2_maha;
WW = W .* reshape(sigmas, size(W));

fprintf('||W|| = %f \n', norm(W(:)));
fprintf('||WW|| = %f \n', norm(WW(:)));



in_s = 55;
out_s = 51;
[Wapprox, C, Z, F, idx_input, idx_output] = bispace_svd(WW, iclust, iratio, oclust, oratio, 0, in_s, out_s);

L2_err = norm(WW(:) - Wapprox(:)) / norm(WW(:));
fprintf('||WW - Wapprox|| / ||WW|| = %f\n', L2_err);

Wapprox = Wapprox ./ reshape(sigmas, size(WW));
L2_err = norm(W(:) - Wapprox(:)) / norm(W(:));
fprintf('||W - Wapprox|| / ||W|| = %f\n', L2_err);

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
% 
% load 'generated_mats/layer2_bisubspace_svd_2_2_error_maha.mat'
% idx = find(ismember(rank_codes, code));
% if isempty(idx)
%     rank_codes{end+1} = code;
%     errors(end+1) = error / (i * plan.input.batch_size);
% else
%     errors(idx) = error / (i * plan.input.batch_size);
% end
% 
% save('generated_mats/layer2_bisubspace_svd_2_2_error_maha.mat', 'errors', 'rank_codes');
% 

C_(CleanGPU);
clear all;
C_(SetDevice, 2);
global plan;
randn('seed', 1);
load_imagenet_model('matthew', 128);

num_colors = 16;
if 0
  fname = sprintf('monochromatic%d_finetuneall_alternate', num_colors);
  load_weights(fname, 1);
  fprintf('\nLoading weights from %s\n\n', fname);
end

W = plan.layer{2}.cpu.vars.W;
metric = load('/misc/vlgscratch3/FergusGroup/denton/mahalanobis_distance_approx.mat');
epsilon = 0.5 ;
sigmas = epsilon + metric.conv1_maha;
WW = W .* sigmas;

% Compute approximation
fprintf('||W|| = %f \n', norm(W(:)));
fprintf('||WW|| = %f \n', norm(WW(:)));

args.num_colors = num_colors;
args.even = 1;
[Wapprox, Wmono, colors, perm] = monochromatic_approx(double(WW), args);
L2_err = norm(WW(:) - Wapprox(:)) / norm(WW(:));
fprintf('||WW - Wapprox|| / ||WW|| = %f\n', L2_err);

Wapprox = Wapprox ./ sigmas;
L2_err = norm(W(:) - Wapprox(:)) / norm(W(:));
fprintf('||W - Wapprox|| / ||W|| = %f\n', L2_err);

% Replace first convolutional layer weights with approximated weights.
if plan.layer{2}.on_gpu
    C_(CopyToGPU, plan.layer{2}.gpu.vars.W, single(Wapprox));
else
    plan.layer{2}.cpu.vars.W = single(Wapprox);
end


% Forward prop
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
% load generated_mats/monochromatic_finetuned_error.mat
% idx = find(num_colors_list == num_colors);
% if isempty(idx)
%     num_colors_list(end+1) = num_colors;
%     errors(end+1) =  error / (i * plan.input.batch_size);
% else
%     errors(idx) =  error / (i * plan.input.batch_size);
% end
% save('generated_mats/monochromatic_finetuned_error.mat', 'errors', 'num_colors_list');

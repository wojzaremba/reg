C_(CleanGPU);
clear all;
C_(SetDevice, 0);
global plan;
randn('seed', 1);


json = ParseJSON('plans/imagenet_maha.txt');
json{1}.batch_size = 128;
add_dropout = 0;
if is_cluster()	
    Plan(json, '/misc/vlgscratch3/FergusGroup/denton/imagenet_data/imagenet_matthew', 1);
else
    Plan(json, sprintf('~/imagenet_data/imagenet_%s', type), 0);
end

plan.input.step = 1;
plan.momentum = 0;
plan.lr = 0;

D = 2;

conv1_maha = zeros(size(plan.layer{2}.cpu.vars.W));
conv2_maha = zeros(size(plan.layer{5}.cpu.vars.W));
conv1_maha_grads = [];
conv2_maha_grads = [];

nbatches = 500;
normalizer = 1;
for b = 1 : nbatches
    plan.input.GetImage(1);
    for d = 1 : D
        plan.layer{end}.d = d;
        ForwardPass();
        BackwardPass();
        
        % Add to conv1_maha
        dW = reshape(C_(CopyFromGPU, plan.layer{2}.gpu.dvars.W), size(conv1_maha));
        conv1_maha = conv1_maha + dW.^2;
        conv1_maha_grads(:, end+1) = dW(:);
        
        % Add to conv2_maha
        dW = reshape(C_(CopyFromGPU, plan.layer{5}.gpu.dvars.W), size(conv2_maha));
        conv2_maha = conv2_maha + dW.^2;
        conv2_maha_grads(:, end+1) = dW(:);
        
        normalizer = normalizer + 1;
    end
    fprintf('Done batch %d\n', b);

end

conv1_maha = sqrt(conv1_maha) / normalizer;
conv2_maha = sqrt(conv2_maha) / normalizer;

mean1 = mean(conv1_maha_grads);
conv1_maha_grads_cen = bsxfun(@minus, conv1_maha_grads, mean1);
conv1_maha_cov = conv1_maha_grads_cen * conv1_maha_grads_cen';

mean2 = mean(conv2_maha_grads);
conv2_maha_grads_cen = bsxfun(@minus, conv2_maha_grads, mean1);
conv2_maha_cov = conv2_maha_grads_cen * conv2_maha_grads_cen';


save('/misc/vlgscratch3/FergusGroup/denton/mahalanobis_distance_approx.mat', 'conv1_maha', 'conv2_maha');
save('/misc/vlgscratch3/FergusGroup/denton/mahalanobis_distance_cov.mat', 'conv1_maha_cov', 'conv2_maha_cov');

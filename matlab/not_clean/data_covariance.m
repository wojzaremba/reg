C_(CleanGPU);
clear all;
C_(SetDevice, 1);
global plan;
randn('seed', 1);
load_imagenet_model('matthew_train', 128);
nimg = length(plan.input.Y);

layer_nr1 = 1;
F1 = 3;
X1 = 7;
M1 = 110;
stride1 = 2;
cov1 = zeros(F1 * X1 * X1);

layer_nr2 = 4;
F2 = 96;
X2 = 5;
M2 = 51;
stride2 = 1;
cov2 = zeros(F2 * X2 * X2);

bs = plan.input.batch_size;
ntrain_batches = 500;
train_err = [];
error = 0;
plan.input.step = 1;
plan.training = 0;
N1 = 0;
N2 = 0;
for b = 1 : 5000
    plan.input.GetImage(1);
    ForwardPass();
    
    % Conv 1 data covariance
    if plan.layer{layer_nr1}.on_gpu
        data = reshape(C_(CopyFromGPU, plan.layer{layer_nr1}.gpu.vars.out), size(plan.layer{layer_nr1}.cpu.vars.out));
    else
        data = plan.layer{layer_nr1}.cpu.vars.out;
    end
    data(:, 225, 225, :) = 0; % pad
    stacked = zeros(F1 * X1 * X1, M1 * M1 * bs);
    ii = 1;
    for x = 1:M1
        for y = 1:M1
            sx = (x - 1) * stride1 + 1;
            ex = sx + X1 - 1;
            sy = (y - 1) * stride1 + 1;
            ey = sy + X1 - 1;
            tmp = data(:, sx:ex, sy:ey, :);
            stacked(:, (ii-1) * 128 + 1: ii * 128) = tmp(:, :)';
            ii = ii + 1;
        end
    end
    N1 = N1 + bs * M1 * M1 * bs;
    mean_stacked = mean(stacked, 2);
    centered_stacked = bsxfun(@minus, stacked, mean_stacked);
    %cov = cov + centered_stacked * centered_stacked';  
    cov1 = cov1 + stacked * stacked';
    
    % Conv2 data covariance
    if plan.layer{layer_nr2}.on_gpu
        data = reshape(C_(CopyFromGPU, plan.layer{layer_nr2}.gpu.vars.out), size(plan.layer{layer_nr2}.cpu.vars.out));
    else
        data = plan.layer{layer_nr2}.cpu.vars.out;
    end
    
    stacked = zeros(F2 * X2 * X2, M2 * M2 * bs);
    ii = 1;
    for x = 1:M2
        for y = 1:M2
            sx = (x - 1) * stride2 + 1;
            ex = sx + X2 - 1;
            sy = (y - 1) * stride2 + 1;
            ey = sy + X2 - 1;
            tmp = data(:, sx:ex, sy:ey, :);
            stacked(:, (ii-1) * 128 + 1: ii * 128) = tmp(:, :)';
            ii = ii + 1;
        end
    end
    N2 = N2 + bs * M2 * M2 * bs;
    mean_stacked = mean(stacked, 2);
    centered_stacked = bsxfun(@minus, stacked, mean_stacked);
    %cov = cov + centered_stacked * centered_stacked';  
    cov2 = cov2 + stacked * stacked';
    fprintf('Batch %d\n', b);  
    
    if mod(b, 100) == 0
        conv2_cov = cov2 / N2;
        conv1_cov = cov1 / N1;
        fname = '/misc/vlgscratch3/FergusGroup/denton/conv1_data_cov.mat';
        save(fname, 'conv1_cov');
        fname = '/misc/vlgscratch3/FergusGroup/denton/conv2_data_cov.mat';
        save(fname, 'conv2_cov');
        fprintf('Saving to %s...\n', fname);
    end
end

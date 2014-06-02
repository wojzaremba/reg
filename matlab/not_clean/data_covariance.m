% C_(CleanGPU);
% clear all;
% C_(SetDevice, 3);
% global plan;
% randn('seed', 1);
% load_imagenet_model('matthew_train', 128);
% nimg = length(plan.input.Y);

layer_nr = 4;
F = 96;
X = 5;
M = 51;
stride = 1;
cov = zeros(F * X * X);

bs = plan.input.batch_size;
ntrain_batches = 500;
train_err = [];
error = 0;
plan.input.step = 1;
plan.training = 0;
N = 0;
for b = 1 : 500
    plan.input.GetImage(1);
    ForwardPass();
    
    if plan.layer{layer_nr}.on_gpu
        data = reshape(C_(CopyFromGPU, plan.layer{layer_nr}.gpu.vars.out), size(plan.layer{layer_nr}.cpu.vars.out));
    else
        data = plan.layer{layer_nr}.cpu.vars.out;
    end
    
    stacked = zeros(F * X * X, M * M * bs);
    ii = 1;
    for x = 1:M
        for y = 1:M
            sx = (x - 1) * stride + 1;
            ex = sx + X - 1;
            sy = (y - 1) * stride + 1;
            ey = sy + X - 1;
            tmp = data(:, sx:ex, sy:ey, :);
            stacked(:, (ii-1) * 128 + 1: ii * 128) = tmp(:, :)';
            ii = ii + 1;
        end
    end
    N = N + bs * M * M * bs;
    mean_stacked = mean(stacked, 2);
    centered_stacked = bsxfun(@minus, stacked, mean_stacked);
    cov = cov + centered_stacked * centered_stacked';  
    fprintf('Batch %d\n', b);  
    
    if mod(b, 100) == 0
        conv2_cov = cov / N;
        fname = '/misc/vlgscratch3/FergusGroup/denton/conv2_data_cov.mat';
        save(fname, 'conv2_cov');
        fprintf('Saving to %s...\n', fname);
    end
end

conv2_cov = cov / N;
save('/misc/vlgscratch3/FergusGroup/denton/conv2_data_cov.mat', 'conv2_cov');
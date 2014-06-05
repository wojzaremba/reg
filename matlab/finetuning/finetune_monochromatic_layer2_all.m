C_(CleanGPU);
clear all;
C_(SetDevice, 0);
global plan;
randn('seed', 1);
load_imagenet_model('matthew_train', 128);
plan.training = 1;

iclust = 48;
oclust = 2;
rank = 6;
num_colors = 6;


% Load weights for finetuned monochromatic layer
%fname = sprintf('monochromatic%d_finetuneall', num_colors);
fname = '/misc/vlgscratch3/FergusGroup/denton/monochromatic6_finetuneall.mat';
load_weights_training(fname, 1);
fprintf('\nLoading weights from %s\n\n', fname);

fname = sprintf('monochromatic%d_layer2_bisubspace_%d_%d_%d_finetuneall', num_colors, iclust, oclust, rank);

restart = 1;
if ~restart
  load_weights_training(fname, 2);
  fprintf('\nLoading weights from %s\n\n', fname);
end

% Replace first conv layer with approximated weights
W = plan.layer{2}.cpu.vars.W;
fprintf('Conv 1: ||W|| = %f \n', norm(W(:)));
args.num_colors = num_colors;
args.even = 1;
[Wapprox, Wmono, colors, perm] = monochromatic_approx(double(W), args);
L2_err = norm(W(:) - Wapprox(:)) / norm(W(:));
fprintf('Conv 1: ||W - Wapprox|| / ||W|| = %f\n', L2_err);
if plan.layer{2}.on_gpu
    C_(CopyToGPU, plan.layer{2}.gpu.vars.W, single(Wapprox));
else
    plan.layer{2}.cpu.vars.W = single(Wapprox);
end

% Replace second conv layer with approximated weights
approx_layer = 5;
W = plan.layer{approx_layer}.cpu.vars.W;
fprintf('Conv 2: ||W|| = %f \n', norm(W(:)));
args.iclust = iclust;
args.oclust = oclust;
args.cluster_type = 'kmeans';
args.k = rank;
args.in_s = 55;
args.out_s = 51;

[Wapprox, F, C, XY, perm_in, perm_out, num_weights] = bisubspace_lowrank_approx_nosep(double(W), args);

L2_err = norm(W(:) - Wapprox(:)) / norm(W(:));
fprintf('Conv 2: ||W - Wapprox|| / ||W|| = %f\n', L2_err);
C_(CopyToGPU, plan.layer{approx_layer}.gpu.vars.W, single(Wapprox));
plan.layer{approx_layer}.cpu.vars.W = single(Wapprox);

% Finetuning parameters
min_layer = 6;
plan.momentum = 0.9;
plan.lr = 0.001;

nimg = length(plan.input.Y);
bs = plan.input.batch_size;
n_batches = floor(nimg / bs);
nval_batches = 24;
nepoch = 1;
ntrain_batches = n_batches;
eval_freq = 50;
train_err = [];
val_err = [];
step = [];
for epoch = 1 : nepoch
    plan.input.step = nval_batches + 1;
    plan.repeat = epoch;
    error = 0;
    for b = 1 : ntrain_batches
        
        if b > 1000
            plan.lr = 0.0001;
        end
        if b > 1500
            plan.lr = 0.00001;
        end
        
        plan.input.GetImage(1);
        ForwardPass();
        e = plan.classifier.GetScore(5);
        error = error + e;
        eval_train_freq = 25;
        if mod(b, eval_train_freq) == 0
            fprintf('(%d - %d) Train error: %d / %d = %f\n', b, plan.input.step - 1, error, eval_train_freq * bs, error / (eval_train_freq * bs));
            train_err(end+1) = error / (eval_train_freq * bs);
            error = 0;
        end
       
        PartialBP(min_layer);
        
        if mod(b, eval_freq) == 0
            old_step = plan.input.step;
            plan.training = 0;
            plan.input.step = 1;
            val_error = 0;
            for bb = 1 : nval_batches
                plan.input.GetImage(0);
                ForwardPass();
                val_error = val_error + plan.classifier.GetScore(5);
            end   
            fprintf('Val error: %d / %d = %f %%\n', val_error, bb * bs, val_error / (bb * bs));
            val_err(end+1) = val_error / (bb * bs);
            plan.input.step = old_step;
            plan.training = 1;
        end

        if mod(b, 250) == 0
           fprintf('\nSaving weights to file %s...', fname);
           save_weights(fname, train_err, val_err);
           fprintf('Done\n\n');
        end
    end
end

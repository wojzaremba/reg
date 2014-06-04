C_(CleanGPU);
clear all;
C_(SetDevice, 2);
global plan;
randn('seed', 1);
load_imagenet_model('matthew_train', 128);
plan.training = 1;
num_colors = 4;
fname = sprintf('monochromatic%d_finetuneall', num_colors);

restart = 0;
if ~restart
  load_weights_training(fname, 1);
  fprintf('\nLoading weights from %s\n\n', fname);
end

% Replace first conv layer with approximated weights
W = plan.layer{2}.cpu.vars.W;
args.num_colors = num_colors;
args.even = 1;
[Wapprox, Wmono, colors, perm] = monochromatic_approx(double(W), args);
L2_err = norm(W(:) - Wapprox(:)) / norm(W(:));
fprintf(' ||W - Wapprox|| / ||W|| = %f\n', L2_err);
C_(CopyToGPU, plan.layer{2}.gpu.vars.W, single(Wapprox));
plan.layer{2}.cpu.vars.W = single(Wapprox);

% Finetuning parameters
min_layer = 3;
plan.momentum = 0.9;
plan.lr = 0.00001;

nimg = length(plan.input.Y);
bs = plan.input.batch_size;
n_batches = floor(nimg / bs);
nval_batches = 24;
nepoch = 1;
ntrain_batches = n_batches;
eval_freq = 50;
train_err = [];
val_err = [];

for epoch = 1 : nepoch
    plan.input.step = nval_batches + 1;
    plan.repeat = epoch;
    error = 0;
    for b = 1 : ntrain_batches
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
            plan.input.step = 1;
            val_error = 0;
            plan.training = 0;
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


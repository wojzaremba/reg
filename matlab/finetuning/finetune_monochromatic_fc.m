C_(CleanGPU);
clear all;
C_(SetDevice, 1);
global plan;
randn('seed', 1);
load_imagenet_model('matthew_train', 128);
plan.training = 1;

% Replace first conv layer with approximated weights
W = plan.layer{2}.cpu.vars.W;
args.num_colors = 12;
args.even = 1;
[Wapprox, Wmono, colors, perm] = monochromatic_approx(double(W), args);
L2_err = norm(W(:) - Wapprox(:)) / norm(W(:));
fprintf(' ||W - Wapprox|| / ||W|| = %f\n', L2_err);
C_(CopyToGPU, plan.layer{2}.gpu.vars.W, single(Wapprox));
plan.layer{2}.cpu.vars.W = single(Wapprox);

% Finetuning parameters
min_layer = 13;
plan.momentum = 0.9;
plan.lr = 0.001;

nimg = length(plan.input.Y);
bs = plan.input.batch_size;
n_batches = floor(nimg / bs);
nval_batches = 16;
nepoch = 3;
ntrain_batches = n_batches;
eval_freq = 5;
train_err = [];
val_err = [];
step = [];
for epoch = 1 : nepoch
    plan.input.step = nval_batches + 1;
    plan.repeat = epoch;
    error = 0;
    for b = 1 : ntrain_batches
        plan.input.GetImage(1);
        ForwardPass();
        e = plan.classifier.GetScore(5);
        error = error + e;
        fprintf('(%d - %d) Train error: %d / %d = %f %% \t (%d / %d = %f %%)\n', b, plan.input.step - 1, e, bs, e / bs, error, b * bs, error / (b * bs));
        train_err(end+1) = error / (b * bs);
        step(end+1) = plan.input.step;
        PartialBP(min_layer);
        
        if mod(b, 100) == 0
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
    end
end


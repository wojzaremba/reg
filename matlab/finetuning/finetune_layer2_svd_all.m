C_(CleanGPU);
clear all;
C_(SetDevice, 2);
global plan;
randn('seed', 1);
load_imagenet_model('matthew_train', 128);
plan.training = 1;

approx_layer = 5;
W = plan.layer{approx_layer}.cpu.vars.W;

iclust = 2;
oclust = 2;
oratio = 0.5; % (0.6 --> 76), (0.5 --> 64)
iratio = 0.4;  % (0.6 --> 78), (0.5 --> 24), (0.4 --> 19)
odegree = floor(size(W, 1) * oratio / oclust);
idegree = floor(size(W, 4) * iratio / iclust);

fname = sprintf('layer2_svd_%d_%d_%d_%d_finetuneall', iclust, oclust, idegree, odegree);


restart = 0;
if ~restart
  load_weights_training(fname, 2);
  fprintf('\nLoading weights from %s\n\n', fname);
end

% Replace second conv layer with approximated weights


in_s = 55;
out_s = 51;

[Wapprox, C, Z, F, idx_input, idx_output] = bispace_svd(W, iclust, iratio, oclust, oratio, 0, in_s, out_s);

L2_err = norm(W(:) - Wapprox(:)) / norm(W(:));
fprintf('||W - Wapprox|| / ||W|| = %f\n', L2_err);
C_(CopyToGPU, plan.layer{approx_layer}.gpu.vars.W, single(Wapprox));
plan.layer{approx_layer}.cpu.vars.W = single(Wapprox);

% Finetuning parameters
min_layer = 6;
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
step = [];
for epoch = 1 : nepoch
    plan.input.step = nval_batches + 1;
    plan.repeat = epoch;
    error = 0;
    for b = 4000 : ntrain_batches
        
%         if b > 2000 
%             plan.lr = 0.0001;
%         end
%         if b > 4000
%             plan.lr = 0.00001;
%         end
        
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

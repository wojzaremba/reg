C_(CleanGPU);
clear all;
global plan;
randn('seed', 1);
load_imagenet_model('matthew_train', 128);
min_layer = 13;
plan.momentum = 0.9;
plan.lr = 0.0001;

bs = plan.input.batch_size;
nval_batches = 0;
nepoch = 1;
ntrain_batches = 75;
eval_freq = 5;
train_err = [];
val_err = [];
step = [];
for epoch = 1 : nepoch
    plan.input.step = nval_batches + 1;
    error = 0;
    for b = 1 : ntrain_batches
        plan.input.GetImage(1);
        C_(CopyToGPU, plan.input.gpu.vars.out, single(plan.input.cpu.vars.out));
        ForwardPass();
        e = plan.classifier.GetScore(5);
        error = error + e;
        fprintf('(%d - %d) Train error: %d / %d = %f %% \t (%d / %d = %f %%)\n', b, plan.input.step - 1, e, bs, e / bs, error, b * bs, error / (b * bs));
        train_err(end+1) = error / (b * bs);
        step(end+1) = plan.input.step;
        PartialBP(min_layer);
        
%         if mod(b, eval_freq) == 0
%             old_step = plan.input.step;
%             plan.input.step = 1;
%             val_error = 0;
%             for bb = 1 : nval_batches
%                 plan.input.GetImage(0);
%                 ForwardPass();
%                 val_error = val_error + plan.classifier.GetScore(5);
%             end   
%             fprintf('Val error: %d / %d = %f %%\n', val_error, bb * bs, val_error / (bb * bs));
%             val_err(end+1) = val_error / (bb * bs);
%             plan.input.step = old_step;
%         end
    end
end


global plan
init();
load_imagenet_model();
plan.input.step = 1;
bs = plan.input.batch_size;
nval_batches = 156;
nepoch = 1;
ntrain_batches = 100;
eval_freq = 1;

for epoch = 1 : nepoch
    plan.input.step = nval_batches + 1;
    error = 0;
    for b = 1 : ntrain_batches
        plan.input.GetImage(1);
        ForwardPass();
        error = error + plan.classifier.GetScore(5);
        fprintf('Val error: %f %%\n', error, b * bs, error / (b * bs));
        BackwardPass();
        
        if mod(b, eval_freq) == 0
            old_step = plan.input.step;
            plan.input.step = 1;
            val_error = 0;
            for bb = 1 : nval_batches
                plan.input.GetImage(0);
                ForwardPass();
                val_error = val_error + plan.classifier.GetScore(5);
                fprintf('Val error: %f %%\n', val_error, bb * bs, val_error / (bb * bs));
            end               
            plan.input.step = old_step;
        end
    end
end
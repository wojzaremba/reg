global plan;
load_imagenet_model();
min_layer = 13;
plan.momentum = 0.9;
plan.lr = 0.00001;

plan.input.step = 1;
bs = plan.input.batch_size;
nval_batches = 16;
nepoch = 1;
ntrain_batches = 100;
eval_freq = 5;
train_err = [];
val_err = [];
for epoch = 1 : nepoch
    plan.input.step = nval_batches + 1;
    error = 0;
    for b = 1 : ntrain_batches
        plan.input.GetImage(1);
        ForwardPass();
        error = error + plan.classifier.GetScore(5);
        fprintf('Train error: %d / %d = %f %%\n', error, b * bs, error / (b * bs));
        train_err(end+1) = error / (b * bs);
        PartialBP(min_layer);
        
        if mod(b, eval_freq) == 0
            old_step = plan.input.step;
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
        end
    end
end


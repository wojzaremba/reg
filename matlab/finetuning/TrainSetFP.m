C_(CleanGPU);
clear all;
C_(SetDevice, 0);
global plan;
randn('seed', 1);
load_imagenet_model('matthew_train', 128);
nimg = length(plan.input.Y);

bs = plan.input.batch_size;
ntrain_batches = floor(nimg / bs);
train_err = [];
error = 0;
plan.input.step = 1;
plan.training = 0;
for b = 1 : ntrain_batches
    plan.input.GetImage(1);
    ForwardPass();
    e = plan.classifier.GetScore(5);
    error = error + e;
    fprintf('(%d) Train error: %d / %d = %f %% \t (%d / %d = %f %%)\n', b, e, bs, e / bs, error, b * bs, error / (b * bs));
    train_err(end+1) = error / (b * bs);
end



% 197674 / 1165952 = 0.169539

% 216182 / 1166080 = 0.185392  ( cropped imgs)

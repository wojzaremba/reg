C_(CleanGPU);
clear all;
C_(SetDevice, 2);
global plan;
randn('seed', 1);
load_imagenet_model('matthew', 128);

layer_nr = 15;
fc = 3;
fname = sprintf('generated_mats/FC%d_approx_errors.mat', fc);

if (~exist('W', 'var'))
    W = plan.layer{layer_nr}.cpu.vars.W;
    [u, s, v] = svd(W);
end

errors = [];
rank_list = 50:50:1000;
for rr = 1 : length(rank_list)
    rank = rank_list(rr);
    uu = u(:, 1:rank);
    ss = s(1 : rank, 1 : rank);
    vv = v(:, 1:rank);
    Wapprox = single(uu * ss * vv');
    fprintf('||W _ Wapprox|| / ||W|| = %f\n', norm(W(:) - Wapprox(:)) / norm(W(:)));

    orig_weights = prod(size(W));
    approx_weights = prod(size(uu)) + prod(size(ss*vv'));
    fprintf('Weight gain: %f\n', orig_weights / approx_weights);

    orig_ops = prod(size(W));
    approx_ops = (size(W, 1) + size(W, 2)) * rank;
    fprintf('Ops gain: %f\n', orig_ops / approx_ops);

    if plan.layer{layer_nr}.on_gpu
        C_(CopyToGPU, plan.layer{layer_nr}.gpu.vars.W, single(Wapprox));
    else
        plan.layer{layer_nr}.cpu.vars.W = single(Wapprox);
    end

    % Get error
    error = 0;
    plan.input.step = 1;
    nbatches = 390;

    for i = 1:nbatches
        plan.input.GetImage(0);
        ForwardPass(); 
        error = error + plan.classifier.GetScore(5);
        fprintf('%d / %d = %f \n', error, i * plan.input.batch_size, error / (i * plan.input.batch_size));
    end
        
    load(fname);
    idx = find(rank_list == rank);
    errors(idx) =  error / (i * plan.input.batch_size);
    save(fname, 'errors', 'rank_list');


end


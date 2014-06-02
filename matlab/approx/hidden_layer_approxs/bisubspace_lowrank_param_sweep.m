C_(CleanGPU);
clear all;
C_(SetDevice, 2);
global plan;
randn('seed', 1);
load_imagenet_model;

W = plan.layer{5}.cpu.vars.W;

rank_list = [6, 8, 10, 12, 14, 16];
errors = [];
for rr = 1 : length(rank_list)
    rank = rank_list(rr);
    % Replace first convolutional layer weights with approximated weights
    args.iclust = 48;
    args.oclust = 2;
    args.k = rank;
    args.in_s = 55;
    args.out_s = 51;
    args.cluster_type = 'kmeans';

    fprintf('||W|| = %f \n', norm(W(:)));

    [Wapprox, F, C, XY, perm_in, perm_out, num_weights] = bisubspace_lowrank_approx_nosep(double(W), args);
    L2_err = norm(W(:) - Wapprox(:)) / norm(W(:));
    fprintf('||W - Wapprox|| / ||W|| = %f\n', L2_err);


    if plan.layer{5}.on_gpu
        C_(CopyToGPU, plan.layer{5}.gpu.vars.W, single(Wapprox));
    else
        plan.layer{5}.cpu.vars.W = single(Wapprox);
    end

    % Get error
    error = 0;
    plan.input.step = 1;
    nbatches = 390;
    for i = 1:nbatches
        plan.input.GetImage(0);
        ForwardPass(); 
        e = plan.classifier.GetScore(5);
        error = error + e;
        fprintf('(%d) %d / %d = %f     (%d / %d = %f)\n', i, e,  plan.input.batch_size, e /  plan.input.batch_size, error, i * plan.input.batch_size, error / (i * plan.input.batch_size));
    end

    load generated_mats/layer2_bisubspace_48_2_error.mat
    idx = find(rank_list == rank);
    if isempty(idx)
        rank_list(end+1) = rank;
        errors(end+1) =  error / (i * plan.input.batch_size);
    else
        errors(idx) =  error / (i * plan.input.batch_size);
    end
    save('generated_mats/layer2_bisubspace_48_2_error.mat', 'errors', 'rank_list');
end

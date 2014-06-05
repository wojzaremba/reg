function [] = load_weights_training(fname, start_layer)
    global plan
    if isempty(findstr(fname, 'scratch'))
        load(sprintf('generated_mats/%s.mat', fname));
    else
        load(fname);
    end
    
    weighted_layers = [5, 8, 10, 11, 13, 15, 17];
    for i = start_layer : length(weighted_layers)
       layer_nr = weighted_layers(i);
       
       % Copy W
       plan.layer{layer_nr}.cpu.vars.W = reshape(single(trained_weights.W{i}), size(plan.layer{layer_nr}.cpu.vars.W));
       if plan.layer{layer_nr}.on_gpu
          C_(CopyToGPU, plan.layer{layer_nr}.gpu.vars.W,  plan.layer{layer_nr}.cpu.vars.W);
       end
       
       % Copy B
       plan.layer{layer_nr}.cpu.vars.B = reshape(single(trained_weights.B{i}), size(plan.layer{layer_nr}.cpu.vars.B));
       if plan.layer{layer_nr}.on_gpu
          C_(CopyToGPU, plan.layer{layer_nr}.gpu.vars.B,  plan.layer{layer_nr}.cpu.vars.B);
       end
    end
    
end
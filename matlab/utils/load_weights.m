function [] = load_weights(fname, start_layer)
    global plan
    if isempty(findstr(fname, 'scratch'))
        load(sprintf('generated_mats/%s.mat', fname));
    else
        load(fname);
    end
    weighted_layers = [5, 8, 10, 11, 13, 14, 15];
    for i = start_layer : length(weighted_layers)
       layer_nr = weighted_layers(i);
       
       % If dropout was used, scale weights back
%        if strcmp(plan.layer{layer_nr}.type, 'FC')
%            fprintf('Scaling FC%d layer weights back down...\n', layer_nr);
%            trained_weights.W{i} = 0.5 * trained_weights.W{i};
%        end
%        
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
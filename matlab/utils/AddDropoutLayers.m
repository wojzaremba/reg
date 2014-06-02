function [] = AddDropoutLayers()
% Asssumes dropout rate is 0.5
% 14 --> 15
% 15 --> 17
    global plan;
    plan.all_uploaded_weights = struct();
    plan.all_uploaded_weights.plan = struct();
    plan.all_uploaded_weights.plan.layer = {};
    plan.all_uploaded_weights.plan.layer{15} = plan.layer{14};
    plan.all_uploaded_weights.plan.layer{17} = plan.layer{15};
    
    jsons = ParseJSON('plans/imagenet_matthew.txt');    
    end_layers = plan.layer(14:end);
    plan.layer = plan.layer(1:13);
    json =  struct('type', 'Dropout', 'depth', 4096, 'p', 0.5); 
    plan.layer{14} =  eval(sprintf('%s(json);', json.type()));
    
    plan.layer{15} = eval(sprintf('%s(jsons{14});', jsons{14}.type()));
    plan.layer{15}.cpu.vars.W = single(plan.layer{15}.cpu.vars.W * 2);
    if plan.layer{15}.on_gpu
        C_(CopyToGPU, plan.layer{15}.gpu.vars.W, single(plan.layer{15}.cpu.vars.W));
    end
    
    plan.layer{16} = eval(sprintf('%s(json);', json.type()));
    
    plan.layer{17} = eval(sprintf('%s(jsons{15});', jsons{15}.type()));
    plan.layer{17}.cpu.vars.W = single(plan.layer{17}.cpu.vars.W * 2);
    if plan.layer{17}.on_gpu
        C_(CopyToGPU, plan.layer{17}.gpu.vars.W, single(plan.layer{17}.cpu.vars.W));
    end
    
    plan.layer{18} = eval(sprintf('%s(jsons{14});', jsons{16}.type()));
    
    plan.all_uploaded_weights = [];
    
end
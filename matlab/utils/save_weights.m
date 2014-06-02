function [] = save_weights(fname, train_err, val_err)
    global plan
    
    trained_weights = struct();

    trained_weights.W = {};
    trained_weights.W{1} = plan.layer{5}.cpu.vars.W;
    trained_weights.W{2} = plan.layer{8}.cpu.vars.W;
    trained_weights.W{3} = plan.layer{10}.cpu.vars.W;
    trained_weights.W{4} = plan.layer{11}.cpu.vars.W;
    trained_weights.W{5} = plan.layer{13}.cpu.vars.W;
    trained_weights.W{6} = plan.layer{15}.cpu.vars.W;
    trained_weights.W{7} = plan.layer{17}.cpu.vars.W;

    trained_weights.B = {};
    trained_weights.B{1} = plan.layer{5}.cpu.vars.B;
    trained_weights.B{2} = plan.layer{8}.cpu.vars.B;
    trained_weights.B{3} = plan.layer{10}.cpu.vars.B;
    trained_weights.B{4} = plan.layer{11}.cpu.vars.B;
    trained_weights.B{5} = plan.layer{13}.cpu.vars.B;
    trained_weights.B{6} = plan.layer{15}.cpu.vars.B;
    trained_weights.B{7} = plan.layer{17}.cpu.vars.B;
    
    save(sprintf('generated_mats/%s.mat', fname), 'trained_weights', 'train_err', 'val_err');
end
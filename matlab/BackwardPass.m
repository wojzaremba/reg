function BackwardPass()
global plan
    for i = length(plan.layer):-1:2
        printf(2, 'BP for %s\n', plan.layer{i}.name);        
        fptic = tic;
        plan.layer{i}.BP();
        lapse = toc(fptic);
        plan.time.bp(i) = lapse;
        plan.layer{i}.Update();
        plan.layer{i - 1}.cpu.dvars.out = plan.layer{i}.cpu.dvars.X;
    end
end

function BackwardPass()
    global plan
    for i = length(plan.layer):-1:2
        printf(2, 'BP for %s\n', plan.layer{i}.name);    
        
        if plan.layer{i}.on_gpu
            C_(StartTimer);
            plan.layer{i}.BPgpu();
            lapse = C_(StopTimer);
        else 
            fptic = tic;
            plan.layer{i}.BP();
            lapse = toc(fptic);
        end
        plan.time.bp(i) = lapse;
        plan.layer{i}.Update();
        
        % This layer and next on gpu
        if plan.layer{i}.on_gpu && plan.layer{i-1}.on_gpu
            plan.layer{i-1}.gpu.dvars.out = plan.layer{i}.gpu.dvars.X;
        elseif plan.layer{i}.on_gpu && ~plan.layer{i-1}.on_gpu
            plan.layer{i-1}.cpu.dvars.out = C_(CopyFromGPU, plan.layer{i}.dvars.X);
        elseif ~plan.layer{i}.on_gpu && plan.layer{i-1}.on_gpu
            C_(CopyToGPU, plan.layer{i-1}.gpu.dvars.out, plan.layer{i}.cpu.dvars.X);
        elseif ~plan.layer{i}.on_gpu && ~plan.layer{i-1}.on_gpu
            plan.layer{i-1}.cpu.dvars.out = plan.layer{i}.cpu.dvars.X;
        end
    end
end
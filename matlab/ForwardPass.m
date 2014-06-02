function ForwardPass()
global plan
for i = 2:length(plan.layer)
  obj = plan.layer{i};
  printf(2, 'FP for %s\n', obj.name);
  if (obj.on_gpu)
      if (plan.layer{obj.layer_nr - 1}.on_gpu == 0)
        C_(CopyToGPU, obj.gpu.vars.X,  single(plan.layer{obj.layer_nr - 1}.cpu.vars.out));
      else
        obj.gpu.vars.X = plan.layer{obj.layer_nr - 1}.gpu.vars.out;
      end 
      C_(StartTimer);
      obj.FPgpu();
%       obj.cpu.vars.X = plan.layer{obj.layer_nr - 1}.cpu.vars.out; % XXX : remove
%       obj.FP(); % XXX : remove
      lapse = C_(StopTimer);
  else
      if (plan.layer{obj.layer_nr - 1}.on_gpu == 1)
        obj.cpu.vars.X = reshape(C_(CopyFromGPU, plan.layer{obj.layer_nr - 1}.gpu.vars.out), size(obj.cpu.vars.X));
      else
        obj.cpu.vars.X = plan.layer{obj.layer_nr - 1}.cpu.vars.out; 
      end 
      fptic = tic;
      obj.FP();
      lapse = toc(fptic);
  end 
  plan.time.fp(plan.input.step - 1, obj.layer_nr) = lapse;
end
end

init
global plan; load_imagenet_model('matthew', 1)
global debug; debug = 4;
plan.input.step = 1;
plan.input.GetImage(1);
%ForwardPass();
plan.momentum = 0;
plan.lr = 0.1;

for i = 2:length(plan.layer)
  obj = plan.layer{i};
  printf(2, 'FP for %s ', obj.name);

   
  if (isfield(obj.cpu.vars, 'W'))
      W = obj.cpu.vars.W;
      W_gpu = C_(CopyFromGPU, obj.gpu.vars.W);
      assert(norm(W(:) - W_gpu(:)) / norm(W(:)) < 1e-4);

      B = obj.cpu.vars.B;
      B_gpu = C_(CopyFromGPU, obj.gpu.vars.B);
      assert(norm(B(:) - B_gpu(:)) / norm(B(:)) < 1e-4);
  end
  
  if (plan.layer{obj.layer_nr - 1}.on_gpu == 0)
    C_(CopyToGPU, obj.gpu.vars.X,  single(plan.layer{obj.layer_nr - 1}.cpu.vars.out));
  else
    obj.gpu.vars.X = plan.layer{obj.layer_nr - 1}.gpu.vars.out;
  end 
  obj.FPgpu();
  obj.cpu.vars.X = plan.layer{obj.layer_nr - 1}.cpu.vars.out; % XXX : remove
  obj.FP(); % XXX : remove
  
  out = obj.cpu.vars.out;
  out_gpu = C_(CopyFromGPU, obj.gpu.vars.out);
  if strcmp(obj.type, 'Softmax')
      pred = C_(CopyFromGPU, obj.gpu.vars.pred);
      out_gpu = -log(pred(logical(plan.input.cpu.vars.Y)));
  end
  fprintf('\t\t err = %f\n', (norm(out(:) - out_gpu(:)) / norm(out(:))));
  
end


for i = length(plan.layer) : -1 : 2
    obj = plan.layer{i};
    obj.BP();
    obj.BPgpu();
    
    fprintf('\nLayer %d:\n', i);
    dX = obj.cpu.dvars.X;
    dX_gpu = C_(CopyFromGPU, obj.gpu.dvars.X);
    fprintf('dX : %f\n',norm(dX(:) - dX_gpu(:)) / norm(dX(:)));
    
    if (isfield(obj.cpu.vars, 'W'))
        dW = obj.cpu.dvars.W;
        dW_gpu = C_(CopyFromGPU, obj.gpu.dvars.W);
        fprintf('dW : %f\n', norm(dW(:) - dW_gpu(:)) / norm(dW(:)));

        dX = obj.cpu.dvars.X;
        dX_gpu = C_(CopyFromGPU, obj.gpu.dvars.X);
        fprintf('dB : %f\n', norm(dX(:) - dX_gpu(:)) / norm(dX(:)));
    end
    
    obj.cpu.dvars.X = C_(CopyFromGPU, obj.gpu.dvars.X); 
    
    plan.layer{i-1}.cpu.dvars.out = obj.cpu.dvars.X;
	C_(CopyToGPU, plan.layer{i-1}.gpu.dvars.out, obj.cpu.dvars.X); 
end
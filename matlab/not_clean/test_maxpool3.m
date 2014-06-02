init
global plan; load_imagenet_model('matthew', 1)
global debug; debug = 4;
randn('seed', 1);

layer_nr = 8;
obj = plan.layer{layer_nr};

obj.cpu.vars.X = single(randn(size(obj.cpu.vars.X)));
C_(CopyToGPU, obj.gpu.vars.X, obj.cpu.vars.X);

obj.cpu.vars.W = single(randn(size(obj.cpu.vars.W)));
C_(CopyToGPU, obj.gpu.vars.W, obj.cpu.vars.W);

obj.cpu.vars.B = single(randn(size(obj.cpu.vars.B)));
C_(CopyToGPU, obj.gpu.vars.B, obj.cpu.vars.B);

obj.FP();
obj.FPgpu();
cpu_out = obj.cpu.vars.out;
gpu_out = C_(CopyFromGPU, obj.gpu.vars.out);
assert(norm(cpu_out(:) - gpu_out(:)) / norm(cpu_out(:)) < 1e-1);

layer_nr = 9;
obj = plan.layer{layer_nr};

obj.cpu.vars.X = cpu_out + 0.01*randn(size(cpu_out));
C_(CopyToGPU, obj.gpu.vars.X, single(obj.cpu.vars.X));

obj.FP();
obj.FPgpu();
cpu_out = obj.cpu.vars.out;
gpu_out = C_(CopyFromGPU, obj.gpu.vars.out);
assert(norm(cpu_out(:) - gpu_out(:)) / norm(cpu_out(:)) < 1e-1);

% Create new dOut
if ~(isfield(obj.gpu.dvars, 'out'))
    obj.gpu.dvars.out = plan.GetGID();
end
obj.cpu.dvars.out = single(randn(size(obj.cpu.dvars.out)));
C_(CopyToGPU, obj.gpu.dvars.out, single(obj.cpu.dvars.out));

% Make sure X and out align
C_(CopyToGPU, obj.gpu.vars.out, single(obj.cpu.vars.out));
%C_(CopyToGPU, obj.gpu.vars.X, single(obj.cpu.vars.X));

obj.BP();
obj.BPgpu();

cv = obj.cpu.vars;
cd = obj.cpu.dvars;
gv = obj.gpu.vars;
gd = obj.gpu.dvars;

cpu_dX = cd.X;
gpu_dX = C_(CopyFromGPU, gd.X);
         
fprintf('Norm of cpu_dX = %f\n', norm(cpu_dX(:)) );
fprintf('Norm of gpu_dX = %f\n', norm(gpu_dX(:)) );

gpu_dX = reshape(gpu_dX, size(cpu_dX));
diff = cpu_dX - gpu_dX;

fprintf('Norm of cpu_dX - gpu_dX = %f\n', norm(diff(:)) / norm(cpu_dX(:)) );
assert(norm(diff(:)) / norm(cpu_dX(:)) < 1e-2);


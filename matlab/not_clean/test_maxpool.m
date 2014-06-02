clear all;

json = struct();
randn('seed', 1);
json.type = 'MaxPooling';
json.local_2d_patch = struct('patch_rows', 3, ...
                             'patch_cols', 3,...
                            'stride_rows', 2,...
                            'stride_cols', 2);
                           
jsons = {};

jsons{1} = struct('batch_size', 128, 'rows', 12, 'cols', 12, 'depth', 512, 'type', 'TestInput', 'on_gpu', 1);
jsons{2} = json;
jsons{2}.on_gpu = 1;
plan = Plan(jsons);

obj = plan.layer{2};

cv = obj.cpu.vars;
cd = obj.cpu.dvars;
gv = obj.gpu.vars;
gd = obj.gpu.dvars;

cv.X = single(randn(size(cv.X)));
C_(CopyToGPU, gv.X, cv.X);

% Copy vars back and FP
obj.cpu.vars = cv;
obj.cpu.dvars = cd;
obj.gpu.vars = gv;
obj.gpu.dvars = gd;
obj.FP();
obj.FPgpu();

cv = obj.cpu.vars;
cd = obj.cpu.dvars;
gv = obj.gpu.vars;
gd = obj.gpu.dvars;

cpu_out = cv.out;
gpu_out = C_(CopyFromGPU, gv.out);

%assert(norm(cpu_out(:) - gpu_out(:)) / norm(cpu_out(:)) < 1e-4);

% Create dX and dOut vars
cd.out = single(randn(size(cd.out)));
if ~(isfield(gd, 'out'))
    gd.out = plan.GetGID();
end
C_(CopyToGPU, gd.out, cd.out);

% Copy vars back and BP
obj.cpu.vars = cv;
obj.cpu.dvars = cd;
obj.gpu.vars = gv;
obj.gpu.dvars = gd;
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


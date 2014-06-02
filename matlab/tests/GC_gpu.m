function GCgpu()
    clc;
    plan_json = ParseJSON('plans/tests.txt');
    for i = 1 : length(plan_json)
        jsons = {};
        if (strcmp(plan_json{i}.type, 'Softmax'))
            jsons{end + 1} = struct('batch_size', 3, 'rows', 1, 'cols', 1, 'depth', 4, 'type', 'TestInput', 'on_gpu', 1);
        elseif (strcmp(plan_json{i}.type, 'Conv'))
            jsons{end + 1} = struct('batch_size', 3, 'rows', 20, 'cols', 20, 'depth', 3, 'type', 'TestInput', 'on_gpu', 1);
        elseif ( strcmp(plan_json{i}.type, 'LRNormal') || strcmp(plan_json{i}.type, 'MaxPooling'))
            jsons{end + 1} = struct('batch_size', 3, 'rows', 16, 'cols', 16, 'depth', 16, 'type', 'TestInput', 'on_gpu', 1);
        else
            jsons{end + 1} = struct('batch_size', 3, 'rows', 8, 'cols', 10, 'depth', 4, 'type', 'TestInput', 'on_gpu', 1);
        end
        plan_json{i}.on_gpu = 1;
        jsons{end + 1} = plan_json{i};
        plan = Plan(jsons);
        fprintf('\n\nVerifing %d layer %s\n', i, plan.layer{end}.type);
        if (~VerifyLayer())
            assert(0);
            return;
        end
    end
end

function passed = VerifyLayer()
    global plan;
    passed = true;
    layer = plan.layer{end};
    dims = layer.dims;
    h = 1e-4;
    eps = 1e-2;
    layer.gpu.dvars.out = Val(layer.gpu.dvars, 'out', plan.GetGID());
    plan.input.GetImage(1);
    vars = layer.gpu.vars;
    vars.X = plan.input.gpu.vars.out;    
    back_in = single(randn([size(plan.input.cpu.vars.out, 1), dims]));
    if (strcmp(layer.type, 'Softmax'))
        back_in(:) = 1;
    end
    layer.gpu.vars = vars;
    layer.FPgpu();
    if (~ismethod(layer, 'BPgpu'))
        return;
    end
    layer.cpu.dvars.out = back_in;
    C_(CopyToGPU, layer.gpu.dvars.out, back_in);
    layer.BPgpu();
    dvars = layer.gpu.dvars;
    f = fields(dvars);
    for i = 1:length(f)
        if (strcmp(f{i}, 'out') || strcmp(f{i}, 'forward_act') || strcmp(f{i}, 'denoms') || strcmp(f{i}, 'pred') || strcmp(f{i}, 'max') || strcmp(f{i}, 'sum'))
            continue;
        end
        fprintf('Trying verify %s derivative....', f{i});
        name = f{i};
        V = eval(sprintf('C_(CopyFromGPU, layer.gpu.vars.%s);', name));
        dV = eval(sprintf('C_(CopyFromGPU, layer.gpu.dvars.%s);', name));
        dV_num = zeros(size(dV));
        for pos = 1:length(V(:))
            %fprintf('.');
            Vcopy_a = V;
            Vcopy_a(pos) = Vcopy_a(pos) - h;
            eval(sprintf('C_(CopyToGPU, layer.gpu.vars.%s, Vcopy_a);', name));
            layer.FPgpu();
            out_a = C_(CopyFromGPU, layer.gpu.vars.out);
            if strcmp(layer.type, 'Softmax')
                out_a = C_(CopyFromGPU, layer.gpu.vars.pred);
                out_a = -log(out_a(logical(plan.input.cpu.vars.Y)));
            end

            Vcopy_b = V;
            Vcopy_b(pos) = Vcopy_b(pos) + h;
            eval(sprintf('C_(CopyToGPU, layer.gpu.vars.%s, Vcopy_b);', name));
            layer.FPgpu();
            out_b = C_(CopyFromGPU, layer.gpu.vars.out);
            if strcmp(layer.type, 'Softmax')
                out_b = C_(CopyFromGPU, layer.gpu.vars.pred);
                out_b = -log(out_b(logical(plan.input.cpu.vars.Y)));
            end
            dV_num(pos) = dot((out_b(:) - out_a(:)) ./ (2 * h), back_in(:));
        end
        diff = dV_num - dV;
        try
            assert(norm(diff(:)) / max(norm(dV_num(:)), 1) < eps);
            assert(length(dV_num(:)) > 0)
            assert(length(dV(:)) > 0)
            fprintf('%s derivative check DONE', f{i});
            if (norm(dV_num(:)) < 1e-5)
                fprintf(', \nWARNING values close to zero (GC might be unreliable)\n');
                passed = false;
                return;
            end
            fprintf('\n');
        catch
            fprintf('\nnorm diff = %f\n', norm(diff(:)));
            dV_num ./ dV
            fprintf('%s derivative FAILED\n', f{i});
            passed = false;
            return;
        end
    end
end

classdef Layer < handle
    properties
        type
        dims % Output dimensions.
        cpu % variables stored on CPU. Contains parameters.
        gpu % variables stored on GPU. Contains parameters.
        json 
        layer_nr 
        Fun % Activation function.
        dFun % Derivative of activation function.        
        Fun_ % Activation function on GPU.
        dFun_
        patch
        stride
        padding
        gids %
        name
        on_gpu
    end
    
    methods
        
        function obj = Layer(json)
            if(nargin == 0)
                return;
            end
            obj.json = json;
            obj.InitializeVariables();
            obj.SetConnections();
            obj.SetInitialization();
        end
        
        function InitializeVariables(obj)
            global plan
            json = obj.json;
            obj.type = json.type;
            fname = Val(json, 'function', 'RELU');
            obj.Fun = eval(sprintf('@%s;', fname));
            obj.dFun = eval(sprintf('@d%s;', fname));  
             try
                obj.Fun_ = eval(sprintf('Act%s', fname));
                obj.dFun_ = eval(sprintf('dAct%s', fname));                        
            catch
                fprintf('Act%s or dAct%s not implemented on GPU\n', fname, fname);
            end
            obj.cpu = struct('vars', struct(), 'dvars', struct(), 'accum', struct());
            obj.gpu = struct('vars', struct(), 'dvars', struct(), 'accum', struct());
            obj.layer_nr = length(plan.layer) + 1;
            obj.on_gpu = Val(json, 'on_gpu', plan.default_on_gpu);
            obj.name = Val(json, 'name', [json.type, num2str(obj.layer_nr)]);            
        end
        
        function SetConnections(obj)
            json = obj.json;
            if (isfield(json, 'local_2d_patch'))
                patch = json.local_2d_patch;
                obj.patch = [patch.patch_rows, patch.patch_cols];
                obj.stride = [Val(patch, 'stride_rows', 1), Val(patch, 'stride_cols', 1)];
                obj.padding = [Val(patch, 'padding_rows', 0), Val(patch, 'padding_cols', 0)];
                dims = obj.prev_dim();
                new_dims = [ceil((dims(1:2) - obj.patch + 2 * obj.padding) ./ obj.stride) + 1, Val(json, 'depth', dims(3))];
                obj.dims = new_dims;
            else
                obj.patch = [1, 1];
                obj.stride = [1, 1];
                obj.padding = [0, 0];
                if (isfield(json, 'one2one') && json.one2one)
                    obj.dims = obj.prev_dim();
                else
                    obj.dims = [Val(json, 'rows', 1), Val(json, 'cols', 1), Val(json, 'depth', 1)];
                end
            end            
        end
        
        function InitWeights(obj)
        end
        
        function dim = prev_dim(obj)
            global plan
            dim = plan.layer{obj.layer_nr - 1}.dims;
        end
        
        function ret = depth(obj)
            ret = obj.dims(3);
        end
        
        function SetInitialization(obj)
            def_fields = {'mult', 'bias'};
            f_json = fields(obj.json);
            for k = 1:length(f_json)
                if (length(strfind(f_json{k}, 'init_fun')) > 0)
                    eval(sprintf('layer.%s = @%s;', f_json{k}, eval(sprintf('json.%s', f_json{k}))));
                end
                for t = 1:length(def_fields)
                    if (length(strfind(f_json{k}, def_fields{t})) > 0)
                        eval(sprintf('layer.%s = json.%s;', f_json{k}, f_json{k}));
                    end
                end
            end
        end
        
        function RandomWeights(obj, name, dim)
            global plan
            try
                funname = eval(sprintf('obj.init.%s.init_fun', name));
                mult = eval(sprintf('obj.init.%s.mult', name));
                bias = eval(sprintf('obj.init.%s.bias', name));
            catch
                if (strcmp(name, 'W'))
                    funname = 'GAUSSIAN';
                    mult = 0.01;
                    bias = 0;
                else
                    funname = 'CONSTANT';
                    mult = 0;
                    bias = 0;
                end
            end
            eval(sprintf('obj.cpu.vars.%s = single(obj.%s(dim, mult, bias));', name, funname));
        end
        
        function ret = GAUSSIAN(obj, dim, mult, bias)
            ret = randn(dim) * mult + bias;
        end
        
        function ret = UNIFORM(obj, dim, mult, bias)
            ret = rand(dim) * mult + bias;
        end
        
        function ret = CONSTANT(obj, dim, mult, bias)
            assert(bias == 0);
            ret = mult * ones(dim);
        end
        
        function ret = F(obj, X)
            ret = obj.Fun(obj, X);
        end
        
        function ret = dF(obj, X)
            ret = obj.dFun(obj, X);
        end        
        
        function ret = LINEAR(obj, X)
            ret = X;
        end

        function ret = dLINEAR(obj, X)
            ret = ones(size(X));
        end        
        
        function ret = RELU(obj, X)
            ret = max(X, 0);
        end
        
        function ret = dRELU(obj, X)
            ret = X > 0;
        end        
        
        function ret = SIGMOID(obj, X)
            ret = 1 ./ (1 + exp(-X));
        end
        
        function Update(layer)
            global plan;
            lr = plan.lr;
            if (lr == 0)
                return;
            end 
            momentum = plan.momentum;
            if (~layer.on_gpu)
                f = fields(layer.cpu.dvars);
                for i = 1:length(f)
                    if (strcmp(f{i}, 'mask')) || (strcmp(f{i}, 'X')) || (strcmp(f{i}, 'out')) || (strcmp(f{i}, 'max')) || (strcmp(f{i}, 'sum')) || (strcmp(f{i}, 'forward_act')) || (strcmp(f{i}, 'pred')) % Don't backprop sum and max (just temp vars) from softmax
                        continue;
                    end
                    name = f{i};                 
                    eval(sprintf('layer.cpu.accum.%s = (1 - momentum) * layer.cpu.dvars.%s / plan.input.batch_size + momentum * layer.cpu.accum.%s;', name, name, name));
                    eval(sprintf('layer.cpu.vars.%s = layer.cpu.vars.%s - lr * layer.cpu.accum.%s;', name, name, name));
                end
            else                
                f = fields(layer.gpu.dvars);
                for i = 1:length(f)
                    if (strcmp(f{i}, 'mask')) || (strcmp(f{i}, 'X')) || (strcmp(f{i}, 'out')) || (strcmp(f{i}, 'max')) || (strcmp(f{i}, 'sum')) || (strcmp(f{i}, 'forward_act')) || (strcmp(f{i}, 'pred')) % Don't backprop sum and max (just temp vars) from softmax
                        continue;
                    end
                    name = f{i};
                    vars_gid = eval(sprintf('layer.gpu.vars.%s', name));
                    dvars_gid = eval(sprintf('layer.gpu.dvars.%s', name));
                    accum_gid = eval(sprintf('layer.gpu.accum.%s', name));
                    
                    % Copy back to cpu
                    eval(sprintf('layer.cpu.vars.%s = C_(CopyFromGPU, vars_gid);', name));
                    eval(sprintf('layer.cpu.dvars.%s = C_(CopyFromGPU, dvars_gid);', name));
                    eval(sprintf('layer.cpu.accum.%s = C_(CopyFromGPU, accum_gid);', name));
                    
                    eval(sprintf('layer.cpu.accum.%s = (1 - momentum) * layer.cpu.dvars.%s / plan.input.batch_size + momentum * layer.cpu.accum.%s;', name, name, name));
                    eval(sprintf('layer.cpu.vars.%s = layer.cpu.vars.%s - lr * layer.cpu.accum.%s;', name, name, name));
                    
                    eval(sprintf('C_(CopyToGPU, vars_gid, layer.cpu.vars.%s)', name));
                    eval(sprintf('C_(CopyToGPU, dvars_gid, layer.cpu.dvars.%s)', name));
                    eval(sprintf('C_(CopyToGPU, accum_gid, layer.cpu.accum.%s)', name));
                    
                    % accum = (1 - momentum) * dvars / bs + momentum * accum
%                     C_(Scale, accum_gid, single(momentum), accum_gid);
%                     C_(Scale, layer.gpu.vars.temp, single((1 - momentum) / plan.input.batch_size), dvars_gid);
%                     C_(Add, accum_gid, layer.gpu.vars.temp, accum_gid);
%                     
%                     % vars = vars - lr * accum;
%                     C_(Scale, accum_gid, single(lr), layer.gpu.vars.temp);
%                     C_(Subtract, vars_gid, layer.gpu.vars.temp, vars_gid);
                    
                    C_(Scale, accum_gid, single(momentum), accum_gid);
                    C_(Scale, dvars_gid, single((1 - momentum) / plan.input.batch_size), dvars_gid);
                    C_(Add, accum_gid, dvars_gid, accum_gid);
                    C_(Scale, dvars_gid, single(plan.input.batch_size / (1 - momentum)), dvars_gid); 
                  
                    C_(Scale, accum_gid, single(lr), accum_gid); % XXX : Fix it (lose of numerical precision.
                    C_(Subtract, vars_gid, accum_gid, vars_gid);
                    C_(Scale, accum_gid, single(1 / lr), accum_gid); % XXX : Fix it (lose of numerical precision.
                end                
            end
        end        
        
        function DisplayInfo(layer)
            global plan
            fprintf('%s \n', layer.type);
            f = fields(layer.cpu.vars);
            for i = 1:length(f)
                if (strcmp(f{i}, 'X'))
                    continue;
                end
                sparam = eval(sprintf('size(layer.cpu.vars.%s)', f{i}));
                
                fprintf('\n\t%s = [', f{i});
                for k = 1:length(sparam)
                    fprintf('%d ', sparam(k));
                end
                fprintf('] = %d', prod(sparam));                
                
                try
                    fun = func2str(eval(sprintf('layer.init.%s.init_fun', f{i})));
                    mult = eval(sprintf('layer.init.%s.mult', f{i}));
                    bias = eval(sprintf('layer.inti.%s.bias', f{i}));
                    fprintf('fun = %s, mult = %f, bias = %f', fun, mult, bias);
                catch
                end
            end            
            fprintf('\n');
        end
                
        function AddParam(obj, name, dims, includeDer)
            obj.AddParamsOnlyCPU(name, dims, includeDer);
            obj.AddParamsOnlyGPU(name, dims, includeDer);
        end
        
        function AddParamsOnlyCPU(obj, name, dims, includeDer)
            global plan
            if (isempty(plan.all_uploaded_weights) || ~includeDer || strcmp(name, 'out') || strcmp(name, 'X'))
                obj.RandomWeights(name, dims);
            else
                eval(sprintf('obj.cpu.vars.%s = single(plan.all_uploaded_weights.plan.layer{length(plan.layer) + 1}.cpu.vars.%s);', name, name));
            end
            plan.stats.total_vars = plan.stats.total_vars + prod(dims);
            %if (includeDer)
                plan.stats.total_learnable_vars = plan.stats.total_learnable_vars + prod(dims);
                plan.stats.total_vars = plan.stats.total_vars + 2 * prod(dims);
                eval(sprintf('obj.cpu.accum.%s = zeros(dims);', name, name));
                eval(sprintf('obj.cpu.dvars.%s = zeros(dims);', name, name));
            %end
        end
        
        function AddParamsOnlyGPU(obj, name, dims, includeDer)
            global plan
            if (obj.on_gpu == 1)
                vartype = {'vars', 'dvars', 'accum'};
                for i = 1 : length(vartype)
                    var = eval(sprintf('single(obj.cpu.%s.%s)', vartype{i}, name));
                    eval(sprintf('obj.gpu.%s.%s = plan.GetGID();', vartype{i}, name));
                    gid = eval(sprintf('obj.gpu.%s.%s', vartype{i}, name));                
                    C_(CopyToGPU, gid, var);
                    plan.stats.total_vars_gpu = plan.stats.total_vars_gpu + prod(dims);                    
                end
            end
        end

        function Finalize(obj)
            global plan
            obj.InitWeights();
            dims = [plan.input.batch_size, obj.dims];
            obj.AddParamsOnlyCPU('out', dims, true);
            if (obj.layer_nr > 1)                
                pobj = plan.layer{obj.layer_nr - 1};
                obj.AddParamsOnlyCPU('X', [plan.input.batch_size, pobj.dims], true);                
            end
            if (obj.on_gpu)                
                % vars.out corresponds to the next layer vars.X.
                % dvars.X corrensponds to the previous layer dvars.out.
                obj.gpu.vars.out = plan.GetGID();
                C_(CopyToGPU, obj.gpu.vars.out, single(obj.cpu.vars.out));
                plan.stats.total_vars_gpu = plan.stats.total_vars_gpu + prod(dims);               
                
                if (obj.layer_nr > 1)
                    obj.gpu.dvars.X = plan.GetGID();
                    C_(CopyToGPU, obj.gpu.dvars.X, single(obj.cpu.dvars.X));
                    plan.stats.total_vars_gpu = plan.stats.total_vars_gpu + prod(dims);                   
                    pobj = plan.layer{obj.layer_nr - 1};
                    if (pobj.on_gpu)
                        obj.gpu.vars.X = pobj.gpu.vars.out;
                        pobj.gpu.dvars.out = obj.gpu.dvars.X;
                    else
                        obj.gpu.vars.X = plan.GetGID();
                        C_(CopyToGPU, obj.gpu.vars.X, single(obj.cpu.vars.X));
                        plan.stats.total_vars_gpu = plan.stats.total_vars_gpu + prod(dims);                                                
                    end
                end
            else
                if (obj.layer_nr > 1)       
                    pobj = plan.layer{obj.layer_nr - 1};
                    if (pobj.on_gpu)
                        pobj.gpu.dvars.out = plan.GetGID();
                        C_(CopyToGPU, pobj.gpu.dvars.out, single(pobj.cpu.dvars.out));
                        plan.stats.total_vars_gpu = plan.stats.total_vars_gpu + prod(dims);                
                    end
                end
            end
            obj.DisplayInfo();
        end        
    end
end

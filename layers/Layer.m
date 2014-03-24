classdef Layer < handle
    properties
        type
        dims % Output dimensions.
        cpu % variables stored on CPU. Contains parameters.
        json 
        layer_nr 
        Fun % Activation function.
        dFun % Derivative of activation function.        
        Fun_ % Activation function on GPU.
        patch
        stride
        padding
        gids %
        name
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
            obj.cpu = struct('vars', struct(), 'dvars', struct(), 'accum', struct());
            obj.layer_nr = length(plan.layer) + 1;
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
            f = fields(layer.cpu.dvars);
            for i = 1:length(f)
		if (strcmp(f{i}, 'X')) || (strcmp(f{i}, 'out'))
		    continue;
		end 
		name = f{i};    
		eval(sprintf('layer.cpu.accum.%s = (1 - momentum) * layer.cpu.dvars.%s / plan.input.batch_size + momentum * layer.cpu.accum.%s;', name, name, name));
		eval(sprintf('layer.cpu.vars.%s = layer.cpu.vars.%s - lr * layer.cpu.accum.%s;', name, name, name));
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
            global plan
            if (isempty(plan.all_uploaded_weights) || ~includeDer || strcmp(name, 'out') || strcmp(name, 'X'))
                obj.RandomWeights(name, dims);
            else
                eval(sprintf('obj.cpu.vars.%s = plan.all_uploaded_weights.plan.layer{length(plan.layer) + 1}.cpu.vars.%s;', name, name));
            end
            plan.stats.total_vars = plan.stats.total_vars + prod(dims);
            if (includeDer)
                plan.stats.total_learnable_vars = plan.stats.total_learnable_vars + prod(dims);
                plan.stats.total_vars = plan.stats.total_vars + 2 * prod(dims);
                eval(sprintf('obj.cpu.accum.%s = zeros(dims);', name, name));
                eval(sprintf('obj.cpu.dvars.%s = zeros(dims);', name, name));
            end
        end

        function Finalize(obj)
            global plan
            obj.InitWeights();
            dims = [plan.input.batch_size, obj.dims];
            obj.AddParam('out', dims, true);
            if (obj.layer_nr > 1)                
                pobj = plan.layer{obj.layer_nr - 1};
                obj.AddParam('X', [plan.input.batch_size, pobj.dims], true);                
            end
            obj.DisplayInfo();
        end        
    end
end

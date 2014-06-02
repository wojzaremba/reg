classdef Dropout < Layer
    properties
        p
    end
    
    methods
        function obj = Dropout(json)
            obj@Layer(FillDefault(json));
            obj.p = json.p;
            obj.Finalize();
        end
        
        function FPgpu(obj)
            global plan;
            v = obj.gpu.vars;
            if (plan.training)
                rand('seed', 100000 * plan.repeat + plan.input.step);
                ran = rand(size(obj.cpu.vars.X));
                mask = single(ran > obj.p);
                C_(CopyToGPU, v.mask, mask);
                C_(EltwiseMult, v.mask, v.X, v.out);
            else
               C_(Scale, v.X, (1 - obj.p), v.out);
            end
            
        end
        
        function FP(obj)
            global plan;
            vars = obj.cpu.vars;
            out = zeros(size(vars.X));
            if (plan.training)
                rand('seed', 100000 * plan.repeat + plan.input.step);
                ran = rand(size(vars.X));
                idx = logical(ran > obj.p);
                obj.cpu.vars.idx = idx;
                out(idx) = vars.X(idx);
            else
                out = vars.X * (1 - obj.p);
            end
            obj.cpu.vars.out = out;
        end
        
        function BPgpu(obj)
            global plan;
            v = obj.gpu.vars;
            d = obj.gpu.dvars;
            if plan.training
                C_(EltwiseMult, d.out, v.mask, d.X);
            else
                C_(Scale, d.out, (1 - obj.p), d.X);                
            end
        end
        
        function BP(obj)                
            global plan
            data = obj.cpu.dvars.out;
            if (plan.training)
                dX = zeros(size(obj.cpu.vars.X));
                dX(obj.cpu.vars.idx) = data(obj.cpu.vars.idx);
            else
                dX = data * (1 - obj.p);
            end      
            obj.cpu.dvars.X = dX;
        end
        
        function InitWeights(obj)
            obj.AddParam('out', [prod(obj.dims(1:2)), obj.depth()], false);
            obj.AddParam('mask', [prod(obj.dims(1:2)), obj.depth()], false); 
        end        
        
    end
end

function json = FillDefault(json)
json.type = 'Dropout';
json.one2one = true;
end

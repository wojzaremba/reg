classdef FC < Layer
    properties
    end
    
    methods
        function obj = FC(json)
            obj@Layer(FillDefault(json));
            obj.Finalize();
        end      
        
        function FP(obj)
            v = obj.cpu.vars;            
            act = v.X(:, :) * v.W(:, :);
            act = act + repmat(v.B, size(act, 1), 1);
            obj.cpu.vars.forward_act = act;
            obj.cpu.vars.out = obj.F(act);
        end
        
        function BP(obj)
            X = obj.cpu.vars.X;
            W = obj.cpu.vars.W;
            act = obj.cpu.vars.forward_act;
            act = obj.dF(act);
            dX = act .* reshape(obj.cpu.dvars.out, size(act));      
            obj.cpu.dvars.W = X(:, :)' * dX; 
            obj.cpu.dvars.B = sum(dX, 1);
            obj.cpu.dvars.X = dX * W';
        end        
        
        function InitWeights(obj)
            obj.AddParam('B', [1, prod(obj.dims)], true);
            obj.AddParam('W', [prod(obj.prev_dim()), prod(obj.dims)], true);            
        end
    end
end

function json = FillDefault(json)
json.type = 'FC';
end

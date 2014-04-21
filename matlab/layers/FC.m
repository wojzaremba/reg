classdef FC < Layer
    properties
    end
    
    methods
        function obj = FC(json)
            obj@Layer(FillDefault(json));
            obj.Finalize();
        end      
        
        function FPgpu(obj)
            v = obj.gpu.vars;
            C_(Mult, v.X, v.W, v.out);
            C_(AddVector, v.out, v.B, v.out);
            C_(obj.Fun_, v.out, v.out);
        end
        
        function FP(obj)
            v = obj.cpu.vars;            
            act = v.X(:, :) * v.W(:, :);
            act = act + repmat(v.B, size(act, 1), 1);
            obj.cpu.vars.forward_act = act;
            obj.cpu.vars.out = obj.F(act);
        end
        
        function BPgpu(obj)
            dv = obj.gpu.dvars;
            v = obj.gpu.vars;
            if (obj.dFun_ == dActLINEAR)
                v.out = dv.out;
            else
                C_(obj.dFun_, v.out, v.out);                      
                C_(EltwiseMult, v.out, dv.out, v.out);  
            end
            C_(Sum, v.out, 0, dv.B);            
            C_(Transpose, v.X);
            C_(Mult, v.X, v.out, dv.W);
            C_(Transpose, v.W);
            C_(Mult, v.out, v.W, dv.X);
            C_(Transpose, v.X);            
            C_(Transpose, v.W);      
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

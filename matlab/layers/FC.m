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
            C_(Mult, v.X, v.W, v.forward_act);
            C_(AddVector, v.forward_act, v.B, v.forward_act);
            if (obj.Fun_ == ActLINEAR) % XXX : Fix this hack
                temp = C_(CopyFromGPU, v.forward_act);
                C_(CopyToGPU, v.out, temp);
            else
                C_(obj.Fun_, v.forward_act, v.out);
            end
        end
        
        function FP(obj)
            v = obj.cpu.vars;            
            act = v.X(:, :) * v.W(:, :);
            act = act + repmat(v.B, size(act, 1), 1);
            obj.cpu.vars.forward_act = act;
            obj.cpu.vars.out = obj.F(act);
        end
        
        function BPgpu(obj)
            d = obj.gpu.dvars;
            v = obj.gpu.vars;
            % dX = d.forward_act
            if (obj.dFun_ == dActLINEAR)
                %v.out = d.out;
                d.forward_act = d.out;
            else
                C_(obj.dFun_, v.forward_act, d.forward_act);                      
                C_(EltwiseMult, d.forward_act, d.out, d.forward_act);  
            end
            C_(Sum, d.forward_act, 0, d.B);            
            C_(Transpose, v.X);
            C_(Mult, v.X, d.forward_act, d.W);
            C_(Transpose, v.W);
            C_(Mult, d.forward_act, v.W, d.X);
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
            obj.AddParam('forward_act', [obj.dims], false);
        end
    end
end

function json = FillDefault(json)
json.type = 'FC';
end

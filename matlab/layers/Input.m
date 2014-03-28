classdef Input < Layer
    properties
        repeat
        batch_size
        translate
        step
        max_repeat         
    end
    methods
        function obj = Input(json)
            obj@Layer(json);
            global plan            
            obj.repeat = 1;
            obj.step = 1;            
            obj.batch_size = Val(json, 'batch_size', 6);            
            obj.translate = Val(json, 'translate', 0);            
            obj.max_repeat = 10000;    
            plan.input = obj;             
        end       

        function FPmatlab(obj) 
        end
        
        function GetImage(obj, train)        
            global plan
            [X, Y, obj.step] = GetImage_(obj, obj.step, train);
            obj.cpu.vars.out = X;
            obj.cpu.vars.Y = Y;
        end
    end
end



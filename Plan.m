classdef Plan < handle
    properties
        jsons
        debug
        stats
        layer
        input
        classifier
        gid
        time
        upload_weights
        all_uploaded_weights
        lr
        momentum
        training
    end
    
    methods
        function obj = Plan(param1, weights)
            if (ischar(param1))
                jsons = ParseJSON(param1);
            else
                jsons = param1;
            end
            
            obj.jsons = jsons;
            obj.gid = 0;  
            obj.debug = 0;
            randn('seed', 1);
            rand('seed', 1);            
            obj.layer = {};
            if (exist('weights', 'var')) && (~isempty(weights))
                obj.all_uploaded_weights = load(weights);
            end            
            global plan cuda
            plan = obj;     
            cuda = zeros(2, 1);            
            obj.stats = struct('total_vars', 0, 'total_learnable_vars', 0);
            for i = 1:length(jsons)
                json = jsons{i};
                if (strcmp(json.type(), 'Spec'))
                    obj.lr = json.lr;
                    obj.momentum = json.momentum;
                else
                    obj.layer{end + 1} = eval(sprintf('%s(json);', json.type()));
                end
            end
            fprintf('Total number of\n\ttotal learnable vars = %d\n\ttotal vars = %d\n', obj.stats.total_learnable_vars, obj.stats.total_vars);            
            obj.all_uploaded_weights = [];
        end        
        
        function gid = GetGID(obj)
            gid = obj.gid; 
            obj.gid = obj.gid + 1;            
        end
        
    end
end

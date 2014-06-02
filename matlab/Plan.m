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
        default_on_gpu
        upload_weights
        all_uploaded_weights
        lr
        momentum
        training
        repeat
    end

    methods
        function obj = Plan(param1, weights, default_on_gpu)
            if (ischar(param1))
                jsons = ParseJSON(param1);
            else
                jsons = param1;
            end
            if (exist('default_on_gpu', 'var'))
                obj.default_on_gpu = default_on_gpu;
            else
                obj.default_on_gpu = 0;
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
            obj.stats = struct('total_vars', 0, 'total_learnable_vars', 0, 'total_vars_gpu', 0);
            for i = 1:length(jsons)
                json = jsons{i};
                obj.layer{end + 1} = eval(sprintf('%s(json);', json.type()));
            end
            fprintf('Total number of\n\ttotal learnable vars = %d\n\ttotal vars = %d\n\ttotal vars on the gpu = %d\n', obj.stats.total_learnable_vars, obj.stats.total_vars, obj.stats.total_vars_gpu);
            obj.all_uploaded_weights = [];
        end

        function gid = GetGID(obj)
            gid = obj.gid;
            obj.gid = obj.gid + 1;
        end

    end
end

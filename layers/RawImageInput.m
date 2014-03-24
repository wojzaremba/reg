classdef RawImageInput < Input
    properties
        file_pattern
        meanX
        Y
    end
    methods
        function obj = RawImageInput(json)
            obj@Input(FillDefault(json));
            obj.file_pattern = json.file_pattern;
            tmp = load(sprintf('%s/meta.mat', obj.file_pattern));
            obj.meanX = tmp.meanX;
            obj.Y = tmp.Y;
            obj.Finalize();
        end       
        
        function [X, Y, batches] = LoadData(obj, file_pattern, batch_size)
            X = [];
            Y = [];
            batches = -1;
        end
        
        function [X, Y, step] = GetImage_(obj, step, train)                         
            X = zeros(obj.batch_size, obj.dims(1), obj.dims(2), 3);
            Y = zeros(obj.batch_size, 1000);
            from = ((step - 1) * obj.batch_size + 1);
            to = from + obj.batch_size - 1;
            for i = from : to
                name = sprintf('%s/ILSVRC2012_val_%s.JPEG', obj.file_pattern, sprintf('%08d', i));
                idx = i - from + 1;
                img = single(imread(name));
                if (size(img, 1) == obj.dims(1) && size(img, 2) == obj.dims(2))
                    X(idx, :, :, :) = img;
                else
                    X(idx, :, :, :) = permute(img(1:obj.dims(1), 1:obj.dims(2), :), [2, 1, 3]); %img(1:obj.dims(1), 1:obj.dims(2), :);
                    obj.meanX = obj.meanX(1:obj.dims(1), 1:obj.dims(2), :);
                end
                Y(idx, obj.Y(i)) = 1;
            end
            if (size(X, 2) == 221)
                X = single((X-118.380948) ./ 61.896913);
            else
                X = X - repmat(reshape(obj.meanX, [1, obj.dims(1), obj.dims(2), obj.dims(3)]), [obj.batch_size, 1, 1, 1]);
            end
            step = step + 1;
        end               
    end
end

function json = FillDefault(json)
json.type = 'RawImageInput';
end

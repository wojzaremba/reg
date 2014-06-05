classdef RawImageInputTrainImgNet < Input
    properties
        file_pattern
        meanX
        Y
        file_dir
        img_names
    end
    methods
        function obj = RawImageInputTrainImgNet(json)
            obj@Input(FillDefault(json));
            obj.file_pattern = json.file_pattern;
            tmp = load('~/data/imagenet/val_cropped/meta.mat'); % XXX : Hack, fix this loading later.
            obj.meanX = tmp.meanX;
            tmp = load('/misc/vlgscratch3/FergusGroup/denton/imagenet_data/batch_order.mat');
            obj.Y = tmp.batch_order.Y;
            obj.img_names = tmp.batch_order.img_names;
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
            from = (step - 1) * obj.batch_size + 1;
            to = from + obj.batch_size - 1;
            for i = from : to
                img_name = obj.img_names{i};
                folder_name = img_name(1:9);
                fname = sprintf('%s/%s/%s.JPEG', obj.file_pattern, folder_name, img_name); 
                idx = i - from + 1;
                try
                    img = single(imread(fname));
                    do_skip = 0;       
%                     [sy, sx, ~] = size(img);
%                     x1 = floor( (sx - 224) / 2);
%                     x2 = sx - 224 - x1;
%                     y1 = floor( (sy - 224) / 2);
%                     y2 = sy - 224 - y1;

                catch 
                    do_skip = 1;
                end
                while do_skip == 1 || size(img, 3) ~= 3 %|| sx < 224 || sy < 224 % Skip image is it is too small or is monochromatic
                    obj.img_names = obj.img_names([1:i-1, i+1:length(obj.img_names)]);
                    obj.Y = obj.Y([1:i-1, i+1:length(obj.Y)]);
                    img_name = obj.img_names{i};
                    folder_name = img_name(1:9);
                    fname = sprintf('%s/%s/%s.JPEG', obj.file_pattern, folder_name, img_name); 
                    idx = i - from + 1;
                    try
                        img = single(imread(fname));
                        do_skip = 0;                   
%                         [sy, sx, ~] = size(img);
%                         x1 = floor( (sx - 224) / 2);
%                         x2 = sx - 224 - x1;
%                         y1 = floor( (sy - 224) / 2);
%                         y2 = sy - 224 - y1;
                    catch 
                        do_skip = 1;
                    end
                end
                X(idx, :, :, :) = img;%(y1+1:sy - y2, x1+1:sx - x2, :); % Cut out center of image.
                Y(idx, obj.Y(i)) = 1;
                
            end
            X = X - repmat(reshape(obj.meanX, [1, obj.dims(1), obj.dims(2), obj.dims(3)]), [obj.batch_size, 1, 1, 1]);
            step = step + 1;
        end               
    end
end

function json = FillDefault(json)
json.type = 'RawImageInput';
end

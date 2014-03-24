global debug plan
debug = 0;
init();
fprintf('conv mono verif test\n');

dims = [96, 11, 11, 3];

jsons = {};
jsons{1} = struct('batch_size', 128, 'rows', 224, 'cols', 224, 'depth', 3, 'number_of_classes', 10, 'type', 'TestInput');
jsons{2} = struct('local_2d_patch', struct('patch_rows', 11, 'patch_cols', 11, 'stride_rows', 4, 'stride_cols', 4), ...
                  'on_gpu', 1, 'depth', 96, 'function', 'LINEAR', 'type', 'Conv');
jsons{3} = struct('function', 'LINEAR', 'rows', 1, 'cols', 1, 'depth', 10, 'type', 'FC', 'fully_connected', true);      
jsons{4} = struct('type', 'Softmax');
plan = Plan(jsons);    


num_image_colors = 4;
colors = randn([num_image_colors, dims(4)]);
dec = randn(dims(1), dims(4));
Wmono = randn([dims(1), dims(2), dims(3)]);
S = randn(dims(1), dims(2)*dims(3));
assignment = reshape(repmat(1:num_image_colors', [1, dims(1) / num_image_colors]), dims(1), 1);
W = MonochromaticInput.ReconstructW(colors, dec, S, assignment, [dims(1), dims(4), dims(2), dims(3)]);

C_(CopyToGPU, plan.layer{2}.gpu.vars.W, single(W));

S = Scheduler(struct('max_errors', 110, 'no_compilation', 0));
approx = MonochromaticInput(struct('suffix', '_test', 'on_gpu', 1),...
                            struct('num_image_colors', {num_image_colors}), ...
                            struct('origNumColors', {3}, ...
                                   'B_X', {32}, ...
                                   'B_Y', {6}, ...
                                   'imgsPerThread', {4}, ...
                                   'filtersPerThread', {4}, ...
                                   'colorsPerBlock', {1}, ...
                                   'scale', {0}, ...
                                   'checkImgBounds', {0}));     
                                 
S.Add(approx);
S.approx_logs{1}.ClearLog();
S.Run();
S.Printf();

% compiles cuda code for monochromatic filter approximation (with SVD) and tests for
% correctness.
global root_path
init();

if 1 % Just do this once.
    % replace #mock with, func_name 
    fid_read = fopen(strcat(root_path, '/gpu/src/Capprox_template.cu'), 'r');
    fid_write = fopen(strcat(root_path, '/gpu/src/Capprox_gen.cu'), 'wt');
    line = fgets(fid_read);
    while ischar(line)
        line = strrep(line, '#mock' , 'monochromatic_input'); 
        line = strrep(line, '\n', '\\n');
        line = strrep(line, '%', '%%');
        line = strrep(line, '\', '\\');
        fprintf(fid_write, line);
        line = fgets(fid_read);
    end
    fclose(fid_read);
    fclose(fid_write);
end          

%flags
do_compile = 1;

gids.X = 1;
gids.X_mono = 2;
gids.W = 3;
gids.W_mono = 4;
gids.out = 5;
gids.out_mono = 6;
gids.perm = 7;
gids.colors = 8;
gids.X_mono = 9;

numImages = 128;
stride = 2;
padding = 0;
K = 7;
numFilters = 96;
newNumColors = 24;

if do_compile
    % set cuda kernel vars
    cuda_vars.origNumColors = 3;
    cuda_vars.comb_BX = 64;
    cuda_vars.combImgPerThread = numImages / cuda_vars.comb_BX;
    cuda_vars.filtersPerThread = 4; % only relevant if newNumColors <= 4
    cuda_vars.B_Y = 8;
    cuda_vars.B_X = 32;
    cuda_vars.colorsPerBlock = 2;%filtersPerThread * B_Y / filtersPerColor;
    cuda_vars.imgsPerThread = 4; %numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    cuda_vars.scale = 0;
    cuda_vars.checkImgBounds = mod(numImages, cuda_vars.B_X*cuda_vars.imgsPerThread) ~= 0;

    % replace template variables, and compile
    fid_read = fopen(strcat(root_path, 'gpu/src/monochromatic_input_template.cuh'), 'r');
    fid_write = fopen(strcat(root_path, 'gpu/src/monochromatic_input_gen.cuh'), 'wt');
    line = fgets(fid_read);
    while ischar(line)
        fields = fieldnames(cuda_vars);
        for f = 1:length(fields)
           line = strrep(line, strcat('#', fields{f}), num2str(getfield(cuda_vars, fields{f}))); 
        end
        line = strrep(line, '\n', '\\n');
        line = strrep(line, '%', '%%');
        fprintf(fid_write, line);
        line = fgets(fid_read);
    end

    fclose(fid_read);
    fclose(fid_write);

    cd(strcat(root_path, 'gpu/'));
    status = system('make gen');
    cd(root_path);

end
    
% Check correctness
X = ones(numImages, 224, 224, 3);
Wtmp = randn(numFilters, K, K, 3);
num_colors = newNumColors;
[W_approx, W_mono, colors, perm]  = monochromatic_approx(Wtmp, num_colors);
out_ = single(zeros(numImages, 110, 110, numFilters));
out_mono_ = single(zeros(numImages, 110, 110, numFilters));
X_mono = zeros(numImages, 224, 224, newNumColors);

num_runs = 10;

%copy to GPU for mono conv
C_(CopyToGPU, gids.X,  single(X));
C_(CopyToGPU, gids.W_mono,  single(W_mono));
C_(CopyToGPU, gids.out_mono,  out_mono_);
C_(CopyToGPU, gids.perm,  single(perm - 1));
C_(CopyToGPU, gids.colors,  single(colors));
C_(CopyToGPU, gids.X_mono,  single(X_mono));

lapse2 = [];
for t=1:num_runs
    C_(StartTimer);
    C_(approx_pointer, gids.X, gids.W_mono, gids.out_mono, 224, newNumColors, K, stride, padding, gids.perm, gids.colors, gids.X_mono);
    lapse = C_(StopTimer); 
    lapse2 = [lapse2, lapse];
end

monoimages = reshape(C_(CopyFromGPU, gids.X_mono), size(X_mono));
out_mono = reshape(C_(CopyFromGPU, gids.out_mono), size(out_mono_));
C_(CleanGPU);

% copy to GPU for regular conv
C_(CopyToGPU, gids.W,  single(W_approx));
C_(CopyToGPU, gids.X,  single(X));
C_(CopyToGPU, gids.out,  out_);


lapse1 = [];
for t=1:num_runs
    C_(StartTimer);
    C_(ConvAct, gids.X, gids.W, gids.out, size(X, 2), size(X, 4), size(W_approx, 2), stride, padding);
    lapse = C_(StopTimer); 
    lapse1 = [lapse1, lapse];
end
out = reshape(C_(CopyFromGPU, gids.out), size(out_));
C_(CleanGPU);



% are results equal?
fprintf('norm(out_mono(:) - out(:)) / norm(out(:)) = %f\n', (norm(out_mono(:) - out(:)) / norm(out(:))));

speedup = lapse1 ./ lapse2;
fprintf('average speedup = %f\n', mean(speedup));
fprintf('std speedup = %f\n', std(speedup));

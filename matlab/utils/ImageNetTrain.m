out_dir = '/misc/vlgscratch3/FergusGroup/denton/imagenet_data/train_cropped/';
file_pattern = '/misc/vlgscratch3/data/imagenet12/original/train';
[status, cmdout] = system(sprintf('ls %s', file_pattern));
folders = strsplit(cmdout);
folders = folders(1:end-1);
assert(length(folders) == 1000);

nclass = 1000;
nimg = 1300;
% 
% for c = 116 : nclass
%     tic;
%     old_dir_name = sprintf('%s/%s', file_pattern, folders{c});
%     new_dir_name = sprintf('%s/%s', out_dir, folders{c});
%     [status, cmdout] = system(sprintf('mkdir %s', new_dir_name));
%     [status, cmdout] = system(sprintf('ls %s', old_dir_name));
%     img_names = strsplit(cmdout);
%     img_names = img_names(1:end-1);
%     for i = 1 : length(img_names)
%         img_name =  sprintf('%s/%s', old_dir_name, img_names{i});
%         try
%             img = imread(img_name);
%         catch
%             fprintf('Could not read image %s\n', img_name);
%         end
%         [sy, sx, ~] = size(img);
%         x1 = floor( (sx - 224) / 2);
%         x2 = sx - 224 - x1;
%         y1 = floor( (sy - 224) / 2);
%         y2 = sy - 224 - y1;
%         if sx < 224 || sy < 224
%             continue;
%         end
%         img = img(y1+1:sy - y2, x1+1:sx - x2, :);
%         imwrite(img, sprintf('%s/%s', new_dir_name, img_names{i}));
%     end
%     fprintf('Done class %d.\n', c);
%     toc;
%     fprintf('\n');  
% end

ntotal = 0;
pairs = [];
for i = 1 : nclass
    [status, cmdout] = system(sprintf('ls %s/%s | wc -l', out_dir, folders{i}));
    tot = str2num(cmdout);
    ntotal = ntotal + tot;
    fprintf('Total: %d   (%d)\n', ntotal, tot);
    for j = 1 : tot
       pairs(end+1, :) = [i, j]; 
    end
end

bs = 128;
nbatches = floor(ntotal / bs);
rp = randperm(ntotal);
pairs = [];
batches = zeros(nbatches, bs, 2);
for b = 1 : nbatches
    from = (b - 1) * bs + 1;
    to = b * bs;
    batches(b, :, :) = pairs(from:to, :);
    % XXX : Fill inwith pairs.
end


% 
% class_img_perm = randperm(nclass * nimg);
% i = 1;
% nbatches = floor(nimg * nclass / bs);
% for b = 1 : nbatches
%     tic;
%     [status, cmdout] = system(sprintf('mkdir %s/batch_%d/', out_dir, b));
%     ii=1;
%     fname = sprintf('%s/batch_%d/names.txt', out_dir, b);
%     fid = fopen(fname, 'w');
%     Y = [];
%     while ii <=128
%         if i > nclass * nimg
%             b = nbatches;
%             break;
%         end
%         class_num = mod(class_img_perm(i), nclass);
%         img_num = ceil(class_img_perm(i) / nclass);
%         dir_name = sprintf('%s/%s', file_pattern, file_dir(class_num).name);
%         img_dir = dir(dir_name);
%         while img_num > length(img_dir) - 2
%             i = i + 1;
%             if i > nclass * nimg
%                 b = nbatches;
%                 break;
%             end
%             class_num = mod(class_img_perm(i), nclass);
%             img_num = ceil(class_img_perm(i) / nclass);
%             dir_name = sprintf('%s/%s', file_pattern, file_dir(class_num).name);
%             img_dir = dir(dir_name);
%         end
%         img_name = img_dir(2 + img_num).name;
%         full_img_name = sprintf('%s/%s', dir_name, img_name); % Add 2 to skip '.', '..'
% %         img = single(imread(full_img_name));
% %         [sy, sx, ~] = size(img);
% %         x1 = floor( (sx - 224) / 2);
% %         x2 = sx - 224 - x1;
% %         y1 = floor( (sy - 224) / 2);
% %         y2 = sy - 224 - y1;
% %         if sx < 224 || sy < 224
%             
%         [status, cmdout] = system(sprintf('cp %s %s/batch_%d/%d_%d.JPEG', full_img_name, out_dir, b, b, i));
%         Y(ii) = find(label_map == str2num(file_dir(class_num).name(2:end)));
%         fprintf(fid, '%s\n', strrep(img_name, '.JPEG', ''));
%         i = i + 1;
%         ii = ii + 1;
%     end
%     fclose(fid);
%     save('Y.mat', 'Y');
%     fprintf('Done batch %d\n', b);
%     toc;
% end
% 

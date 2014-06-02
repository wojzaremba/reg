load /misc/vlgscratch3/data/imagenet12/original/ILSVRC2012_devkit_t12/data/meta.mat
map = containers.Map;
fname = '/misc/vlgscratch3/FergusGroup/denton/imagenet_data/labels.txt';
fid = fopen(fname, 'r');
line = fgetl(fid);
while ischar(line) 
    line = strsplit(line);
    cnum = str2num(line{1}) + 1;
    cname = line{2};
    map(cname) = cnum;
    line = fgetl(fid);
end

file_pattern = '/misc/vlgscratch3/data/imagenet12/original/train';% '/misc/vlgscratch3/FergusGroup/denton/imagenet_data/train_cropped'; %;
[status, cmdout] = system(sprintf('ls %s', file_pattern));
folders = strsplit(cmdout);
folders = folders(1:end-1);
assert(length(folders) == 1000);
nclass = 1000;
nimg = 1300;

names = {};
Y = [];
for c = 1 : nclass
    dir_name = sprintf('%s/%s', file_pattern,folders{c});
    [status, cmdout] = system(sprintf('ls %s', dir_name));
    img_names = strsplit(cmdout);
    img_names = img_names(1:end-1);
    for i = 1 : length(img_names)
       img_names{i} = img_names{i}(1:end-5); 
    end
    names(end+1 : end + length(img_names)) = img_names;
    Y(end+1 : end + length(img_names)) = map(folders{c}(2:end));
    fprintf('Done class %d.\n', c);
    fprintf('\n');
end

rp = randperm(length(Y));

batch_order.Y = Y(rp);
batch_order.img_names = names(rp);
save('/misc/vlgscratch3/FergusGroup/denton/imagenet_data/batch_order_notcropped.mat', 'batch_order');

label_map = importdata('/misc/vlgscratch3/FergusGroup/denton/imagenet_data/labels.txt');
label_map = label_map(:, 2);
load /misc/vlgscratch3/data/imagenet12/original/ILSVRC2012_devkit_t12/data/meta.mat;

load /misc/vlgscratch3/FergusGroup/denton/imagenet_data/batch_order.mat

for i = 1 : length(batch_order.Y)
   key = synsets(batch_order.Y(i)).WNID(2:end);
   true = find(label_map == str2num(key));
   batch_order.Y(i) = true;
   if true == 66
      v= 0; 
   end
   if mod(i, 1000) == 0
       fprintf('%d\n', i);
   end
end

save('/misc/vlgscratch3/FergusGroup/denton/imagenet_data/batch_order.mat', 'batch_order');

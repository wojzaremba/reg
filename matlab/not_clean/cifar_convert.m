
w = load('/Users/wojto/data/mnist/train');


data = cell(50000, 1);
for i = 1:5
    q = load(sprintf('/Users/wojto/data/cifar10/data_batch_%d.mat', i));
    for k = 1:10000
        X = reshape(q.data(k, :), [32, 32, 3]);
        Y = zeros(10, 1);
        Y(q.labels(k) + 1) = 1;
        X = (single(X) - 128) / 256;
        Y = single(Y);
        data{(i - 1) * 10000 + k} = struct('X', X, 'Y', Y);
    end
end

save('/Users/wojto/data/cifar10/train', 'data');

data = cell(10000, 1);
q = load('/Users/wojto/data/cifar10/test_batch.mat');
for k = 1:10000
    X = reshape(q.data(k, :), [32, 32, 3]);
    Y = zeros(10, 1);
    Y(q.labels(k) + 1) = 1;
    X = (single(X) - 128) / 256;
    Y = single(Y);    
    data{k} = struct('X', X, 'Y', Y);
end
save('/Users/wojto/data/cifar10/test', 'data');

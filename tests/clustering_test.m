randn('seed', 1);
C = {};
C{1} = [1, 1, 0]';
C{2} = [0, 1, 1]';
C{3} = [1, 0, 1]';
C{4} = [1, 1, 1]';

cluster_size = 24;
width = 7;
W = [];

for i = 1 : length(C)
   tmp = reshape(C{i} * randn(cluster_size * width ^ 2, 1)', [length(C{1}), cluster_size, width, width]);
   tmp = permute(tmp, [2, 3, 4, 1]);
   W = cat(1, W, tmp);
end

[Wapprox, Wmono, colors, perm, num_weights] = monochromatic_approx(W, 4);

assert(norm(Wapprox(:) - W(:)) < 1e-4);

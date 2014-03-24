fprintf('scale test\n');
A = single(randn(4, 5, 2));
C_(CopyToGPU, 1, A);
C_(Scale, 1, single(4.), 1);
Q = C_(CopyFromGPU, 1);
assert(norm(Q(:) - 4 * A(:)) < 1e-4);

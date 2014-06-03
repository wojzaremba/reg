function [cova, icova] = covroot(C)
    eps=5e-2;

    [U, S, V] = svd(C, 0);
    ssbis=diag(S)+eps;

    cova = U * diag(ssbis.^(1/2)) * V';
    icova = U * diag(ssbis.^(-1/2)) * V';
end
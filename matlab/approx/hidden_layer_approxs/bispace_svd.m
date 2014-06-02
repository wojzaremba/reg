function [Wapprox, C, Z, F, idx_input, idx_output] = bispace_svd(W, iclust, iratio, oclust, oratio, conseq, in_s, out_s)
    fprintf('iclust = %d, iratio = %f, oclust = %d, oratio = %f, conseq = %d\n', iclust, iratio, oclust, oratio, conseq);
    oclust_sz = size(W, 1) / oclust;
    iclust_sz = size(W, 4) / iclust;    
    
    odegree = floor(size(W, 1) * oratio / oclust);
    idegree = floor(size(W, 4) * iratio / iclust);

    orig_ops = out_s * out_s * prod(size(W));
    
    approx_ops = iclust * oclust * [in_s * in_s * iclust_sz * idegree, ...
                                    out_s * out_s * idegree * odegree * size(W, 2) * size(W, 3), ...
                                    out_s * out_s * odegree * oclust_sz];
    
    fprintf('Input rank: %d\n', idegree);
    fprintf('Output rank: %d\n', odegree);
    fprintf('Gain : %f\n', orig_ops / (sum(approx_ops)));
    fprintf('Transform 1 : %f\n', approx_ops(1) / sum(approx_ops));
    fprintf('Conv : %f\n', approx_ops(2) / sum(approx_ops));
    fprintf('Transform 3 : %f\n', approx_ops(3) / sum(approx_ops));
    if (~conseq)
        WW = W(:,:);
        idx_output = litekmeans(WW', oclust);
        WW = permute(W, [4 2 3 1]);
        WW = WW(:, :);
        idx_input = litekmeans(WW', iclust);    
    else
        for i = 1 : iclust
            idx_input(((i - 1) * iclust_sz + 1) : (i * iclust_sz)) = i;
        end
        for o = 1 : oclust
            idx_output(((o - 1) * oclust_sz + 1) : (o * oclust_sz)) = o;
        end                                
    end
    
    C = zeros(size(W, 4) / iclust, idegree, iclust, oclust);
    Z = zeros(odegree, size(W, 2), size(W, 3), idegree, iclust, oclust);
    F = zeros(size(W, 1) / oclust, odegree, iclust, oclust);
    
    for i = 1 : iclust
        for o = 1 : oclust
            oidx = idx_output == o;
            iidx = idx_input == i;
            
            Wtmp = W(oidx, :, :, iidx);                        
            [u, s, v] = svd(Wtmp(:, :));
            F_ = u(:, 1:odegree) * s(1:odegree, 1:odegree);

            Wtmptmp = F_ * v(:, 1:odegree)';
            F(:, :, i, o) = F_;
            Wapprox_tmp = reshape(v(:, 1:odegree)', [odegree, size(Wtmp, 2), size(Wtmp, 3), size(Wtmp, 4)]);

            Wapprox_tmp = permute(Wapprox_tmp, [4, 1, 2, 3]);
            [u, s, v] = svd(Wapprox_tmp(:, :));
            C_ = u(:, 1:idegree) * s(1:idegree, 1:idegree);
            C(:, :, i, o) = C_;
            Z_ = v(:, 1:idegree);
            Wtmptmptmp_ = C_ * Z_';            
            Z(:, :, :, :, i, o) = reshape(Z_, [odegree, size(W, 2), size(W, 3), idegree]);
            
%             if (i == 1) && (o == 1)
%                 fprintf('i = %d, o = %d\n', i, o);
%                 fprintf('target = %f\n', norm(Wtmp(:)));
%                 fprintf('temporary target = %f\n', norm(Wtmptmp(:)));
%                 fprintf('temporary temporary target = %f\n', norm(Wtmptmptmp_(:)));
%                 fprintf('initial = %f\n', norm(Z_(:)));
%                 fprintf('\n'); 
%             end
        end
    end

    fprintf('\n\n');
    Wapprox = zeros(size(W));
    for i = 1 : iclust
        for o = 1 : oclust
            oidx = idx_output == o;
            iidx = idx_input == i; 
            
            C_ = C(:, :, i, o);
            Z_ = Z(:, :, :, :, i, o);
            F_ = F(:, :, i, o);
            Z_ = permute(Z_, [4, 1, 2, 3]);
            Z_ = Z_(:, :)'; 
            ZC = Z_ * C_';
                         
            Wtmptmptmp = reshape(ZC, [odegree, size(W, 2), size(W, 3), iclust_sz]);
            Wtmptmptmp = Wtmptmptmp(:, :);
            Wtmp = F_ * Wtmptmptmp; 
            
            
            Wtmp = reshape(Wtmp, [oclust_sz, size(W, 2), size(W, 3), iclust_sz]);
            Wapprox(oidx, :, :, iidx) = Wtmp;   
            
%             if (i == 1) && (o == 1)            
%                 fprintf('i = %d, o = %d\n', i, o);
%                 fprintf('target = %f\n', norm(Wtmp(:)));
%                 fprintf('temporary target = %f\n', norm(Wtmptmp(:)));
%                 fprintf('temporary temporary target = %f\n', norm(Wtmptmptmp(:)));
%                 fprintf('initial = %f\n', norm(Z_(:)));
%                 fprintf('\n');
%             end
        end
    end

    fprintf('norm(W(:)) = %f\n', norm(W(:)));
    fprintf('norm(Wapprox(:)) = %f\n', norm(Wapprox(:)));
    fprintf('||W - Wapprox|| / ||W|| = %f\n', norm(W(:) - Wapprox(:)) / norm(W(:)));
end


function [Wapprox, F, C, XY, perm_in, perm_out, num_weights, idx_input, idx_output] = bisubspace_lowrank_approx_nosep(W, args)
% This approximation performs bi-clustering on input and output feature
% coordinates. After clustering, each kernel is then approximated by a sum
% of k rank one tensors. 
%
% W : dimensions (Fout, X, Y, Fin)
% args.iclust : Number of groups in input partition
% args.oclust : Number of groups in output partition
% args.k : Rank of tensor approximation (sum of k rank one tensors)

    original_complex = prod(size(W)) * args.out_s * args.out_s;
    iclust_sz = size(W, 4) / args.iclust;
    oclust_sz = size(W, 1) / args.oclust;
    target_complex = args.iclust * args.oclust * args.k * ...
                      [iclust_sz * args.in_s * args.in_s, ...
                       args.out_s * args.out_s * size(W, 2) * size(W, 3), ...
                       oclust_sz * args.out_s * args.out_s];
                   
    fprintf('Gain : %f \n', original_complex / sum(target_complex));
    fprintf('Transform 1 : %f \n', target_complex(1) / sum(target_complex));
    fprintf('Conv : %f \n', target_complex(2) / sum(target_complex));    
    fprintf('Transform 2 : %f \n', target_complex(3) / sum(target_complex));    
    

    iclust_sz = size(W, 4) / args.iclust;
    oclust_sz = size(W, 1) / args.oclust;
    num_weights = (iclust_sz + oclust_sz + size(W, 2) * size(W, 2)) * args.iclust * args.oclust * args.k;

    
   % Find partition of input and output coordinates.
    if (strcmp(args.cluster_type, 'subspace'))
        WW = W(:,:);
        keyboard;
        idx_output = subspace_cluster(WW', args.oclust);
        WW = permute(W, [4 2 3 1]);
        WW = WW(:, :);
        idx_input = subspace_cluster(WW', args.iclust);
    elseif (strcmp(args.cluster_type, 'kmeans'))
        MAXiter = 1000; % Maximum iteration for KMeans Algorithm
        REPlic = 100;
        WW = W(:,:);
        idx_output = litekmeans(WW', args.oclust);
        WW = permute(W, [4 2 3 1]);
        WW = WW(:, :);
        idx_input = litekmeans(WW', args.iclust);
    else
        assert(0);
    end
    
    [~, perm_in] = sort(idx_input);
    [~, perm_out] = sort(idx_output);
    
    rast=1;
        
    % Now compress each cluster.
    Wapprox = zeros(size(W));
    F = zeros([oclust_sz, args.k, args.iclust, args.oclust]);
    C = zeros([iclust_sz, args.k, args.iclust, args.oclust]);
    XY = zeros([size(W, 2)^2,args.k, args.iclust, args.oclust]);
    for i = 1 : args.oclust
        for j = 1 : args.iclust
            Io = find(idx_output == i);
            Ii = find(idx_input == j);
            chunk = W(Io, :, :, Ii);
            chunk = permute(chunk, [1 4 2 3]);
            chunk0 = reshape(chunk, size(chunk,1), size(chunk,2), size(chunk,3)*size(chunk,4));
            
            %Compute a low-rank approximation of the kernel.
            [f, c, xy, ~, cappr] = rankoneconv3D(chunk0, args.k);
            Wapprox(Io, :, :, Ii)=permute(reshape(cappr, size(chunk)),[1 3 4 2]);
            F(:, :, j, i) = f;
            C(:, :, j, i) = c;
            XY(:, :, j, i) = xy;
            assignment{rast}.Io = Io;
            assignment{rast}.Ii = Ii;
            rast = rast + 1;
        end
    end
    
end



% function [Wapprox_best, Fbest, Cbest, XYbest, perm_in_best, perm_out_best, num_weights] = bisubspace_lowrank_approx_nosep(W, args)
% % This approximation performs bi-clustering on input and output feature
% % coordinates. After clustering, each kernel is then approximated by a sum
% % of k rank one tensors. 
% %
% % W : dimensions (Fout, X, Y, Fin)
% % args.iclust : Number of groups in input partition
% % args.oclust : Number of groups in output partition
% % args.k : Rank of tensor approximation (sum of k rank one tensors)
%    
%     iclust_sz = size(W, 4) / args.iclust;
%     oclust_sz = size(W, 1) / args.oclust;
%     num_weights = (iclust_sz + oclust_sz + size(W, 2) * size(W, 2)) * args.iclust * args.oclust * args.k;
% 
%     best_recon_err = Inf;
%     niter = 20;
%     
%     for it = 1 : niter
%         fprintf('\t %d / %d %f\n', it, niter, best_recon_err);
%        % Find partition of input and output coordinates.
%         if (strcmp(args.cluster_type, 'subspace'))
%             WW = W(:,:);
%             keyboard;
%             idx_output = subspace_cluster(WW', args.oclust);
%             WW = permute(W, [4 2 3 1]);
%             WW = WW(:, :);
%             idx_input = subspace_cluster(WW', args.iclust);
%         elseif (strcmp(args.cluster_type, 'kmeans'))
%             s = RandStream('mt19937ar','Seed',it);
%             MAXiter = 1000; % Maximum iteration for KMeans Algorithm
%             REPlic = 100;
%             WW = W(:,:);
%             perm=randperm(s, size(W, 1));
%             idx_output = litekmeans(WW(perm, :)', args.oclust);
%             WW = permute(W, [4 2 3 1]);
%             WW = WW(:, :);
%             perm=randperm(s, size(W, 4));
%             idx_input = litekmeans(WW(perm, :)', args.iclust);
%         else
%             assert(0);
%         end
% 
%         [~, perm_in] = sort(idx_input);
%         [~, perm_out] = sort(idx_output);
% 
%         rast=1;
% 
%         % Now compress each cluster.
%         Wapprox = zeros(size(W));
%         F = zeros([oclust_sz, args.k, args.iclust, args.oclust]);
%         C = zeros([iclust_sz, args.k, args.iclust, args.oclust]);
%         XY = zeros([size(W, 2)^2,args.k, args.iclust, args.oclust]);
%         for i = 1 : args.oclust
%             for j = 1 : args.iclust
%                 Io = find(idx_output == i);
%                 Ii = find(idx_input == j);
%                 chunk = W(Io, :, :, Ii);
%                 chunk = permute(chunk, [1 4 2 3]);
%                 chunk0 = reshape(chunk, size(chunk,1), size(chunk,2), size(chunk,3)*size(chunk,4));
% 
%                 %Compute a low-rank approximation of the kernel.
%                 [f, c, xy, ~, cappr] = rankoneconv3D(chunk0, args.k);
%                 Wapprox(Io, :, :, Ii)=permute(reshape(cappr, size(chunk)),[1 3 4 2]);
%                 F(:, :, j, i) = f;
%                 C(:, :, j, i) = c;
%                 XY(:, :, j, i) = xy;
%                 assignment{rast}.Io = Io;
%                 assignment{rast}.Ii = Ii;
%                 rast = rast + 1;
%             end
%         end
%         
%         recon_err = norm(W(:) - Wapprox(:)) / norm(W(:));
%         if (recon_err < best_recon_err)
%             best_recon_err = recon_err;
%             Wapprox_best = Wapprox;
%             Fbest = F;
%             Cbest = C;
%             XYbest = XY;
%             perm_in_best = perm_in;
%             perm_out_best = perm_out;
%         end
%         
%     
%     end
%     
% end


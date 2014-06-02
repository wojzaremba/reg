function [Wapprox, V_output, V_input, U_output, U_input, num_weights] = bisubspace_lowrank_approx_kflats(W, args)

    original_complex = prod(size(W)) * args.in_s * args.in_s;
    iclust_sz = size(W, 4) / args.iclust;
    oclust_sz = size(W, 1) / args.oclust;
%     target_complex = args.iclust * args.oclust * args.k * ...
%                       [iclust_sz * args.in_s * args.in_s, ...
%                        args.out_s * args.out_s * size(W, 2) * size(W, 3), ...
%                        oclust_sz * args.out_s * args.out_s];
%                    
%     fprintf('Gain : %f \n', original_complex / sum(target_complex));
%     fprintf('Transform 1 : %f \n', target_complex(1) / sum(target_complex));
%     fprintf('Conv : %f \n', target_complex(2) / sum(target_complex));    
%     fprintf('Transform 2 : %f \n', target_complex(3) / sum(target_complex));    
    

    iclust_sz = size(W, 4) / args.iclust;
    oclust_sz = size(W, 1) / args.oclust;

    num_weights = (iclust_sz + oclust_sz + size(W, 2) * size(W, 2)) * args.iclust * args.oclust * args.k;

    %we will look for separable k-flats approximations of the tensor W
    WW=W(:,:);
    [idx_output, U_output, V_output] = litekflats(WW', args.oclust, round(args.rho_out*oclust_sz)) ;
    WW=permute(W, [4 2 3 1]);
    WW = WW(:,:);
    [idx_input, U_input, V_input] = litekflats(WW', args.iclust, round(args.rho_in*iclust_sz));
    
    %each bi-cluster output_cluster k and input_cluster l is approximated by a smaller 4-tensor. 
    %W_{k,l} approx C_k tilde{W}_{k,l} O_l where C_k is a C x rho*C matrix and O_l is a rho*O x O matrix. 
    
    Wapprox = 0*W;
    for i = 1 : args.oclust
            Io = find(idx_output == i);
            chunk = W(Io,:,:,:);
            chbis = chunk(:,:);
            tutu = U_output{i}*U_output{i}';
            chbis = chbis * tutu;
            chbis = reshape(chbis,size(chunk));
            Wapprox(Io,:,:,:) = chbis;
    end
    for j = 1 : args.iclust
            Ii = find(idx_input == j);
            chunk = Wapprox(:,:,:,Ii);
            chunk2 = permute(chunk, [4 2 3 1]);
            chbis = chunk2(:,:);
            tutu = U_input{j}*U_input{j}';
            chbis = chbis * tutu;
            chbis = reshape(chbis,size(chunk2));
            chbis = permute(chbis, [4 2 3 1]);
            Wapprox(:,:,:,Ii) = chbis;
    end
            

end




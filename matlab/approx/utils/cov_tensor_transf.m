function [Wout] = cov_tensor_transf(Win, Sigma)

    % W : dimensions (Fout, X, Y, Fin)

    tempo = Win(:,:);
    tempo = tempo * Sigma ; 
    Wout = reshape(tempo, size(Win));

end
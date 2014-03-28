function [value, Wgrad]=convbound(W)

W = permute(W,[1 4 2 3]);
S=size(W);
R0=S(3);
Ws=zeros(S(1),S(2),R0^2);
for i=1:S(1)
    for j=1:S(2)
    aux = dct2(squeeze(W(i,j,:,:)));
    Ws(i,j,:)=aux(:);
    end
end
for i=1:size(Ws,3)
la(i) = norm(squeeze(Ws(:,:,i)));
end
[value, pos] = max(la);
tmp = squeeze(Ws(:,:,pos));
[u,s,v]=svds(tmp,1);
template=zeros(R0);
template(pos)=1;
F=idct2(template);
gradW = tmp * v * v';
Wgrad = gradW(:)* F(:)';
Wgrad = reshape(Wgrad, S);
Wgrad = permute(Wgrad, [1 3 4 2]);







function [Wout]=convshrink(W,alpha)

W = permute(W,[1 4 2 3]);
Wout=W;
S=size(W);
R0=S(3);
Ws=zeros(S(1),S(2),R0^2);
for i=1:S(1)
    for j=1:S(2)
    aux = dct2(squeeze(W(i,j,:,:)));
    Ws(i,j,:)=aux(:);
    end
end
Wsbis=Ws;
for i=1:size(Ws,3)
[uu,ss,vv]=svd(squeeze(Ws(:,:,i)),0);
Wsbis(:,:,i)=uu*(ss.^alpha)*vv';
end
for i=1:S(1)
for j=1:S(2)
	aux=reshape(squeeze(Wsbis(i,j,:)),S(3),S(3));
	Wout(i,j,:,:)=idct2(aux);
end
end

Wout=permute(Wout,[1 3 4 2]);




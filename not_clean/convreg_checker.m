
%%let's check the gradient of the operator norm

W=randn(50,10,10,3);

[ref,Wgrad]=convbound(W);

I=randperm(numel(W));
M=400;
eps=1e-5;

for m=1:M
Wbis=W;
Wbis(I(m))=Wbis(I(m))+eps;
[refbis]=convbound(Wbis);
emp(m) = eps^(-1)*(refbis-ref);
ana(m) = Wgrad(I(m));

end

emp = emp / norm(emp);
ana = ana / norm(ana);

norm(emp-ana)/norm(emp)



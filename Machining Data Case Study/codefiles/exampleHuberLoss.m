% anonymous function for huberloss
err =  @(Y,T) Y-T;
huberloss = @(Y,T,W,d)sum(W.*((0.5*(abs(Y-T)<=d).*(Y-T).^2) + ((abs(Y-T)>d).*abs(Y-T)-0.5)))/sum(W);

Y
T

for i = 1:numel(ds)
    delta = ds(i);
    dHLdY(i) = (1/N).*sum((-(Y-T)<delta + -delta.*sign(Y-T)));
end
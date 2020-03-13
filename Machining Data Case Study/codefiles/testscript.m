A=rand(16,12);  %matrix of 16*16
[a b] = size(A);  % get the size of A =16*16
c=4;d=3;   % reshape it into 4*4 matrices
l=0;
for i=1:c:a-3
for j=1:d:b-3
C=A((i:i+3),(j:j+3));
eval(['out_' num2str(l) '=C']);
l=l+1;
end
end
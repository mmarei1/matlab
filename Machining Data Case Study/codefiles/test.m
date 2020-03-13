p = dlmread('p.txt');
q = dlmread('q.txt');

bg = opt_kernel_comb(p,q);
[sigmas,tmean,tvar,tcdf]=mmd_linear_combo(p,q,[bg],[1])
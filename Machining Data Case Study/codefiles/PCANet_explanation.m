% PCANet steps 
% 1) for each pixel in image, extract x patches of dimensions k1,k2
% 2) calculate the mean of the patches, subtract them from each patch
% and form matrix X = [Xbar_1, Xbar_2, Xbar_N] 
% 3) Implement PCA to minimize reconstruction error for each filter, by
% applying the following minimization operation
% min V ||X-VV^Tx||^2 s.t. V^T*V = I_L , which denotes the L1 principal 
% eigenvectors of XX^T
% 4) Form Wl PCA filters by mapping the eigenvectors v into a matrix
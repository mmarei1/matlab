% t-sne visualization of the predictor variables
load sortedLocalTable.mat
predictors = table2array(sortedLocalTable(:,4:8));
categs = unique(sortedLocalTable.Var10(:));

rng('default')
%data_sne1 = tsne(predictors,'Algorithm','exact','Distance','cosine');
subplot(2,2,1)
g1=gscatter(data_sne1(:,1),data_sne1(:,2),sortedLocalTable.Var10(:));
title('Cosine')

rng('default')
%data_sne2 = tsne(predictors,'Algorithm','exact','Distance','mahalanobis');
subplot(2,2,2)
g2 = gscatter(data_sne2(:,1),data_sne2(:,2),sortedLocalTable.Var10(:));
title('Mahalanobis')

rng('default')
%data_sne3 = tsne(predictors,'Algorithm','exact','Distance','chebychev');
subplot(2,2,3)
g3 = gscatter(data_sne3(:,1),data_sne3(:,2),sortedLocalTable.Var10(:));
title('Chebychev')

rng('default')
%data_sne4 = tsne(predictors,'Algorithm','exact','Distance','euclidean');
subplot(2,2,4)
g4 = gscatter(data_sne4(:,1),data_sne4(:,2),sortedLocalTable.Var10(:));
title('Euclidean')
%%
% visualize a 3D variant of the most promising t-SNE
rng default
[data_sne,sne_loss] = tsne(predictors,'Algorithm','exact','NumDimensions',3);
figure(4); clf reset;
v = double(sortedLocalTable.Var10(:));
vs = sparse(1:numel(v),v,ones(size(v)),numel(v),3);
c = full(vs);
scatter3(data_sne(:,1),data_sne(:,2),data_sne(:,3),data_sne(:,4),data_sne(:,5),data_sne(:,6));
title('3-D Embedding')


% 
% subplot(2,2,3)
% g5 = plot(0,0,0,0,0,0,0,0,0,0,0,0);
% axis off
% legend(g5,unique(categs))

figure(2); clf reset;
classDists = histogram(sortedLocalTable.Var10(:))


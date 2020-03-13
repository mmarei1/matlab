%% Load data as table
load('allFiles_Table.mat')
tf1 = allFiles_Table.Magnification == 100;
% Create first source domain datastore based on subset of images defined by tf1;
% Labels assigned based on variant
source1_ds = imageDatastore(allFiles_Table.FileName(tf1),"Labels",categorical(allFiles_Table.Variant(tf1),'Ordinal',0));
% change the customized image read function to a specific pre-processing
% function, which pre-crops the images based on Fast Marching algorithm
source1_ds.ReadFcn = @segmentAndCropImage_DS;
[s1_train,s1_val] = splitEachLabel(source1_ds,0.7,0.3);
% Create second source domain dataset based on categorical cutting lengths
% with ordinal definitions ignored
clabels_s2 = categorical(allFiles_Table.CuttingDistance(tf1),'Ordinal',0);
% Same datastore, different categorical labels
source2_ds = source1_ds;
source2_ds.Labels = clabels_s2;
[s2_train,s2_val] = splitEachLabel(source2_ds,0.7,0.3);
% Same datastore, discrete label categories
clabels_s3 = {'x0','x1','x2','x3','x4','x5','x6','xF'};
[clabels,cedges] = discretize(allFiles_Table.CuttingDistance(tf1),8,'categorical',clabels_s3);
source3_ds = source1_ds;
source3_ds.Labels = clabels;
[s3_train,s2_val] = splitEachLabel(source1_ds,0.7,0.3);
%%%

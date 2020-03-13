% MMD example on target domain dataset
%% Step 1: compute the activations of the target domain features based on pseudo-class labels

net_source = alexnet;

% feature layers of AlexNet which we wish to evaluate are fc7 and fc8
fc7_source = net_source.Layers(20);
fc8_source = net_source.Layers(23);

% the features in the source domain are the existing network weights
fc7_weights = net_source.Layers(20).Weights;
fc8_weights = net_source.Layers(23).Weights;

wfc7 = mat2gray(fc7_weights);
wfc8 = mat2gray(fc8_weights);

wfc7 = imresize(wfc7,5);
wfc8 = imresize(wfc8,5);

figure(1); clf reset;
montage(wfc7)

figure(2); clf reset;
montage(wfc8)


% the features in the target domain are the computed activations from the
% model on the training datastore
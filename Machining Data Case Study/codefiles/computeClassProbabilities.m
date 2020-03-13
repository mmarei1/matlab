% Source Domain labels information (ImageNet 2012 data)
numExamples_Source = 14197122;
% Alexnet ILSVRC2012 images = 1.2 million

numPoints_Source = 1.2E6;
numCategories_Source = numel(net1.Layers(25).Classes);
imageCategories_Source = 1:numCategories_Source;

examplesPerClass_Source = numPoints_Source/numCategories_Source;

% Target domain labels information (tool image dataset)
numExamples_Target = 327;
% ylabel data falls in a continuous range between 0 and 1
% To calculate the unique labels in this dataset we need to estimate the
% number of examples within a label bin, with bins at
% 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1
binBounds = [0:0.1:1.2];
figure(1); clf reset;
h1 = histogram(knownLabelData.ylabel,binBounds,'Normalization','probability');
examplesPerClass_Target = h1.BinCounts

probDist_Targets = fitdist(knownLabelData.ylabel,'half-normal','by',binBounds);
marginal_probDist_Source = numClasses_src/numPoints_src;
% we need to compute the source and target domain distributions
% for the feature vectors of the model

% features_Train_source = extractFeatures();
% features_Test_source = extractFeatures();

% features_Train_target = extractFeatures();
% features_Test_target = extractFeatures();

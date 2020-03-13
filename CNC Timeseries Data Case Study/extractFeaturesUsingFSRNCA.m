% Extract FSRNCA Features
% Helper script to extract FSRNCA feature set from available features
% NOTE: this script expects input data specified a
% xtrain: the training input predictor sequences
% ytrain: the training output target sequences
% xtest: the testing input predictor sequences
% ytest: the testing output target sequences
% DO NOT RUN separately

fprintf("Feature Selection Approach: FSRNCA \n------------------------------------------\n\n")
nca = fsrnca(xtrain,ytrain,'Verbose',1,'FitMethod','exact','Solver','lbfgs')
    % select individual featuree based on an arbitrary parameter value threshold
    vt = 0.1;
    allFeatures = 1:numFeatures;
    selectedFeatures = allFeatures(nca.FeatureWeights > vt)
    figure(1); clf reset;
    plot(nca.FeatureWeights,'ro');
    hold on;
    plot(selectedFeatures,nca.FeatureWeights(selectedFeatures),'bo')
    xlabel('Feature Index')
    ylabel('Feature Weight')
    grid on
    title(['FSRNCA: Selected Features with Weights > ',num2str(vt)],'FontSize',18);
    [featureValues, featureIndexes] = sort(nca.FeatureWeights(nca.FeatureWeights > 0.1),'descend');
    
    lossvalue = loss(nca,xtest,ytest);
    % steps to improve the performance of the feature selector for sequence data
    % 1. refit the model

    numObs = size(xtrain,1);
    nca2 = refit(nca,'FitMethod','exact','Verbose',1,'Lambda',0.5/numObs)
    lossvalue2 = loss(nca2,xtest,ytest)
    % 2. 
    % Comment out this section to remove feature selection implementation on the dataset
    XTrain_cell_fs = cell(numel(XTrain_cell),1);
    XTest_cell_fs = cell(numel(XTest_cell),1);
    for i = 1:numel(XTrain_cell)
        XTrain_cell_fs{i} = XTrain_cell{i}(selectedFeatures,:);
    end

    for i = 1:numel(XTest_cell)
        XTest_cell_fs{i} = XTest_cell{i}(selectedFeatures,:);
    end
    fprintf("FSRNCA Features Extracted: %d \n",numel(selectedFeatures))
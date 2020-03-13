function  [net,bl,assertionResult] = assertAndLoadModel(name,numClasses,lossName,lossVal,classWeights, classBias)
    fprintf("'LossName' Argument passed to Assert function: \n LossName: %s\n LossVal: %d \n",lossName,lossVal)
    assertionResult = '';
    weightsMatrix = [];
    biasVector = [];
    % cast classes array into a categorical array with ordinal flag set
    % false
    if strcmp(lossName, "cross")
        if ~iscategorical(numClasses) && (numClasses ~= 0)
            classCategs = categorical(cellstr(numClasses),'Ordinal',0);
            cc = classCategs
        elseif iscategorical(numClasses)
            cc = numClasses;
            fprintf("Already categorical\n")
        end
    numClasses = unique(cc);
    fprintf("Classes: %d \n",numel(numClasses));
    else
        fprintf("Loss is mse or compound. Output prediction is a regression value (no categories). \n");
        cc = 0;
        classWeights = 0;
        weightsMatrix = [];
        biasVector = [];
    end
    if numel(classWeights) == numel(numClasses)
        fprintf("Number of weights matches the number of available classes \n");
        weightsMatrix = diag(classWeights)*ones(numel(classWeights),128);
        biasVector = classBias;
    end
    % Specify base layer structure
    % Preceding layer is the "FC8" layer from the CNN 
    % the preceding layer is the network's FC8 equivalent.
    % idea: extract the activations of this layer after training,
    % then train with this variant
    bl = [...
            batchNormalizationLayer('Name','bn_final'),...
            reluLayer('Name','relu_head_1'),...
            fullyConnectedLayer(256,'Name','fc_downsampling_2'),...
            batchNormalizationLayer('Name','bn2_final'),...
            reluLayer('Name','relu_head_2'),...
            fullyConnectedLayer(128,'Name','fc_downsampling_3'),...
            batchNormalizationLayer('Name','bn3_final'),...
            sigmoidActivationLayer(1,'siglayer'),...
            fullyConnectedLayer(1,'Name','fcfinal'),...
            regressionLayer('Name','regOut_final')...
            ];    
    try
        if name == "alexnet"
            net = alexnet;
        elseif name == "resnet18"
            net = resnet18;
        elseif name == "resnet50"
            net = resnet50;
        elseif name == "resnet101"
            net = resnet101;
        elseif name == "inceptionv3"
            net = inceptionv3;
        elseif name == "squeezenet"
            net = squeezenet;
            bl_squeezenet = [...
                reluLayer('Name','relu_head_1'),...
                fullyConnectedLayer(256,'Name','fc_downsampling_2','WeightL2Factor',100,'BiasL2Factor',10),...
                batchNormalizationLayer('Name','bn_2_final'),...  
                reluLayer('Name','relu_head_2'),...
                fullyConnectedLayer(128,'Name','fc_downsampling_3','WeightL2Factor',50,'BiasL2Factor',5),...
                batchNormalizationLayer('Name','bn3_final'),...
                sigmoidActivationLayer(1,'sigmoidLayer'),...
                fullyConnectedLayer(1,'Name','fcfinal','WeightL2Factor',0.5,'BiasL2Factor',0.2),...
                regressionLayer('Name','regOut_final')...
                ];
            bl = bl_squeezenet;   
        else
            try
                net = eval(name);
            catch exc2;
                warning('Evaluating model name failed: Invalid model name');
            end
           warning('Either bad base layer stack implementation or base CNN %s not found',name);
           net = [];
           bl = [];
           assertionResult = 'model not found';
        end
        % swap final baselayer layers in case lossname is cross
        if strcmp(lossName, "cross")
            bl = bl(1:end-3);
            % calculate the number of outputs of the (new) final layer so
            % that a weight matrix can be constructed to match the output
            % of the classifier
            %numInputsToFC = bl(end).InputSize;
            % classWeights is specified as a vector of weights
            bl = [bl, ...
                reluLayer('Name','relu_final'),...
                fullyConnectedLayer(numel(numClasses),'Name','fc_numClasses','Weights',weightsMatrix,'Bias',biasVector),...
                softmaxLayer('Name','sm1'),...
                classificationLayer('Name','classOut','Classes',numClasses)];
            
        elseif strcmp(lossName, "wcross")
            bl = bl(1:end-2);
            % calculate the number of outputs of the (new) final layer so
            % that a weight matrix can be constructed to match the output
            % of the classifier
            %numInputsToFC = bl(end).InputSize;
            % classWeights is specified as a vector of weights
            bl = [bl, ...
                fullyConnectedLayer(numel(numClasses),'Name','fc_numClasses'),...
                softmaxLayer('Name','sm1'),...
                customWeightedClassificationLayer('Name','classOut','Classes',numClasses,'Weights',classWeights)]; 
        elseif strcmp(lossName,"mse")
            fprintf('%s \n',lossName);
            bl = bl(1:end);
        elseif strcmp(lossName,"compoundLoss")
            fprintf('%s \n',lossName);
            bl = bl(1:end-1);
            bl = [bl, compoundLossLayer(lossVal,'regout_MMD_MSE')];
        elseif strcmp(lossName,"none")
            bl = bl(1:end-1);
            %bl = [bl, fullyConnectedLayer(1,'Name','fc_final_computeSigmoidal')];
            fprintf("Note: must compute sigmoid outputs in the training function! \n");
        end
        assertionResult = 'complete';
    catch exc
       warning('Not a valid model. Try net = loadPretrainedCNN("alexnet") or help loadPretrainedCNN \n - EXC: %s ',exc.identifier);
       net = [];
       bl = [];
       assertionResult = getReport(exc);
    end
end
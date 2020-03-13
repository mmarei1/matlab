%% layer = customWeightedClassificaitonLayer("Name",name,"Classes",classes,"ClassWeights",classWeights)
% define layer that weights the classification output by the classification
% weights using a weighted cross-entropy loss function.
% 
% takes the following arguments:
%     name: layer name (should be unique to other layers in the network)
%     classes: the classes that the network is trained on
%     classWeights: a numClasses-by-1 matrix, specifying the class weights
% Created by Mohamed Marei, 2020
%%--------------------------------------------------------------------------------------------------
%%
classdef customWeightedClassificationLayer < nnet.layer.ClassificationLayer
               
    properties
        % Vector of weights corresponding to the classes in the training
        % data
        %Name
        %Classes
        ClassWeights
    end

    methods
        function layer = customWeightedClassificationLayer(nameKW, name, classesKW, classes, cwKW, classWeights)
            % layer = weightedClassificationLayer(classWeights) creates a
            % weighted cross entropy loss layer. classWeights is a row
            % vector of weights corresponding to the classes in the order
            % that they appear in the training data.
            % 
            % layer = weightedClassificationLayer(classWeights, name)
            % additionally specifies the layer name. 

            % Set class weights
            layer.ClassWeights = classWeights;

            % Set layer name
            if strcmp(nameKW,'Name')
                layer.Name = name;
            end
            if strcmp(classesKW,'Classes')
                layer.Classes = classes;
            end
            if strcmp(cwKW,'ClassWeights') 
                layer.ClassWeights = classWeights;
            end
            
            % Set layer description
            layer.Description = 'Weighted cross entropy';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the weighted cross
            % entropy loss between the predictions Y and the training
            % targets T.

            N = size(Y,4);
            Y = squeeze(Y);
            T = squeeze(T);
            W = layer.ClassWeights;
            if isempty(W)
                W = 1;
            end
            Wsum = sum(W);
            W = W./wsum;
            loss = -sum(W*(T.*log(Y)))/N;
        end
    end
end
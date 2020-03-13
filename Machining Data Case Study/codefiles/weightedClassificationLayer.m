classdef weightedClassificationLayer < nnet.layer.ClassificationLayer
               
    properties
        % Vector of weights corresponding to the classes in the training
        % data
        Name
        Classes
        ClassWeights
    end

    methods
        function layer = weightedClassificationLayer(nameKW, name, classesKW, classes, cwKW, classWeights)
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
            if strcmp(cwKW,'Classes') 
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
    
            loss = -sum(W*(T.*log(Y)))/N;
        end
    end
end
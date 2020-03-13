classdef preprocessedImageSatastore < matlab.io.Datastore & ...
        matlab.io.datastore.MiniBatchable & ...
        matlab.io.datastore.Shuffleable & ...
        matlab.io.datastore.PartitionableByIndex & ...
        matlab.io.datastore.BackgroundDispatchable
    
    properties
        MiniBatchSize
    end
    
    properties(SetAccess = protected)
        NumObservations
    end
    
    properties(Access = protected)
        
        CurrentFileIndex
        InputImds
        OutputImds
    end
    
    methods
        function ds = preprocessedImageDatastore(inputImds,outputImds, miniBatchSize)
            ds.InputImds = copy(inputImds);
            ds.OutputImds = copy(outputImds);
            ds.InputImds.Readsize = miniBatchSize;
            ds.OutputImds.ReadSize = miniBatchSize;
            ds.NumObservations = length(inputImds.Files);
            ds.MiniBatchSize = miniBatchSize;
            ds.CurrentFileIndex = 1;
        end
        
        function dsnew = shuffle(ds)
            dsnew = copy(ds);
            shuffledIndexOrder = randperm(ds.NumObservations);
            dsnew.InputImds.Files = dsnew.InputImds.Files(shuffledIndexOrder);
            dsnew.OutputImds.Files = dsnew.OutputImds.Files(shuffledIndexOrder);
        end
        
        function dsnew = partitionByIndex(ds,indices)
            dsnew = copy(ds);
            dsnew.InputImds.Files = dsnew.InputImds.Files(indices);
            dsnew.OutputImds.Files = dsnew.OutputImds.Files(indices);
        end

        function tf = hasdata(ds)
            % Return true if more data is available
            tf = hasdata(ds.InputImds);
        end

        function [data,info] = read(ds)            
            % Read one batch of data
            inputImageData = read(ds.InputImds);
            outputImageData = read(ds.OutputImds);
            data = table(inputImageData,outputImageData);
            info.batchSize = size(data,1);
            ds.CurrentFileIndex = ds.CurrentFileIndex + info.batchSize;
            info.currentFileIndex = ds.CurrentFileIndex;  
        end

         function [data,info] = readByIndex(ds,indices)
            inputImdsNew = copy(ds.InputImds);
            outputImdsNew = copy(ds.OutputImds);
            inputImdsNew.Files = inputImdsNew.Files(indices);
            outputImdsNew.Files = outputImdsNew.Files(indices);
            X = readall(inputImdsNew);
            Y = readall(outputImdsNew);

            data = table(X,Y);
            info.CurrentReadIndices = indices;
        end
        
        function reset(ds)
            % Reset to the start of the data
            reset(ds.InputImds);
            reset(ds.OutputImds);
            ds.CurrentFileIndex = 1;
        end
        
    end 

    methods (Hidden = true)

        function frac = progress(ds)
            % Determine percentage of data read from datastore
            frac = (ds.CurrentFileIndex-1)/ds.NumObservations;
        end
    end
end
%            idxFC = findCNNFCLayers(net)
            % if the network has more than 2 fc layers, we take the
            % last 2 of those (i.e. fc 7 or fc8)
%             if numel(idxFC) > 1
%                 i1 = idxFC(1);
%                 i2 = idxFC(end-1);
%                 i3 = idxFC(end);
%                 fprintf("Indexes of First FC layer: %d \n",i1);
%                 fprintf("Indexes of Second FC layer: %d \n",i2);
%                 fprintf("Indexes of Last FC layer: %d \n",i3);
% 
%                 %lgraph(i2).WeightsInitializer = "zeros";
%                 layersBefore(i2).WeightsInitializer = "zeros"
%                 layersBefore(i3).WeightsInitializer = "zeros"
%                 fprintf("Weights Initializer for fully connected layers %s and %s set to zeros. \n",lgraph.Layers(i2).Name,lgraph(i3).Name);
%             elseif numel(idxFC) == 1
%                 layersBefore(idxFC).WeightsInitializer = "zeros";
%                 fprintf("Weights Initializer for fully connected layer %d set to zeros. \n",idxFC);
%             else
%                 idx = numel(lgraph)-3;
%                 layersBefore(idx).WeightsInitializer = "zeros";
%                 fprintf("Weights Initializer for fully connected layer equivalent %d set to zeros. \n",idx);
%             end
            
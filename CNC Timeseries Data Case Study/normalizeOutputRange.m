% [Out] = normalizeOutputRange(In,In_min,In_max) normalizes cell data by finding
% the minimum and maximum values across the entire input range In. 
% The normalization is performed according to the following operation:
%           Out = (In - In_min) ./(In_max - In_min)
% where In is the cell array being normalized, In_min and In_max are
% calculated from In.
% 
% The inputs to this function are:
% In - the cell array being normalized
% min - the minimum value of Out in the normalized scale (default is 0)
% max - the maximum value of Out in the normalized scale (default is 1)
% 
% If the min and max values are specified, the function uses those values
% instead.
%
% [Out, ymin, ymax] = normalizeOutputRange(In,...) additionally outputs
% the minimum and maximum values calculated from In.
%
% Created by Mohamed Marei (C), on 25/08/2019
% 
function [ydata_n, ymin, ymax, y_mat] = normalizeOutputRange(ydata,omin,omax)
    % convert entire cell array to double and obtain its min and max
    % y_mat1 = [ydata{:}];
    y_mat1 = cell2mat(horzcat(ydata{:}));
    ymin = min(y_mat1);
    ymax = max(y_mat1);
    
    y_mat = normalize(y_mat1,'range',[omin,omax]);
    
    startIndex = 1;
    for i = 1:numel(ydata)
        % create a sequence temporary variable to store extracted elements
        % from the ith cell
        sl = length(ydata{i});
        endIndex = startIndex+sl-1;
        tempSeq = y_mat(startIndex:endIndex);
        tmp{i} = tempSeq;
        startIndex = endIndex+1;
        ydata_n{i} = tmp{i};
    end
end
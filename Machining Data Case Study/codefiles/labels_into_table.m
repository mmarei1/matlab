load labelsTable.mat;
load newDataTable.mat;

for i=1:size(newDataTable,1)
    % check what experiment we're on
    [e,xl,yl] = retrieveValues(newDataTable,i,distances,wear_values);
    experiments(i) = e;
    xlabels(i) = xl;
    ylabels(i) = yl;
    id(i) = i;
end

% write a new table from the previous table
updatedDataTable = table(id',experiments',table2array(newDataTable(:,"filename")),xlabels',ylabels','VariableNames',{'id','exp','filename','xlabel','ylabel'});

% interpolate between missing values to find adequate estimates
% for j = 1:25
%     % sort x values for the x_labels in each experiment
%     
%     subtable = sortrows(updatedDataTable(updatedDataTable.exp== j,:));
%     
%     % missing y value index
%     missingValueIndex = subtable.ylabel == 0;
%     
%     % known x and y values
%     known_xValues = table2array(subtable.ylabel ~= 0,"xlabel");
%     known_yValues = table2array(subtable.ylabel ~= 0,"ylabel");
%     
%     
%     x_for_missingYValues = table2array(subtable(missingValueIndex,"xlabel"));
%     
%     estimated_yValues = interp1(known_xValues,known_yValues,x_for_missingYValues);
%     
%     e_y_values{i} = estimated_yValues;
% end


function [exp_count,xlabel,ylabel] = retrieveValues(newDataTable,tableIndex,distances,wear_values)
    currentFilename = table2array(newDataTable(tableIndex,"filename"));
    exp_count = str2double(extractBetween(currentFilename,"A","-W"));
    
    % find the matching xlabel from the 
    xlabel = abs(str2double(extractBetween(currentFilename,"-W","M[100]")));
    
    % find the corresponding yLabel for the given exp_count and xlabel
    arrayOfDistances = distances{exp_count};
    
    indexOfxLabelDistance = find(xlabel == arrayOfDistances);
    
    arrayOfyLabelValues = wear_values{exp_count};
    
    ylabel = arrayOfyLabelValues(indexOfxLabelDistance);
    
    % if ylabel is empty, assign value zero
    if isempty(ylabel)
        ylabel = 0;
    end
end
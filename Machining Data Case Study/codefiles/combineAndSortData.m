% script to output a new table sorted by experiment number and the length
% of wear
load cuttingParametersTable.mat
load localTable.mat
startIndex = 1;
endIndex = 0;
%classLabels = categorical([1:6],{'H1','H2','H3','H4','H5','H6'});
speeds = reshape(repmat(cuttingSpeeds,5,1),25,1);
sortedLocalTable = table('Size',[553,9],'VariableTypes',{'double','double','string','double','double','double','double','double','double'},'VariableNames',{'id','exp','filename','vc','fd','ae','rrc','xlabel','ylabel'});
for i = 1:numel(unique(localTable.exp))
    % set the current experiment to i
    currentExp = i;
    % take all the sub-experiments under i to be the new experiments
    expRows = localTable.exp == currentExp;
    subTable = localTable(expRows,:);
    numRows = height(subTable);
    %subTable = localTable(expRows,:);
    %oldIDs = localTable.id(expRows);
    newSubtable = sortrows(subTable,'xlabel');
    sprintf('exp: %d,  numRows: %d', currentExp,numRows)
    endIndex = startIndex+numRows-1;
    newIDs = startIndex:endIndex;
    ae = cuttingParametersTable.CuttingWidth(i)*ones(numel(newIDs),1);
    fd = cuttingParametersTable.feedRate(i)*ones(numel(newIDs),1);
    rrc = cuttingParametersTable.rrcTool(i)*ones(numel(newIDs),1);
    vc = speeds(currentExp)*ones(numel(newIDs),1);
% assign parameters individually
    sidx = newIDs(1)-startIndex+1;
    sortedLocalTable.id(newIDs) = newIDs;
    sortedLocalTable.exp(newIDs) = currentExp*ones(numRows,1);
    sortedLocalTable.filename(newIDs) = newSubtable.filename(:);
    sortedLocalTable.vc(newIDs) = vc;
    sortedLocalTable.fd(newIDs) = fd;
    sortedLocalTable.ae(newIDs) = ae;
    sortedLocalTable.rrc(newIDs) = rrc;
    sortedLocalTable.xlabel(newIDs) = newSubtable.xlabel(:);
    sortedLocalTable.ylabel(newIDs) = newSubtable.ylabel(:);
    
    startIndex = endIndex+1;
end
%%
% iterate over entire array and add class labels
for i = 1:height(sortedLocalTable)
    x = sortedLocalTable.xlabel(i);
    y = sortedLocalTable.ylabel(i);
    if x == 0
       class(i) = 1; 
    elseif x > 0 && x <= 10
       class(i) = 2;
    elseif x > 10 && x <= 40 && y < 0.4
       class(i) = 3;
    elseif x > 40 && x <= 100 && y < 0.4
       class(i) = 4;
    elseif x > 100 && x <= 160 && y <= 0.6
       class(i) = 5;
    else
       class(i) = 6;
    end 
end
class
categs = categorical(class,[1:6],{'Fresh','Break-in','Early Wear','Intermediate Wear','Damaged','Fully Worn'},'Ordinal',1);
% save the resulting table to use in our prediction framework
%save('sortedLocalTable.mat','sortedLocalTable');
sortedLocalTable = addvars(sortedLocalTable,categs','After','ylabel');
%localTable = sortedLocalTable;
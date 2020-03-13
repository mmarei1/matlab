% outCell = mapMatToCellArray(inArr,numRows,lengthVec) returns a 1xD
% output cell array with M-by-N cells, where D = numel(lengthVec),
% M is the number of rows specified by the scalar numRows and N is the 
% Dth value of lengthVec 
% 
% Example Usage:
% 
% Reshape an input array of size 5×21 into a cell array of 5 rows and a
% decreasing number of colums specified by the sequence [6 5 4 3 2 1]
% 
%         inArr = magic(5,21);
%         numRows = size(inArr,1);
%         lengthVec = 6:-1:1;
%         outCell = mapMatToCellArray(inArr, numRows,lengthVec)
% 
% The output result is:
%
%   outCell =
%
%       1×6 cell array
%
%       Columns 1 through 4
%
%           {5×6 double}   {5×5 double}   {5×4 double}   {5×3 double}
%
%       Columns 5 through 6
%
%           {5×2 double}   {5×1 double}
%
% Created by Mohamed Marei, on 30/08/2019

function outCell = mapMatToCellArray(inArr,numFeatures,lengthVec)
startIndex = 1;
    for i = 1:numel(lengthVec)
        endIndex = startIndex+lengthVec(i)-1;
        outCell{i} = inArr(1:numFeatures,startIndex:endIndex);
        startIndex = endIndex+1;
    end
end
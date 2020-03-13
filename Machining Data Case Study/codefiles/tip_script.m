% use AlexNet for regression?

% Resize the images to 227x227x3 images for training purposes
fpath = 'C:\Users\Mareim\OneDrive - Coventry University\NewImages\Edge';
imds = imageDatastore(fpath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
inputSize = [227 227 3];

% subset of the data which has the prelabeled wear values
prelabeledData = updatedDataTable(updatedDataTable.ylabel>0,:);

imds = imageDatastore(prelabeledData.filename(:),'Labels',prelabeledData.xlabel(:));

net = alexnet;
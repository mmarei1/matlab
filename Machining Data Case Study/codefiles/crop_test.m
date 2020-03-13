% Script to test cropping images
filepath = 'C:\Users\mareim\Documents\MATLAB\NewSandbox\Machining Data Case Study\NewImages\Edge\Healthy\';
basename = 'A7-W-80M[100]-1.jpg';
imageName = [filepath basename];
% maskName = [filepath 'A25-W-20M[100]-1.jpg'];
I = imread(imageName);

% crop each image
[croppedImage, bbA,bbN] = cropMyImage(I);
bbA
bbN
% show the results of each crop
%CI = imread(croppedImage);
imshow(croppedImage); title('Batch crop image')
text(400,50,sprintf('Dimensions: %3g by %3g',bbN(3), bbN(4)));
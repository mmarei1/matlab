function [outImage,BW,rect_coords]=segmentAndCropFMM(inputImage,vOffset)
%% Segment Image Using Fast Marching Method Algorithm
% Segment an object in an image using Fast Marching Method based on 
% differences in grayscale intensity as compared to the seed locations.
% 
% Read image.
if isempty(vOffset)
    vOffset = 150;
end
I = inputImage;
ImgColor =I;
%imshow(I)
%title('Original Image')
%% 
% Create mask and specify seed location. You can also use |roipoly| to create 
% the mask interactively.
I = rgb2gray(I);
mask = false(size(I)); 
mask(1,1) = true;
%% 
% Compute the weight array based on grayscale intensity differences.
W = graydiffweight(I, mask, 'GrayDifferenceCutoff', 25);
%% 
% Segment the image using the weights.

thresh = 0.01;
[BW, D] = imsegfmm(W, mask, thresh);
%figure(2)
%BW %=~BW;
%imshow(BW)
%title('Segmented Image')
%% 
% You can threshold the geodesic distance matrix |D| using different thresholds 
% to get different segmentation results.
% Invert the mask selection by passing ~BW as input
imageMask = bwareafilt(~BW,1);
 % measure the bounding box surrounding the foreground, i.e. the region
 % of interest
labeledImage = logical(imageMask);
measurements = regionprops(labeledImage,'BoundingBox');
% extrema = regionprops(labeledImage,'Extrema')
% imageOut = regionprops(labeledImage,'Image');
% figure(5); 
% imshow(imageOut.Image)
bbA = measurements.BoundingBox;
hStart = bbA(1);
vStart = bbA(2);
hHeight = 800;
hWidth = 800;

%     if bbA(2) >= 300
%         vShift = 0;
%         hShift = 0;
%         vPad = 0;
%         hPad = 0;
%     end
    
    
%%
%vOffset = 150; % number of pixels to lower the vertical start point
bbN = [hStart, vStart-vOffset, hWidth, hHeight];
outImage = imcrop(ImgColor,bbN);
%BW = imcrop(BW,bbN);
rect_coords = bbN;
%figure(2); 
%imshow(outCrop);
%     hold on;
%     rectangle('Position',rect_coords,'EdgeColor','red');
end
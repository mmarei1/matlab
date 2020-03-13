function [croppedImage, bbA, bbN] = cropMyImage(I)
    % implement sobel edge detection with a threshold of 0.015
    edgeMask = edge(rgb2gray(I),'sobel',0.015);
    
    % mask the image with the edge mask
    imageMask = bwareafilt(edgeMask,1);
    
    % measure the bounding box surrounding the foreground, i.e. the region
    % of interest
    labeledImage = bwlabel(imageMask);
    measurements = regionprops(labeledImage,'BoundingBox');
    % pad the bounding box slightly to reduce the likelihood of cropping
    % out too much of the image
%     hShift = -40;
%     vShift = 140;
%     hPad = 0;
%     vPad = -160;
   
    
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
    
    bbN = [hStart, vStart-200, hWidth, hHeight];
    
    croppedImage = imcrop(I,bbN);
end


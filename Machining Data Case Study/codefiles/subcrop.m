% function to help subdivide pic into m x n square images of size SIZE, 
% where m = size(h,2) and n=size(w,2), and SIZE is a scalar value. 
% h and w are 1-D coordinate vectors specifying the indexes of the vertical 
% and horizontal start points for the cropping operations 
% The output of subcrop is an image cell array of the cropped subimages
% 
% Example:
%
% Crop a 1600 x 1200 image into 12 sub-images of size 400 x 400:
%
%   imfile = 'A8-W-0.2M[100]-11.jpg';
%   im = imread(imfile);
% 
%   h = [1 401 801];
%   w = [1 401 801 1201];
%   SIZE = 400;
% 
%   cw = subcrop(im,h,w,SIZE);
function cropWindows = subcrop(im,h,w,SIZE)

counter = 1;
for i = 1:size(w,2)
    % for each gridcol of the image
    x = w(i);
    for j = 1:size(h,2)
        y = h(j);
        windowToCrop = [x,y,SIZE-1,SIZE-1];
        c = imcrop(im,windowToCrop);
        cropWindows{counter} = c;
        counter = counter+1;
    end
end
end
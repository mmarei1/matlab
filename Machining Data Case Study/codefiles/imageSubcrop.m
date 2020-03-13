% divide an image into blocks
clear all; close all; clc;

imfile = 'A8-W-0.2M[100]-11.jpg';
im = imread(imfile);

SIZE = 200;

% crop image to focus on internal area
[croppedImage, bbA, bbN] = cropMyImage(im);
figure(1); clf reset
imshow(croppedImage)
title(sprintf("Cropped image, %3g by %3g pixels",bbN(3),bbN(4)))

hn = 1:SIZE:size(croppedImage,1);
wn = 1:SIZE:size(croppedImage,2);

% h = [1 401 801];
% w = [1 401 801 1201];
counter = 1;
imNew = croppedImage;

cw = subcrop(imNew,hn,wn,SIZE);

% plot the results of the cw operation
% figure(2); clf reset; 
% imshow(imtile(cw));
% 
% figure(3); clf reset;
% subplot(1,2,1);
% imshow(im);
% subplot(1,2,2);
% imshow(imNew);

meansArray = [];
imrows = 3;
for i = 1:size(cw,2)
    if (size(cw{i},1) == SIZE) && (size(cw{i},2)== SIZE)
        sq{i} = cw{i};
    end
end

r = [1,2,3;4,5,6;7,8,9];

% for i=1:size(r,1)
%     % tile the first three images into each row
%   f = r(i,1);
%   s = r(i,2);
%   t = r(i,3);
%   imrect{i}= imtile([sq{r(f)},sq{r(s)},sq{r(t)}],'GridSize',[1,1]);
% end

figure(4); clf reset
montage(sq,'Indices',[1,2,3;5,6,7;9,10,11],'BackgroundColor','red','BorderSize',[2 2])
% return the highest mean value images

% until none of the subplots are left
% while totalsubs <= 2*size(cw,2)-1
%     totalsubs = totalsubs+1;
%     cw_h = rgb2hsv(cw{i});
%     subplot(size(hn,2),size(wn,2),totalsubs,'align');
%     imshow(cw{i});
%     totalsubs = totalsubs+1;
%     subplot(6,4,totalsubs);
%     imhist(cw_h);
%     i = i+1;
% end
% if image is mostly black, discard
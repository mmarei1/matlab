% load image files to read
load('/home/mareim/Downloads/cutting_tool_images/allFiles_Table.mat')
p1 = string(allFiles_Table.FileName(1));
p2 = string(allFiles_Table.FileName(2));
p3 = string(allFiles_Table.FileName(3));

testIm1 = imread(p1);
testIm2 = imread(p2);
testIm3 = imread(p3);
%%
% crop outputs
[bwSC1,outCropSC1,rc1] = segmentAndCropImage(testIm1,'fast',0.0131,10);
[bwSC2,outCropSC2,rc2] = segmentAndCropImage(testIm2,'fast',0.0131,10);
[bwSC3,outCropSC3,rc3] = segmentAndCropImage(testIm3,'fast',0.0131,10);
% plot images
figure(1);subplot(3,3,1);
% test image 1
figure(1); subplot(3,3,1);
imshow(testIm1);
title(extractAfter(p1,'images/'));
hold on; rectangle('Position',rc1,'EdgeColor','r','LineWidth',2);
hold off;
figure(1); subplot(3,3,4);
imshow(outCropSC1);
% test image 2
figure(1); subplot(3,3,2);
imshow(testIm2);
title(extractAfter(p2,'images/'));
hold on; rectangle('Position',rc2,'EdgeColor','r','LineWidth',2);
hold off;
figure(1); subplot(3,3,5);
imshow(outCropSC2);
% test image 3
figure(1); subplot(3,3,3);
imshow(testIm3);
title(extractAfter(p3,'images/'));
hold on; rectangle('Position',rc3,'EdgeColor','r','LineWidth',2);
hold off;
figure(1); subplot(3,3,6);
imshow(outCropSC3);
figure(1); subplot(3,3,7);
imshow(bwSC1);
figure(1); subplot(3,3,8);
imshow(bwSC2);
figure(1); subplot(3,3,9);
imshow(bwSC3);
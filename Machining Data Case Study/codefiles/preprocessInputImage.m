function imgOut = preprocessInputImage(imgIn,meanImage)
%A1 = imread("A7-W-80M[100]-1.jpg");
imgIn = double(imgIn);
    imsz = size(imgIn);
    timgOut = zeros(imsz);
% isolate each channel output
for i=1:3
    imgChannel_mean(i) = mean(meanImage(:,:,i),'all');
    imgChannel_stdv(i) = std(meanImage(:,:,i),0,[1,2]);  
    timgOut(:,:,i) = (imgIn(:,:,i) - imgChannel_mean(i)).*(1/imgChannel_stdv(i));
end
% compute mean and standard deviation of each channel output
    imgOut = timgOut;
end
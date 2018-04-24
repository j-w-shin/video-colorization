function [img SURF] = ComputeSURF(imgName, ratio)
%% Convert the image into Lab Color Space.
img = im2double(imread(imgName));
img = imresize(img, ratio);

[r c d] = size(img);
if d == 3
    img = rgb2lab(img);
else
    img = cat(3, img, img, img);
    img = rgb2lab(img);
end

%% Compute the SIFT features.
bUpright = true;
bExtended = true;
verbose = false;
% Create Integral Image
iimg=IntegralImage_IntegralImage(img(:, :, 1));
[row col] = size(img(:, :, 1));
SURF = zeros(row, col, 128);
for I=1:row
%     clc; disp([num2str(I) '/' num2str(row)]);
    for J=1:col
        ip.x = J;
        ip.y = I;
        ip.scale = 1;
        SURF(I, J, :) =SurfDescriptor_GetDescriptor(ip, bUpright, bExtended, iimg, verbose);
    end
end
return;
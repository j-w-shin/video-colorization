% This function is written to compute the SIFT features for the input
% image. It resize the image first and then computes the SIFT features.

function [img SIFT] = ComputeSIFT(imgName, ratio)

%% SIFT parameters
patchsize=8;
gridspacing=1;

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
SIFT =dense_sift(img(:, :, 1), patchsize, gridspacing);

%% Remove the border area of the image.
img = img(4:r-3, 4:c-3, :);

return;
function TransferColor
clc;
colImg = imresize(imread('input\island.png'), 0.4);
inpImg = imresize(imread('input\grayisland.png'), 0.4);

% colImg = imread('col.jpg');
% inpImg = imread('gray.jpg');
% 
% colImg = imread('01.png');
% inpImg = imread('02.png');
% colImg = imresize(colImg, 0.5);
% inpImg = imresize(inpImg, 0.5);
colImg = rgb2lab(colImg);
tmpImg = colImg(:,:,1)/100;
inpImg = cat(3, inpImg, inpImg, inpImg);
inpImg = rgb2lab(inpImg);

[r1, c1, d1] = size(inpImg);
outImg = zeros(r1, c1, 3);
outImg(:,:,1) = inpImg(:,:,1);

% inpImg(:,:,1) = im2double(inpImg(:,:,1));
% tmpImg = im2double(tmpImg);

N = ones(7);
stdInp = stdfilt(inpImg(:,:,1)/100, N);
stdCol = stdfilt(tmpImg, N);

stdInp = (stdInp * 0.1 + (inpImg(:,:,1)/100) * 0.9);
stdCol = (stdCol * 0.1 + tmpImg * 0.9);


for I=1:r1
    for J=1:c1
        [r, c] = findMatch(stdInp(I, J), stdCol);
        outImg(I, J, 2) = colImg(r, c, 2);
        outImg(I, J, 3) = colImg(r, c, 3);
    end
    disp(I);
end
outImg = lab2rgb(outImg);
% outImg(:,:,1)=medfilt2(outImg(:,:,1));
% outImg(:,:,2)=medfilt2(outImg(:,:,2));
% outImg(:,:,3)=medfilt2(outImg(:,:,3));
imwrite(outImg, 'transferColor.png');
imshow(outImg);
%lab2double(applycform(rgb, makecform('srgb2lab')));
return;

function [r, c] = findMatch(v, outImg)

r = 1;
c = 1;
d = 100;
[r2, c2] = size(outImg);
diff = abs(outImg - v);
for I = 1:r2
    for J = 1:c2
        if diff(I, J) <= d
            d = diff(I, J);
            r = I;
            c = J;
        end
    end
end

return;
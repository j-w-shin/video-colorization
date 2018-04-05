function MergeTwoImages
clc;
keyword = 'Rooster_1';
inpDir = 'D:\Raj\ACM MM\Code\Input\';
outDir = 'D:\Raj\ACM MM\Result\';
sm = imread([inpDir keyword '\InputSegMask.png']);
imF = imread([outDir keyword 'F.png']);
imB = imread([outDir keyword 'B.png']);

ch1 = imF(:, :, 1); ch2 = imB(:, :, 1); 
ch2(sm) = ch1(sm);  imF(:, :, 1) =  ch2; 

ch1 = imF(:, :, 2); ch2 = imB(:, :, 2); 
ch2(sm) = ch1(sm);  imF(:, :, 2) =  ch2; 

ch1 = imF(:, :, 3); ch2 = imB(:, :, 3); 
ch2(sm) = ch1(sm);  imF(:, :, 3) =  ch2; 

imwrite(imF, [outDir 'Irony\' keyword '.png']);
return;

% function MergeTwoImages
% keyword = 'Butterfly018';
% inpDir = 'D:\Raj\ACM MM\Code\Input\';
% outDir = 'D:\Raj\ACM MM\Result\';
% sm1 = imread([inpDir keyword '_A\InputSegMask.png']);
% sm2 = imread([inpDir keyword '_B\InputSegMask.png']);
% im1 = imread([outDir keyword '_AF.png']);
% im2 = imread([outDir keyword '_BF.png']);
% imB = imread([outDir keyword '_AB.png']);
% 
% ch1 = im1(:, :, 1); ch2 = imB(:, :, 1); ch3 = im2(:, :, 1);
% ch2(sm1) = ch1(sm1); ch2(sm2) = ch3(sm2); im1(:, :, 1) =  ch2; 
% 
% ch1 = im1(:, :, 2); ch2 = imB(:, :, 2); ch3 = im2(:, :, 2);
% ch2(sm1) = ch1(sm1); ch2(sm2) = ch3(sm2); im1(:, :, 2) =  ch2; 
% 
% ch1 = im1(:, :, 3); ch2 = imB(:, :, 3); ch3 = im2(:, :, 3);
% ch2(sm1) = ch1(sm1); ch2(sm2) = ch3(sm2); im1(:, :, 3) =  ch2; 
% 
% imwrite(im1, [outDir 'Irony\' keyword '_.png']);
% return;
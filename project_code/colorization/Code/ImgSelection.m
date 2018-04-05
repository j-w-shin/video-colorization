%function [Im1, internetGrayImg, Sift1, Sift2] = ImgSelection(InputImage, InputSegmentedImage, internetColorImgName, internetSegImgName, ratio)
function [Im1, internetGrayImg, s1, s2] = ImgSelection(InputImage, internetColorImgName, ratio)

patchsize=8;
gridspacing=1;
SIFTflowpara.alpha=2;
SIFTflowpara.d=40;
SIFTflowpara.gamma=0.005;
SIFTflowpara.nlevels=4;
SIFTflowpara.wsize=5;
SIFTflowpara.topwsize=20;
SIFTflowpara.nIterations=60;

InputImg = im2double(imread(InputImage));
InputImg = imresize(InputImg, ratio);

if size(InputImg, 3) == 3
    InputImg = rgb2lab(InputImg);
    InputImg = InputImg(:, :, 1);
    %InputImg = rgb2gray(InputImg);
else
    InputImg = cat(3, InputImg, InputImg, InputImg);
    InputImg = rgb2lab(InputImg);
    InputImg = InputImg(:, :, 1);
end

[ImgRow ImgCol] = size(InputImg);

s1=dense_sift(InputImg,patchsize,gridspacing);

Im1 = InputImg(4:ImgRow-3, 4:ImgCol-3);
% 
% [ir, ic, ch] = size(Sift1);
% % s1 = zeros(ImgRow-6, ImgCol-6, 130);
% % s1(:,:,3:130) = dense_sift(InputImg,patchsize,gridspacing);
% % s1(:,:,1) = InputImg(4:ImgRow-3, 4:ImgCol-3)/100;
% % s1(:,:,2) = stdfilt(InputImg(4:ImgRow-3, 4:ImgCol-3)/100, ones(7))*2;
% ImageEnergyList = zeros(1, numImg);
internetColorImg = im2double(imread(internetColorImgName));
internetColorImg = imresize(internetColorImg, ratio);
internetColorImg = rgb2lab(internetColorImg);
% Im2=internetColorImg;
internetGrayImg = internetColorImg; %rgb2gray(internetColorImg);

s2=dense_sift(internetGrayImg(:,:,1), patchsize, gridspacing);
[ImgRow ImgCol] = size(internetGrayImg(:,:,1));    
internetGrayImg = internetGrayImg(4:ImgRow-3, 4:ImgCol-3, :);
% % s2 = zeros(ImgRow-6, ImgCol-6, 130);
% % s2(:,:,3:130) = dense_sift(internetGrayImg(:,:,1), patchsize, gridspacing);
% % s2(:,:,1) = internetGrayImg(4:ImgRow-3, 4:ImgCol-3, 1)/100;
% % s2(:,:,2) = stdfilt(internetGrayImg(4:ImgRow-3, 4:ImgCol-3, 1)/100, ones(7)) * 2;
return;


%% To get segmented image region. 
function SegImgRegion = GetPositionOfSegmentedRegion(SegImg)
[r c] = find(SegImg==true);
minR = min(r);      minC = min(c);
maxR = max(r);      maxC = max(c);
SegImgRegion = zeros(2, 2); 
SegImgRegion(1,1) = minR;   SegImgRegion(2,1) = maxR;
SegImgRegion(1,2) = minC;   SegImgRegion(2,2) = maxC;
return;
%% To Get the list of internet images from folder.
function ImgPath = GetImageNames(ImagesFolder)
DirThisMatlab = cd;
cd(ImagesFolder);
DirThisImg = cd;

ImageNames = dir('*.png');
numImg = size(ImageNames,1);
ImgPath = cell(numImg/2,2);

currImgIdx = 0;
for counter = 1:numImg
    currSegImgName = ImageNames(counter).name;
    k = strfind(currSegImgName, '_SalientRegion.png');
    if isempty(k)
        %   colored image.
        continue;
    end
    currColImgName = currSegImgName(1:k(1)-1);
    currColImgName = [currColImgName '.png']; %#ok<*AGROW>
    
    if exist(currColImgName, 'file') == 0
        error('Color image does not exist.');
    end
    
    currImgIdx = currImgIdx+1;
    tokens = regexp(currColImgName, '_', 'split');
    ImgPath{currImgIdx,1} = [DirThisImg '\' currColImgName];
    ImgPath{currImgIdx,2} = [DirThisImg '\' currSegImgName];  
end
    
ImgPath(currImgIdx+1:end,:) = [];
cd(DirThisMatlab);
return;









% function ImgFolder = MatchSIFTFlow(ImgFolder)
% 
% numImg = size(ImgFolder,1);
% for ii = 1:numImg
%     currEnergyList = ImgFolder{ii,5};
%     currEnergyList = currEnergyList(end);
%     currEnergyList = currEnergyList.data;
%     ImgFolder{ii,6} = currEnergyList(end);
% end
% 
% SortScore = cell2mat(ImgFolder(:,6));
% SortScore(:,2) = 1:38;
% SortScore = sortrows(SortScore,1);
% 
% DirSearchImg = 'C:\Users\Alex Chia\Desktop\Internet Vision Project\Third party codes\siftFlow\Imgs Rooster\Images\';
% InputImg = imread('Rooster.jpg');
% InputImg = rgb2gray(InputImg);
% InputImg = imresize(imfilter(InputImg,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
% 
% [ImgRow ImgCol] = size(InputImg);
% inputDiagonal = sqrt(ImgRow*ImgRow + ImgCol*ImgCol);
% 
% for ii = 1:numImg
%     clc;
%     disp(['Num Img: ' num2str(ii) ' of ' num2str(numImg) '.']);
%     
%     ImgIdx = SortScore(ii,2);
%     
%     currPairColorImgName = ImgFolder{ImgIdx,1};
%     currPairColorImgName = [DirSearchImg currPairColorImgName]; %#ok<*AGROW>
%     currPairColorImg = imread(currPairColorImgName);
%     currPairColorImg = im2double(currPairColorImg);
%     currPairGrayImg = rgb2gray(currPairColorImg);
%     
%     currPairSegImgName = ImgFolder{ImgIdx,2};
%     currPairSegImgName = [DirSearchImg currPairSegImgName];
%     currPairSegImg = imread(currPairSegImgName);
%     
%     currPairSegImg = im2bw(currPairSegImg);
%     [r c] = find(currPairSegImg==1);
%     minR = min(r);      maxR = max(r);
%     minC = min(c);      maxC = max(c);
%     
%     currPairGrayImg = currPairGrayImg(minR:maxR, minC:maxC,:);
%     currPairSegImg = currPairSegImg(minR:maxR, minC:maxC,:);        
%     
%     diffR = maxR-minR+1;    diffC = maxC-minC+1;
%     currDiagonal = sqrt(diffR*diffR + diffC*diffC);
%     currResizeRatio = inputDiagonal/currDiagonal;
%     currPairSegImg = imresize(currPairSegImg, currResizeRatio);
% 	currPairGrayImg = imresize(currPairGrayImg, currResizeRatio);
%     
%     RandomImg = rand(size(currPairGrayImg,1), size(currPairGrayImg,2));
%     
%     q = currPairGrayImg;
%     q(~currPairSegImg) = 1;
%     currPairGrayImg(~currPairSegImg) = RandomImg(~currPairSegImg);
% 
%     im2 = currPairGrayImg;
%     
%     im2=imresize(imfilter(im2,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
%     
%     vx = ImgFolder{ImgIdx,3};
%     vy = ImgFolder{ImgIdx,4};    
%     warpI2=warpImage(im2,vx,vy);
%     figure;imshow(InputImg);set(gca,'Position',[0 0 1 1]);
%     figure;imshow(warpI2); set(gca,'Position',[0 0 1 1]);
%     figure; imshow(q);set(gca,'Position',[0 0 1 1]);
%     pause;
%     close all;
% end
%     
% 
% 
% return;

% function MatchSIFTFlow
% 
% cellsize=3;
% gridspacing=1;
% addpath(fullfile(pwd,'mexDenseSIFT'));
% addpath(fullfile(pwd,'mexDiscreteFlow'));
% 
% InputImg = imread('Rooster.jpg');
% InputImg = rgb2gray(InputImg);
% 
% InputImg = imresize(imfilter(InputImg,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
% InputImg=im2double(InputImg);
% sift1 = mexDenseSIFT(InputImg,cellsize,gridspacing);
% 
% [ImgRow ImgCol] = size(InputImg);
% inputDiagonal = sqrt(ImgRow*ImgRow + ImgCol*ImgCol);
% 
% DirSearchImg = 'C:\Users\Alex Chia\Desktop\Internet Vision Project\Third party codes\siftFlow\Imgs Rooster\Images\';
% 
% 
% ImgFolder = GetImg(DirSearchImg);
% 
% numImg = size(ImgFolder,1);
% 
% 
% 
% SIFTflowpara.alpha=2*255;
% SIFTflowpara.d=40*255;
% SIFTflowpara.gamma=0.005*255;
% SIFTflowpara.nlevels=4;
% SIFTflowpara.wsize=2;
% SIFTflowpara.topwsize=10;
% SIFTflowpara.nTopIterations = 60;
% SIFTflowpara.nIterations= 30;
% 
% 
% 
% 
% for ii = 1:numImg
%     clc;
%     disp(['Num Img: ' num2str(ii) ' of ' num2str(numImg) '.']);
%     
%     currPairColorImgName = ImgFolder{ii,1};
%     currPairColorImgName = [DirSearchImg currPairColorImgName]; %#ok<*AGROW>
%     currPairColorImg = imread(currPairColorImgName);
%     currPairColorImg = im2double(currPairColorImg);
%     currPairGrayImg = rgb2gray(currPairColorImg);
%     
%     currPairSegImgName = ImgFolder{ii,2};
%     currPairSegImgName = [DirSearchImg currPairSegImgName];
%     currPairSegImg = imread(currPairSegImgName);
%     
%     currPairSegImg = im2bw(currPairSegImg);
%     
%     [r c] = find(currPairSegImg==1);
%     minR = min(r);      maxR = max(r);
%     minC = min(c);      maxC = max(c);
%     
%     currPairGrayImg = currPairGrayImg(minR:maxR, minC:maxC,:);
%     currPairSegImg = currPairSegImg(minR:maxR, minC:maxC,:);    
%     
%     diffR = maxR-minR+1;    diffC = maxC-minC+1;
%     currDiagonal = sqrt(diffR*diffR + diffC*diffC);
%     currResizeRatio = inputDiagonal/currDiagonal;
%     currPairSegImg = imresize(currPairSegImg, currResizeRatio);
%     currPairGrayImg = imresize(currPairGrayImg, currResizeRatio);
%     
%     
%     RandomImg = rand(size(currPairGrayImg,1), size(currPairGrayImg,2));
%     
%     currPairGrayImg(~currPairSegImg) = RandomImg(~currPairSegImg);
%     
%     im2 = currPairGrayImg;
%     
%     im2=imresize(imfilter(im2,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
%     sift2 = mexDenseSIFT(im2,cellsize,gridspacing);
%     
%     [vx,vy,energylist]=SIFTflowc2f(sift1,sift2,SIFTflowpara);
% 
%     ImgFolder{ii,3} = vx;
%     ImgFolder{ii,4} = vy;
%     ImgFolder{ii,5} = energylist;
%     
%     if mod(ii,50) == 0
%         save('SiftFlow.mat', 'ImgFolder');
%     end    
% end
% save('SiftFlow.mat', 'ImgFolder');
% return;
% 
% 
% 
% function ImgFolder = GetImg(DirSearchImg)
% DirThisMatlab = cd;
% cd(DirSearchImg);
% FileName = dir('*.png');
% numFile = size(FileName,1);
% 
% ImgFolder = cell(numFile,5);
% currImgIdx = 0;
% for counter = 1:numFile
%     currFileName = FileName(counter).name;
%     if isempty(strfind(currFileName, 'SalientRegion.png'))
%         continue;
%     end
%     currImgIdx = currImgIdx + 1;
%     currColorFileName = strtok(currFileName, 'SalientRegion.png');
%     currColorFileName = currColorFileName(1:end-1);
%     currColorFileName = [currColorFileName '.png']; %#ok<AGROW>
%     
%     if exist(currColorFileName, 'file') == 0
%        error('File not found');
%     end
%     
%     ImgFolder{currImgIdx, 1} = currColorFileName;
%     ImgFolder{currImgIdx, 2} = currFileName;
% end
% ImgFolder(currImgIdx+1:end,:) = [];
% cd(DirThisMatlab);
% return;



















% % function ImgFolder = MatchSIFTFlow(ImgFolder)
% % 
% % numImg = size(ImgFolder,1);
% % for ii = 1:numImg
% %     currEnergyList = ImgFolder{ii,5};
% %     currEnergyList = currEnergyList(end);
% %     currEnergyList = currEnergyList.data;
% %     ImgFolder{ii,6} = currEnergyList(end);
% % end
% % 
% % SortScore = cell2mat(ImgFolder(:,6));
% % SortScore(:,2) = 1:50;
% % SortScore = sortrows(SortScore,1);
% % 
% % DirSearchImg = 'C:\Users\Alex Chia\Desktop\Internet Vision Project\Download Image\Parrot Google\Saliency_S2P\ShapeContextFilter\';
% % InputImg = imread('Parrot.jpg');
% % InputImg = imresize(imfilter(InputImg,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
% % 
% % for ii = 1:numImg
% %     clc;
% %     disp(['Num Img: ' num2str(ii) ' of ' num2str(numImg) '.']);
% %     
% %     ImgIdx = SortScore(ii,2);
% %     
% %     currPairColorImgName = ImgFolder{ImgIdx,1};
% %     currPairColorImgName = [DirSearchImg currPairColorImgName]; %#ok<*AGROW>
% %     currPairColorImg = imread(currPairColorImgName);
% %     currPairColorImg = im2double(currPairColorImg);
% %     
% %     currPairSegImgName = ImgFolder{ImgIdx,2};
% %     currPairSegImgName = [DirSearchImg currPairSegImgName];
% %     currPairSegImg = imread(currPairSegImgName);
% %     
% %     currPairSegImg = im2bw(currPairSegImg);
% %     
% %     r = currPairColorImg(:,:,1);    g = currPairColorImg(:,:,2);    b = currPairColorImg(:,:,3);
% %     r(~currPairSegImg) = 0;       g(~currPairSegImg) = 1;       b(~currPairSegImg) = 1;
% %     im2 = cat(3,r,g,b);
% %     
% %     im2=imresize(imfilter(im2,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
% %     
% %     vx = ImgFolder{ImgIdx,3};
% %     vy = ImgFolder{ImgIdx,4};    
% %     warpI2=warpImage(im2,vx,vy);
% %     figure;imshow(InputImg);figure;imshow(warpI2);
% %     pause;
% %     close all;
% % end
% %     
% % 
% % 
% % return;
% 
% function MatchSIFTFlow
% 
% cellsize=3;
% gridspacing=1;
% addpath(fullfile(pwd,'mexDenseSIFT'));
% addpath(fullfile(pwd,'mexDiscreteFlow'));
% 
% InputImg = imread('Parrot.jpg');
% InputImg = imresize(imfilter(InputImg,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
% InputImg=im2double(InputImg);
% sift1 = mexDenseSIFT(InputImg,cellsize,gridspacing);
% 
% DirSearchImg = 'C:\Users\Alex Chia\Desktop\Internet Vision Project\Download Image\Parrot Google\Saliency_S2P\ShapeContextFilter\';
% 
% ImgFolder = GetImg(DirSearchImg);
% 
% numImg = size(ImgFolder,1);
% 
% 
% 
% SIFTflowpara.alpha=2*255;
% SIFTflowpara.d=40*255;
% SIFTflowpara.gamma=0.005*255;
% SIFTflowpara.nlevels=4;
% SIFTflowpara.wsize=2;
% SIFTflowpara.topwsize=10;
% SIFTflowpara.nTopIterations = 60;
% SIFTflowpara.nIterations= 30;
% 
% 
% 
% 
% for ii = 1:numImg
%     clc;
%     disp(['Num Img: ' num2str(ii) ' of ' num2str(numImg) '.']);
%     
%     currPairColorImgName = ImgFolder{ii,1};
%     currPairColorImgName = [DirSearchImg currPairColorImgName]; %#ok<*AGROW>
%     currPairColorImg = imread(currPairColorImgName);
%     currPairColorImg = im2double(currPairColorImg);
%     currPairColorImg = rgb2gray(currPairColorImg);
%     
%     currPairSegImgName = ImgFolder{ii,2};
%     currPairSegImgName = [DirSearchImg currPairSegImgName];
%     currPairSegImg = imread(currPairSegImgName);
%     
%     currPairSegImg = im2bw(currPairSegImg);
%     
%     r = currPairColorImg(:,:,1);    g = currPairColorImg(:,:,2);    b = currPairColorImg(:,:,3);
%     r(~currPairSegImg) = 0.5;       g(~currPairSegImg) = 0.5;       b(~currPairSegImg) = 0.5;
%     im2 = cat(3,r,g,b);
%     
%     im2=imresize(imfilter(im2,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
%     sift2 = mexDenseSIFT(im2,cellsize,gridspacing);
%     
%     [vx,vy,energylist]=SIFTflowc2f(sift1,sift2,SIFTflowpara);
% 
%     ImgFolder{ii,3} = vx;
%     ImgFolder{ii,4} = vy;
%     ImgFolder{ii,5} = energylist;
%     
%     if mod(ii,50) == 0
%         save('SiftFlow.mat', 'ImgFolder');
%     end    
% end
% save('SiftFlow.mat', 'ImgFolder');
% return;
% 
% 
% 
% function ImgFolder = GetImg(DirSearchImg)
% DirThisMatlab = cd;
% cd(DirSearchImg);
% FileName = dir('*.png');
% numFile = size(FileName,1);
% 
% ImgFolder = cell(numFile,5);
% currImgIdx = 0;
% for counter = 1:numFile
%     currFileName = FileName(counter).name;
%     if isempty(strfind(currFileName, 'SalientRegion.png'))
%         continue;
%     end
%     currImgIdx = currImgIdx + 1;
%     currColorFileName = strtok(currFileName, 'SalientRegion.png');
%     currColorFileName = currColorFileName(1:end-1);
%     currColorFileName = [currColorFileName '.png']; %#ok<AGROW>
%     
%     if exist(currColorFileName, 'file') == 0
%        error('File not found');
%     end
%     
%     ImgFolder{currImgIdx, 1} = currColorFileName;
%     ImgFolder{currImgIdx, 2} = currFileName;
% end
% ImgFolder(currImgIdx+1:end,:) = [];
% cd(DirThisMatlab);
% return;

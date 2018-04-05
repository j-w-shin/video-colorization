function test()
% load matlab.mat
clc;
load island_lab.mat
inImg = inImg/100;
grayImg = cat(3, inImg, inImg, inImg);
grayImg = rgb2lab(grayImg);
superPixelCount = max(label1(:));

ComputeConfidenceMap(N, label1)

%% Get Color Info of each superpixel.
CrImg = inCol(:, :, 2);
CbImg = inCol(:, :, 3);
spMean = zeros(superPixelCount, 1);
colorInfo = zeros(superPixelCount, 2);
for i=1:superPixelCount
    % Get the first NN mask.
    sp2 = (label2 == N(5, i));
    colorInfo(i, 1) = mean(CrImg(sp2));
    colorInfo(i, 2) = mean(CbImg(sp2));
    spMean(i) = mean(inImg(label1 == i));
end

nnList = GetListofNeighbouringSuperPixels(label1, 3);
ApplyBP(inImg, label1, nnList, spMean, colorInfo);
% width = 80;
% numClust = 14;
% [clustNum, centroid] = kmeans(colorInfo, numClust);
% % 
% template = zeros(width, numClust * width, 3);
% template(:,:,1) = 50;
% 
% for i=1:(numClust * width) - 1
%     template(:, i, 2) = centroid(floor(i/width) + 1, 1);
%     template(:, i, 3) = centroid(floor(i/width) + 1, 2);
% end
% template(:, numClust * width, 2) = centroid(numClust, 1);
% template(:, numClust * width, 3) = centroid(numClust, 2);
% 
% template = lab2rgb(template);
% figure;imshow(template);
% pp = inCol;
% pp(:,:,1)=60;
% figure;imshow(lab2rgb(pp));

% grayImg = lab2rgb(grayImg);
% imwrite(grayImg, 'ScribbledImg.png');
% pp=colorize(inImg, grayImg);
% imwrite(pp, 'Result.png');
return;

function ScribbledColorsImg = ApplyBP(inImg, labels, nnList, superpixelMean, colorMatrix)

NumLabels = 8;
NumIterations = 3000;
SpCount = size(superpixelMean, 1);
[clustNum, centroid] = kmeans(colorMatrix, NumLabels);

[pairs wcosts] = ComputePairwiseStrength(inImg, labels, nnList, superpixelMean);

pairs = pairs - 1;

NumPairs = size(pairs, 1)/2;

lcost = size(SpCount * NumLabels, 1);
for k = 0:NumLabels-1
    for sp = 1:SpCount
        c = colorMatrix(sp, :) - centroid(k+1, :);
        c = sqrt(c(1) * c(1) + c(2) * c(2));
        lcost(k*SpCount+sp) = c;
    end
end
dist = zeros(NumLabels * NumLabels, 1);
for i = 0:NumLabels-1
    for j = 1:NumLabels
        c = centroid(i+1, :) - centroid(j, :);
        c = sqrt(c(1) * c(1) + c(2) * c(2));
        dist(i*NumLabels+j) = c;
    end
end

TextFileName_Input = 'InputBeliefPropagation.txt';
TextFileName_Output = 'OutputBeliefPropagation.txt';
exe_string = 'FastPD.exe';
if exist(TextFileName_Input, 'file') ~= 0
    delete(TextFileName_Input);
end
if exist(TextFileName_Output, 'file') ~= 0
    delete(TextFileName_Output);
end

fid = fopen(TextFileName_Input, 'wb');

fwrite(fid, SpCount, 'int');
fwrite(fid, NumPairs, 'int');
fwrite(fid, NumLabels, 'int');
fwrite(fid, NumIterations, 'int');

fwrite(fid, lcost, 'float');
fwrite(fid, pairs, 'int');
fwrite(fid, dist, 'float');
fwrite(fid, wcosts, 'float');
fclose(fid);

disp('Performing belief propagation...');
exeString = [exe_string ' ' TextFileName_Input ' ' TextFileName_Output];
dos(exeString);

fid=fopen('OutputBeliefPropagation.txt', 'r');
txt = textscan(fid, '%d %d');
fclose(fid);
bpColor = txt{1, 2} + 1;
% bpColor = clustNum;
ScribbledColorsImg = CreateScribbleImg(inImg, labels, bpColor, clustNum, centroid);%colorMatrix);
ScribbledColorsImg = lab2rgb(ScribbledColorsImg);
p1=inImg * 100;
p2=inImg; p2(:)=0;
p3=inImg; p3(:)=0;
inImg = cat(3, p1, p2, p3);
inImg = lab2rgb(inImg);
inImg = rgb2gray(inImg);
pp=colorize(inImg, ScribbledColorsImg);
figure;imshow(IncreaseSaturation(pp, 20));
imwrite(pp, 'Results\result_bp.png');

return;


function Result = GetListofNeighbouringSuperPixels(labels, diskSize)

TotalSuperpixels = max(labels(:));
Result = cell(TotalSuperpixels, 3);

se = strel('disk', diskSize);
for I = 1:TotalSuperpixels
    sp = (labels == I);
    spd = imdilate(sp, se);
    spNN = unique(labels(spd));
    spNN = spNN(spNN ~= I);
    Result{I, 1} = I;
    Result{I, 2} = size(spNN, 1);
    Result{I, 3} = spNN;
end

return

function [pairs wcost] = ComputePairwiseStrength(inImg, labels, nnList, superpixelMean)

TotalSuperpixels = max(labels(:));
wcost = zeros(TotalSuperpixels, TotalSuperpixels);
nnInfo = false(TotalSuperpixels, TotalSuperpixels);
index = 1;
pairs = zeros(TotalSuperpixels * 18, 1);

for I = 1:TotalSuperpixels
    spNN = nnList{I, 3};
    spNN = spNN(spNN > I);
    spStd = std(inImg(labels == I));
    for J = 1:size(spNN)
        K = spNN(J);
        intensityDiff = abs(superpixelMean(I) - superpixelMean(K));
        stdDiff = abs(spStd - std(inImg(labels == K)));
        wcost(I, K) = sqrt(intensityDiff .^ 2 + stdDiff .^2);
        wcost(I, K) = wcost(I, K) / (intensityDiff + stdDiff);
        nnInfo(I, K) = true;

        pairs(index) = I;
        index = index + 1;
        pairs(index) = spNN(J);
        index = index + 1;
    end
end
minVal = min(wcost(nnInfo));
maxVal = max(wcost(nnInfo));
wcost(nnInfo) = (wcost(nnInfo) - minVal) / (maxVal - minVal);

pairs = pairs(1:(index - 1));
wcost = wcost';
nnInfo = nnInfo';
wcost = wcost(nnInfo);
return

function ScribbledColorsImg = CreateScribbleImg(inImg, labels, bpColor, clustNum, colorMatrix)


% Number of super pixels in the gray image.
% Each label represents one super pixel.
numSuperPixels = size(clustNum, 1); 

ScribbledSeedsColorsImg_Y = inImg * 100;   
ScribbledSeedsColorsImg_Cr = inImg;   
ScribbledSeedsColorsImg_Cb = inImg;   
ScribbledSeedsColorsImg_Cr(:) = 0;
ScribbledSeedsColorsImg_Cb(:) = 0;

SEScribble = strel('disk', 7);

for currSuperPixelLabel = 1:numSuperPixels
    % Get the binary mask of the current super pixel.
    currLabelBinaryMap = labels == currSuperPixelLabel;
    % Need to call one function here that will get the color of that 
    % super pixel from wrapped image/s.
%     if sum(currLabelBinaryMap(:)) == 0
%         continue;
%     end

    %   Getting the center of the super pixel, while ensuring that the
    %   center of the super pixel is within the super pixel.
    Cr = colorMatrix(bpColor(currSuperPixelLabel), 1);
    Cb = colorMatrix(bpColor(currSuperPixelLabel), 2);
    
    [r c] = find(currLabelBinaryMap==1);
    [currCenter MaxDiff] = GetMedoid([r c]);
    ScribbledSeedsColorsImg_Cr(currCenter(1), currCenter(2)) = Cr;
    ScribbledSeedsColorsImg_Cb(currCenter(1), currCenter(2)) = Cb;

%     %   Assigning more pixels within the super pixel colors.
%     currLabelBinaryMap = imerode(currLabelBinaryMap,SEScribble);  
% 
%     ScribbledSeedsColorsImg_g(currLabelBinaryMap) = rgb_g;
%     ScribbledSeedsColorsImg_b(currLabelBinaryMap) = rgb_b;
end
ScribbledColorsImg = cat(3, ScribbledSeedsColorsImg_Y, ...
                                 ScribbledSeedsColorsImg_Cr, ...
                                 ScribbledSeedsColorsImg_Cb);
return;

%   Function will find the medoid from Data.
%   Data is a N x D matrix, where N is the number of feature vectors and D
%   is the dimension of each feature vector.
function [currSuperPixelColor MaxColorSqDiff] = GetMedoid(Data)
numData = size(Data,1);
%   Finding the distance between itself and its neighbors.
DistMatrix = zeros(numData,numData);
for currIdx = 1:numData
    currFV = Data(currIdx,:);
    nextFV = Data;
    currFV = repmat(currFV, numData, 1);
    
    currDist = abs(currFV - nextFV);
    currDist = sum(currDist,2);
    DistMatrix(:,currIdx) = currDist;
end
%   Finding the total distance between itself and all its neighbor.
DistLinearMatrix = sum(DistMatrix);
[minDist minIdx] = min(DistLinearMatrix);
currSuperPixelColor = Data(minIdx,:);
MaxColorSqDiff = max(DistMatrix(minIdx,:));
return;

function sat = IncreaseSaturation(RGB, val)
HSV = rgb2hsv(RGB);
% "20% more" saturation:
HSV(:, :, 2) = HSV(:, :, 2) * (1 + (val/100));
% or add:
% HSV(:, :, 2) = HSV(:, :, 2) + 0.2;
HSV(HSV > 1) = 1;  % Limit values
sat = hsv2rgb(HSV);
return;


function ComputeConfidenceMap(nnLabel, labelMap)
nnList = GetListofNeighbouringSuperPixels(labelMap, 5);
spCount = size(nnList, 1);
img = labelMap;

for I=1:spCount
    spNN = nnList{I, 3};
    nnCount = 0;
    for J=1:nnList{I, 2}
        if nnLabel(1, I) == nnLabel(1, spNN(J))
            nnCount = nnCount + 1;
        end
    end
    img(img == I) = nnCount;
end
img = img / max(img(:));
figure;imshow(img);
return;
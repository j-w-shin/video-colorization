function test_median_Voting_New(keyword)
close all;
% keyword = 'lion';
outputFileName = ['~/Documents/DL/video-colorization/project_code/colorization/Code/Results/' keyword];
% mkdir(outputFileName);
test_median1([keyword '_lab.mat'], outputFileName);
delete([keyword '_lab.mat']);
return;

function test_median1(fName, outputFileName)

load(fName)
%% Get the Y channel of YCbCr color space that to be used by box filter.


p2 = inImg; p2(:) = 0;
p3 = inImg; p3(:) = 0;
grayImg = cat(3, inImg, p2, p3);
grayY = rgb2ycbcr(lab2rgb(grayImg));    
grayY = grayY(:,:,1);
%% Get Gray Image that to be used by Lavin's Code.
inImg = lab2rgb(grayImg);
edImg = (inImg - min(inImg(:)))/(max(inImg(:)) - min(inImg(:)));
inImg = rgb2gray(inImg);
%% Compute the scribbled image.
superPixelCount = max(label1(:));
colorInfo = zeros(superPixelCount, 5);
clc; disp('Computing Seeds...');

confMap = inImg;
confMap(:) = 0;
pixMap = im2bw(inImg);
pixMap(:) = 0;

spMap = im2bw(inImg);
spMap(:) = 0;

maxMap = im2bw(inImg);
maxMap(:) = 0;

for i=1:superPixelCount
    % Get the centroid of the input gray Image.
    sp1 = (label1 == i);
    [row col] = find(sp1==1);
    minR = min(row);      maxR = max(row);
    minC = min(col);      maxC = max(col);
    rc1 = floor((maxR + minR)/2); cc1 = floor((maxC + minC)/2);
    % Get the NN SP with Minimum Cost.
    nnCost = costMatrix(1:nnCount, i);
    index = find(nnCost == min(nnCost(:)));
    if size(index, 1) == 0 
        continue;
    end
%     disp([num2str(i) ' ... ' num2str(index(1))])
    index = N(index(1), i); 
    % Get the color info of Super pixel.
    colorInfo(i, 1) = rc1;
    colorInfo(i, 2) = cc1;
    colorInfo(i, 3) = spCol2(index, 3);
    colorInfo(i, 4) = spCol2(index, 4);
end
EdisonLabel = edImg;
EdisonLabel(:) = -1;
%[ISeg EdisonLabel modes regSize grad conf]=edison_wrapper(edImg,@RGB2Luv,'step',2,'SpatialBandWidth',7,'RangeBandWidth',6.5,'MinimumRegionArea',20);
[ISeg EdisonLabel modes regSize grad conf]=edison_wrapper(edImg,@RGB2Luv,'step',2,'SpatialBandWidth',1,'RangeBandWidth',1,'MinimumRegionArea',50);
% EdisonLabel = UpdateSegMask(EdisonLabel);
% ColorizeSegment_castle(EdisonLabel, label1, [outputFileName '_seg.png']);
%ColorizeSegment(EdisonLabel, [outputFileName '_seg.png']);
%imwrite(lab2rgb(ISeg), [outputFileName '_seg.png']);
for ed=0:max(EdisonLabel(:))
    segMask = EdisonLabel == ed;
    if sum(segMask(:)) < 10
        continue;
    end
    % Get the mask of selected segment.
    label = label1 .* segMask;
    % Get the mask of unselected segment.
    lb = label1 .* (~segMask); 
    % Get the list of superpixels in unselected segments.
    sSP = uint16(unique(lb));
    sSP = sSP(sSP~=0);
    sPixCount = max(size(sSP));
    % Get the list of superpixels in selected segments.
    segmentedSP = uint16(unique(label));
    segmentedSP = segmentedSP(segmentedSP~=0);
    for i=1:sPixCount
        if min(size(find(segmentedSP==sSP(i))))
            s1 = label == sSP(i);
            s1 = sum(s1(:));
            s2 = lb == sSP(i);
            s2 = sum(s2(:));
            if s2 > s1 
                segmentedSP = segmentedSP(segmentedSP~=sSP(i));
            end
        end
    end

    segmentedPixCount = max(size(segmentedSP));
    segmentedColor = zeros(segmentedPixCount, 2);
    if segmentedPixCount < 3
        continue;
    end
    for i=1:segmentedPixCount
        segmentedColor(i, 1) = colorInfo(segmentedSP(i), 3);
        segmentedColor(i, 2) = colorInfo(segmentedSP(i), 4);
    end
    try
        index = kmeans(segmentedColor, 2);
    catch err
        continue;
    end
    s1 = sum(index == 1);
    s2 = sum(index == 2);
    if 1%(abs(s1-s2) / (s1+s2)) > 0.10      
        if s1 > s2
                A = mean(segmentedColor((index == 1), 1));
                B = mean(segmentedColor((index == 1), 2));
                colorInfo(segmentedSP(index == 1), 3) = A;
                colorInfo(segmentedSP(index == 1), 4) = B;                
                colorInfo(segmentedSP(index == 2), 3) = A;
                colorInfo(segmentedSP(index == 2), 4) = B;
                pixMap = BinaryMap(pixMap, label1, segmentedSP(index == 2));
        else
                A = mean(segmentedColor((index == 2), 1));
                B = mean(segmentedColor((index == 2), 2));
                colorInfo(segmentedSP(index == 1), 3) = A;
                colorInfo(segmentedSP(index == 1), 4) = B;  
                colorInfo(segmentedSP(index == 2), 3) = A;
                colorInfo(segmentedSP(index == 2), 4) = B;                
                pixMap = BinaryMap(pixMap, label1, segmentedSP(index == 1));
        end
    end
    l1=segmentedSP(index == 1);
    l2=segmentedSP(index == 2);
    for c=1:length(l1)
        confMap(label1 == l1(c)) = 1 - (s1/(s1 + s2));
    end
    for c=1:length(l2)
        confMap(label1 == l2(c)) = 1 - (s2/(s1 + s2));
    end    
end

imwrite(pixMap, [outputFileName '_ColorMap.png']);
imwrite(confMap/max(confMap(:)), [outputFileName '_Conf1.png']);
% if 1
%     return;
% end

for i=1:superPixelCount
    spNN = nnList1{i, 3};
    spNNC1 = colorInfo(spNN, 3);
    spNNC2 = colorInfo(spNN, 4);
    m1 = median(spNNC1);
    m2 = median(spNNC2);
    if colorInfo(i, 1) && colorInfo(i, 2)
        grayImg(colorInfo(i, 1), colorInfo(i, 2), 2) = m1;
        grayImg(colorInfo(i, 1), colorInfo(i, 2), 3) = m2;
    end
end
% for i=1:superPixelCount
% %     if colorInfo(i, 1) && colorInfo(i, 2)
%         grayImg(colorInfo(i, 1), colorInfo(i, 2), 2) = colorInfo(i, 3);
%         grayImg(colorInfo(i, 1), colorInfo(i, 2), 3) = colorInfo(i, 4);
% %     end
% end

clc; disp('Performing Image Colorization...');
grayImg = lab2rgb(grayImg);
imwrite(grayImg, [outputFileName '_ScribbledImg.png']);
pp=colorize(inImg, grayImg);
% max_d=floor(log(min(size(grayY,1),size(grayY,2)))/log(2)-2);
% iu=floor(size(grayY,1)/(2^(max_d-1)))*(2^(max_d-1));
% ju=floor(size(grayY,2)/(2^(max_d-1)))*(2^(max_d-1));
% id=1; jd=1;
% grayY=grayY(id:iu,jd:ju,:);
pp = smoothScribbleImage(rgb2ycbcr(pp), grayY);
pp = smoothScribbleImage(rgb2ycbcr(pp), grayY);
imwrite(pp, [outputFileName '_Result.png']);
%figure; imshow(pp);
pp=IncreaseSaturation(pp, 0.20);
imwrite(pp, [outputFileName '_Result_sat.png']);
%figure; imshow(pp);

for i=1:superPixelCount
    spMap(colorInfo(i, 1), colorInfo(i, 2)) = 1;
end

% all_vals = unique(confMap);
% top_val  = all_vals(end-9);

tmp_mapped = 1 - ((1 - confMap).*spMap);

% [~, sortedIndex] = sort(tmp_mapped(:), 'ascend');
% max_index = sortedIndex(1:10);

exit_flag = false;
curr_idx_count = 0;
num_pixs = 300;

min_idx_limited = zeros(num_pixs, 1);

while true
    curr_min = min(tmp_mapped(:));
%     curr_count = size(find(tmp_mapped == curr_min), 1);
%     if curr_count + curr_idx_count > 10
%         exit_flag = true;
%     end

    min_idx = find(tmp_mapped == curr_min);
    min_idx = min_idx(randperm(size(min_idx, 1)));

    for i=1:size(min_idx, 1)
        min_idx_limited(curr_idx_count + 1) = min_idx(i);
        curr_idx_count = curr_idx_count + 1;
        if curr_idx_count >= num_pixs
            exit_flag = true;
            break;
        end
    end 
    
    if exit_flag
        break;
    end

    tmp_mapped(min_idx) = Inf;
end

for i=1:size(min_idx_limited, 1)
    [x_idx, y_idx] = ind2sub(size(tmp_mapped), min_idx_limited(i));
    maxMap(x_idx, y_idx) = 1;
end



mapped = maxMap.*pp;

% [~, idx] = max(mapped);
% mapped(idx) = 1;
% mapped = mapped.*pp;

% idx = find(mapped >= top_val);
% mapped = mapped(idx);
imwrite(mapped, [outputFileName '_pixmap_sat.png']);

% fileID = fopen('mapped_map.txt','w');
% fprintf(fileID,'%6s %12s\n','x','exp(x)');
% fprintf(fileID,'%6.2f %12.8f\n',A);
% fclose(fileID);

return;

function RGB = IncreaseSaturation(RGB, val)
HSV = rgb2hsv(RGB);
% "20% more" saturation:
HSV(:, :, 2) = HSV(:, :, 2) * (1 + val);
% or add:
% HSV(:, :, 2) = HSV(:, :, 2) + 0.2;
HSV(HSV > 1) = 1;  % Limit values
RGB = hsv2rgb(HSV);
return;


function pixMap = BinaryMap(pixMap, label, spList)
for I=1:length(spList)
    pixMap(label == spList(I)) = 1;
end
return;

function ColorizeSegment_castle(edlabel, label1, fName)
inImg = edlabel;
inImg(:) = 0;
r = inImg;
g = inImg;
b = inImg;
for i=0:max(edlabel(:))
    r(edlabel == i) = rand(1) * 255;
    g(edlabel == i) = rand(1) * 255;
    b(edlabel == i) = rand(1) * 255;
end
bd = im2bw(imread('castle_boundary.png'));
r(bd)=0;
g(bd)=0;
b(bd)=0;
inImg = cat(3, r, g, b);
t=DecreaseVisibility(uint8(inImg), 0.75);
% r = rgb2gray(uint8(inImg));
% g = r;
% b = r;
r = t(:, :, 1);
g = t(:, :, 2);
b = t(:, :, 3);

for ed=0:max(edlabel(:))
    segMask = edlabel == ed;
    if sum(segMask(:)) < 10
        continue;
    end
    % Get the mask of selected segment.
    label = label1 .* segMask;
    % Get the mask of unselected segment.
    lb = label1 .* (~segMask); 
    % Get the list of superpixels in unselected segments.
    sSP = uint16(unique(lb));
    sSP = sSP(sSP~=0);
    sPixCount = max(size(sSP));
    % Get the list of superpixels in selected segments.
    segmentedSP = uint16(unique(label));
    segmentedSP = segmentedSP(segmentedSP~=0);
    for i=1:sPixCount
        if min(size(find(segmentedSP==sSP(i))))
            s1 = label == sSP(i);
            s1 = sum(s1(:));
            s2 = lb == sSP(i);
            s2 = sum(s2(:));
            if s2 > s1 
                segmentedSP = segmentedSP(segmentedSP~=sSP(i));
            end
        end
    end
    segmentedPixCount = max(size(segmentedSP));
    if segmentedPixCount < 3
        continue;
    end
    segMask = edlabel == ed;
    [row col] = size(segMask);
    tmp = false(row+2, col+2);
    tmp(2:row+1, 2:col+1) = segMask;
    sm = imerode(tmp, strel('disk', 4));
    sm = sm(2:end-1, 2:end-1);
    sm = im2bw(segMask-sm);
    r(sm)=1;
    g(sm)=1;
    b(sm)=0;
end
seg = cat(3, r, g, b);
% inImg = cat(3, r, g, b);
imwrite(uint8(inImg), fName);
imwrite(seg, 'Result/castle_bd.png');
return;

function RGB = DecreaseVisibility(RGB, val)
HSV = rgb2hsv(RGB);
% "20% more" saturation:
HSV(:, :, 3) = HSV(:, :, 3) * val;
% or add:
% HSV(:, :, 2) = HSV(:, :, 2) + 0.2;
HSV(HSV > 1) = 1;  % Limit values
RGB = hsv2rgb(HSV);
return;

% function ColorizeSegment_castle(edlabel, label1, fName)
% inImg = edlabel;
% inImg(:) = 0;
% r = inImg;
% g = inImg;
% b = inImg;
% for i=0:max(edlabel(:))
%     r(edlabel == i) = rand(1) * 255;
%     g(edlabel == i) = rand(1) * 255;
%     b(edlabel == i) = rand(1) * 255;
% end
% bd = im2bw(imread('castle_boundary.png'));
% r(bd)=0;
% g(bd)=0;
% b(bd)=0;
% inImg = cat(3, r, g, b);
% r = rgb2gray(uint8(inImg));
% g = r;
% b = r;
% for ed=0:max(edlabel(:))
%     segMask = edlabel == ed;
%     if sum(segMask(:)) < 10
%         continue;
%     end
%     % Get the mask of selected segment.
%     label = label1 .* segMask;
%     % Get the mask of unselected segment.
%     lb = label1 .* (~segMask); 
%     % Get the list of superpixels in unselected segments.
%     sSP = uint16(unique(lb));
%     sSP = sSP(sSP~=0);
%     sPixCount = max(size(sSP));
%     % Get the list of superpixels in selected segments.
%     segmentedSP = uint16(unique(label));
%     segmentedSP = segmentedSP(segmentedSP~=0);
%     for i=1:sPixCount
%         if min(size(find(segmentedSP==sSP(i))))
%             s1 = label == sSP(i);
%             s1 = sum(s1(:));
%             s2 = lb == sSP(i);
%             s2 = sum(s2(:));
%             if s2 > s1 
%                 segmentedSP = segmentedSP(segmentedSP~=sSP(i));
%             end
%         end
%     end
%     segmentedPixCount = max(size(segmentedSP));
%     if segmentedPixCount < 3
%         continue;
%     end
%     segMask = edlabel == ed;
%     [row col] = size(segMask);
%     tmp = false(row+2, col+2);
%     tmp(2:row+1, 2:col+1) = segMask;
%     sm = imerode(tmp, strel('disk', 4));
%     sm = sm(2:end-1, 2:end-1);
%     sm = im2bw(segMask-sm);
%     r(sm)=255;
%     g(sm)=255;
%     b(sm)=0;
% end
% seg = cat(3, r, g, b);
% % inImg = cat(3, r, g, b);
% imwrite(uint8(inImg), fName);
% imwrite(uint8(seg), 'Result/castle_bd.png');
% return;
function ColorizeSegment(label, fName)
inImg = label;
inImg(:) = 0;
r = inImg;
g = inImg;
b = inImg;
for i=0:max(label(:))
    r(label == i) = rand(1) * 255;
    g(label == i) = rand(1) * 255;
    b(label == i) = rand(1) * 255;
end
inImg = cat(3, r, g, b);
imwrite(uint8(inImg), fName);
return;

function sm = UpdateSegMask(ed)
    sm = ed;
    maxCount = max(ed(:));
    mask1 = im2bw(imread('building2_building1.png'));
    mask2 = im2bw(imread('building2_water.png'));
    mask1=imfill(mask1, 'holes');
    ed(mask1) = maxCount + 1;
    ed(mask2) = maxCount + 2;
    smList = unique(ed(:));
    for I=1:length(smList)
        sm(ed == smList(I)) = I-1;
    end
return;
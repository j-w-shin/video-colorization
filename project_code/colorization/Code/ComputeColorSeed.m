function ComputeColorSeed

    patchsize=8;
    gridspacing=1;
    SIFTflowpara.alpha=2;
    SIFTflowpara.d=40;
    SIFTflowpara.gamma=0.005;
    SIFTflowpara.nlevels=4;
    SIFTflowpara.wsize=5;
    SIFTflowpara.topwsize=20;
    SIFTflowpara.nIterations=60;

    %%  Step 1. Get the images names.
%     InputImage = 'G:\Alex\NewExperiment\SF_Colorization\input\test1.png';
%     InputSegmentedImage = 'G:\Alex\NewExperiment\SF_Colorization\input\test1_sm.png';
     InputImage = 'G:\Alex\NewExperiment\SF_Colorization\input\bride.png';
     InputSegmentedImage = 'G:\Alex\NewExperiment\SF_Colorization\input\bride_sm.png';
%     InputImage = 'G:\Alex\NewExperiment\SF_Colorization\input\Mona-Lisa.jpg';
%     InputSegmentedImage = 'G:\Alex\NewExperiment\SF_Colorization\input\Mona-Lisa_sm.jpg';
    % InputImage = 'G:\Alex\NewExperiment\SF_Colorization\input\C_tea.png';
    % InputSegmentedImage = 'G:\Alex\NewExperiment\SF_Colorization\input\c_tea_sm.png';

%     internetColorImgName = 'G:\Alex\NewExperiment\SF_Colorization\input\Mona-Lisa.jpg';
%     internetSegImgName = 'G:\Alex\NewExperiment\SF_Colorization\input\Mona-Lisa_sm.jpg'; 
%     internetColorImgName = 'G:\Alex\NewExperiment\SF_Colorization\input\test1.png';
%     internetSegImgName =   'G:\Alex\NewExperiment\SF_Colorization\input\test1_sm.png';
    internetColorImgName = 'G:\Alex\NewExperiment\SF_Colorization\input\colImg1.png';
    internetSegImgName = 'G:\Alex\NewExperiment\SF_Colorization\input\colImg1_SM.png'; 
%     internetColorImgName = 'G:\Alex\NewExperiment\SF_Colorization\input\tea.png';
%     internetSegImgName = 'G:\Alex\NewExperiment\SF_Colorization\input\tea_sm.png';    
    
    InputImg = im2double(imread(InputImage));
    resizeRatio = 0.6;
    if size(InputImg, 3) == 3
        InputImg = rgb2gray(InputImg);
    end
    InputImg = imresize(InputImg, resizeRatio);
    Im1=InputImg;
    InputSegImg = imread(InputSegmentedImage);  
    InputSegImg = imresize(InputSegImg, resizeRatio);
    InputSegImg = im2bw(InputSegImg);
    InputSegImg = imfill(InputSegImg, 'holes');

    [r c] = find(InputSegImg==1);
    minR = min(r);      maxR = max(r);
    minC = min(c);      maxC = max(c);

    InputImg = InputImg(minR:maxR, minC:maxC,:);
    Im1 = Im1(minR:maxR, minC:maxC,:);
    InputSegImg = InputSegImg(minR:maxR, minC:maxC,:);  

    [ImgRow ImgCol] = size(InputImg);
    inputDiagonal = sqrt(ImgRow*ImgRow + ImgCol*ImgCol);

    Sift1=dense_sift(InputImg,patchsize,gridspacing);

%     [ir, ic, ch] = size(Sift1);
%     s1 = zeros(ir, ic, ch+1);
%     s1(:,:,2:ch+1) = Sift1;
%     s1(:,:,1) = InputImg(4:ImgRow-3, 4:ImgCol-3);

    internetColorImg = im2double(imread(internetColorImgName));
    Im2=internetColorImg;
    internetGrayImg = rgb2gray(internetColorImg);

    internetSegImg = imread(internetSegImgName);
    internetSegImg = im2bw(internetSegImg);
    internetSegImg = imfill(internetSegImg, 'holes');

    [r c] = find(internetSegImg==1);
    minR = min(r);      maxR = max(r);
    minC = min(c);      maxC = max(c);

    internetGrayImg = internetGrayImg(minR:maxR, minC:maxC,:);
    Im2 = Im2(minR:maxR, minC:maxC,:);
    internetSegImg = internetSegImg(minR:maxR, minC:maxC,:);        

    diffR = maxR-minR+1;    diffC = maxC-minC+1;
%     currDiagonal = sqrt(diffR*diffR + diffC*diffC);
    currResizeRatio = [ImgRow ImgCol]%inputDiagonal/currDiagonal;
    Im2 = imresize(Im2, currResizeRatio);
    internetGrayImg = imresize(internetGrayImg, currResizeRatio);
    internetSegImg = imresize(internetSegImg, currResizeRatio);

    RandomImg = rand(size(internetGrayImg,1), size(internetGrayImg,2));

    internetGrayImg(~internetSegImg) = RandomImg(~internetSegImg);
    Sift2=dense_sift(internetGrayImg, patchsize, gridspacing);

    % [ImgRow ImgCol] = size(internetGrayImg);    
    % [ir, ic, ch] = size(Sift2);
    % s2 = zeros(ir, ic, ch+1);
    % s2(:,:,2:ch+1) = Sift2;
    % s2(:,:,1) = internetGrayImg(4:ImgRow-3, 4:ImgCol-3);
    % [vx,vy,energylist]=SIFTflowc2f(s1,s2,SIFTflowpara);
    [vx,vy,energylist]=SIFTflowc2f(Sift1, Sift2, SIFTflowpara);
    [warpI1 mask1 XX1 YY1]=warpImage(Im2, vx, vy);
    
    [vx,vy,energylist]=SIFTflowc2f(Sift2, Sift1, SIFTflowpara);
    [warpI2 mask2 XX2 YY2]=warpImage(Im1, vx, vy);
    
    
    figure;imshow(Im1);title('Image 1');
    figure;imshow(Im2);title('Image 2');
    imwrite(Im1, 'tmp\col.png');
    imwrite(Im2, 'tmp\col_I.png');
    figure;imshow(warpI2);title('Warped image 2');
    imwrite(warpI1, 'tmp\col_w.png');
%     warpI1=warpImage(internetGrayImg,vx,vy);
    imwrite(rgb2gray(warpI1), 'tmp\gray_w.png');
    imwrite(mask1, 'tmp\mask.png');
    imwrite(internetGrayImg, 'tmp\gray.png');
    figure;imshow(InputImg);title('Image 1');
    imwrite(InputImg, 'tmp\grayInput.png');
    figure;imshow(warpI2);title('Warped image 2');
    figure;imshow(internetGrayImg);
    imwrite(InputSegImg, 'tmp\seg.png');

    disp(['Minimum Energy : ' num2str( sum(energylist(1, 4).data(:))) ' found for the image:']);
    disp('Finished ....');
return;


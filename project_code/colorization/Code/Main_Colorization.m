function Main_Colorization(bw_name, refName)
close all;
warning off;
param.spCount = 1400;

keyWord = bw_name;
param.inRatio = 1;
param.colRatio = 1;
param.nnSize = 512;
param.RecomputationRequired = 1;
inpColImg = cell(1, 1);
InputImg = ['~/Documents/DL/video-colorization/project_code/ideepcolor/bw_frames/' keyWord '.jpg'];
inpColImg{1} = ['~/Documents/DL/video-colorization/project_code/colorization/Code/Input/' refName '.jpg'];
Colorization(keyWord, InputImg, inpColImg, param);

return;
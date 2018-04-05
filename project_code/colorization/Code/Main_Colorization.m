function Main_Colorization(bw_name, refName)
close all;
warning off;
param.spCount = 2000;

keyWord = bw_name;
param.inRatio = 1;
param.colRatio = 1;
param.nnSize = 660;
param.RecomputationRequired = 1;
inpColImg = cell(1, 1);
InputImg = ['/home/otter/Classworks/dl/project_code/colorization/Code/Input/' keyWord '.jpg'];
inpColImg{1} = ['/home/otter/Classworks/dl/project_code/colorization/Code/Input/' refName '.jpg'];
Colorization(keyWord, InputImg, inpColImg, param);

return;
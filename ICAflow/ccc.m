original_path = pwd;
script_full_path = mfilename('fullpath'); %file path
[upperPath,~] = fileparts(script_full_path); %gets current folder path
cd (upperPath);
load('..\TetrapodPSF\Sequence2.mat');
cd (original_path);
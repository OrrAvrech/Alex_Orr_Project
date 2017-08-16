function [ original_path ] = cdir( str )
%%% Goto function directory relative to script path

% In order to stay in the current folder -->
% Use cd(original_path) after the function call

original_path = pwd;
script_full_path = mfilename('fullpath'); %file path
[upperPath,~] = fileparts(script_full_path); %gets current folder path
cd (upperPath);

% Goto path = srt
cd(str);


end


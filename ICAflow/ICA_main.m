clc; close all; clear all;
%% Generate Tetrapod for random (x,y) emitter locations

num_planes = 2;

num_emitters_vec = [3, 4]; % num_planes length
% Ordinarily will be computed from Simulate_Emitters

x_cell = cell(1,num_planes);
y_cell = cell(1,num_planes);
for i = 1:num_planes
    x_cell{i} = (rand(num_emitters_vec(i), 1)-0.5)*10e-6;
    y_cell{i} = (rand(num_emitters_vec(i), 1)-0.5)*10e-6;
end

num_sources = sum(num_emitters_vec);
NumFrames = 6;
if NumFrames < num_sources
    NumFrames = num_sources + 1;
end

original_path = pwd;
script_full_path = mfilename('fullpath'); %file path
[upperPath,~] = fileparts(script_full_path); %gets current folder path
cd (upperPath)
cd('..\TetrapodPSF\');

Sequence2 = TetrapodGenerator(x_cell, y_cell, num_planes, NumFrames);

cd (original_path);
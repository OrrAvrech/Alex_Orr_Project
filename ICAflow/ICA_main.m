clc; close all; clear all;
%% Generate Tetrapod for random (x,y) emitter locations

num_planes = 3;

num_emitters_vec = [1, 2, 1]; % num_planes length
% Ordinarily will be computed from Simulate_Emitters

x_cell = cell(1,num_planes);
y_cell = cell(1,num_planes);
for i = 1:num_planes
    x_cell{i} = (rand(num_emitters_vec(i), 1)-0.5)*10e-6;
    y_cell{i} = (rand(num_emitters_vec(i), 1)-0.5)*10e-6;
end

num_sources = sum(num_emitters_vec);
NumFrames = 6;
if NumFrames <= num_sources
    NumFrames = num_sources + 1; %NumFrames minimal value
end

original_path = cdir('..\TetrapodPSF\');
Sequence2 = TetrapodGenerator(x_cell, y_cell, num_planes, NumFrames);
cd (original_path);

%% Run FastICA

IC = Tetrapod_ICA(num_sources, Sequence2.LinearCombinations);

%% Visualize ICA Input and Output

imagesIC_flag = 1;
BlinkMovie_flag = 1;

Visualize_Sources(Sequence2.LinearCombinations, IC, imagesIC_flag, BlinkMovie_flag);

%% Get results error rate
% x_orig = zeros(1,sum(cellfun('length',x_cell))); %preallocate
% y_orig = zeros(1,sum(cellfun('length',x_cell)));
x_orig = [];
y_orig = [];
for i = 1:size(x_cell,2) %%cell to vec
    curr_x = cell2mat(x_cell(i));
    curr_y = cell2mat(y_cell(i)');
    el_num = numel(curr_x);
    x_orig(end+(1:el_num)) =  curr_x;
    y_orig(end+(1:el_num)) =  curr_y;

end


error_rate = ICA_acc( x_orig, y_orig, IC )

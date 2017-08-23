clc; close all; clear all;
%% Generate Tetrapod for random (x,y) emitter locations


num_emitters_vec = [2 0 0 1 0 0 0 1 1]; % num_planes length
zPosidx = find(num_emitters_vec);
z_orig = [];
for jj = 1 : numel(zPosidx)
    z_orig(end+1 : (end + num_emitters_vec(zPosidx(jj)))) = zPosidx(jj)*ones(1,num_emitters_vec(zPosidx(jj))); 
end


num_planes = numel(num_emitters_vec);

% Ordinarily will be computed from Simulate_Emitters

x_cell = cell(1,num_planes);
y_cell = cell(1,num_planes);
for i = 1:num_planes
    x_cell{i} = (rand(num_emitters_vec(i), 1)-0.5)*10e-6;
    y_cell{i} = (rand(num_emitters_vec(i), 1)-0.5)*10e-6;
end

num_sources = sum(num_emitters_vec);
NumFrames = 2;
if NumFrames <= num_sources
    NumFrames = num_sources + 1; %NumFrames minimal value
end

original_path = cdir('..\TetrapodPSF\');
Sequence2 = TetrapodGenerator(x_cell, y_cell, num_planes, NumFrames);

% I = mat2gray(Sequence2.LinearCombinations(:,:,1));
% imwrite(I, 'tetrapod.tif');
%imagesc(Sequence2.LinearCombinations(:,:,1));
% imshow(I);

%sequence = TetrapodGenerator(num2cell(x_orig(1)), num2cell(y_orig(1)), 1, 1)
cd (original_path);

%% Run FastICA

IC = Tetrapod_ICA(num_sources, Sequence2.LinearCombinations);

%% Visualize ICA Input and Output

imagesIC_flag = 0;
BlinkMovie_flag = 0;

Visualize_Sources(Sequence2.LinearCombinations, IC, imagesIC_flag, BlinkMovie_flag);

%% Get results error rate
% x_orig = zeros(1,sum(cellfun('length',x_cell))); %preallocate to save
% memory?
% y_orig = zeros(1,sum(cellfun('length',x_cell)));
x_orig = cell2vec(x_cell);
y_orig = cell2vec(y_cell);
y_orig = [];

% [x_rec, y_rec] = pix_metric(x_orig(1), y_orig(1))
I = mat2gray(IC(:,:,1));
figure(3);imshow(I); hold on; 
% error_rate = ICA_acc( x_orig, y_orig, IC )
% %% Test pix metric
% close all;
% I = mat2gray(IC(:,:,1));
% figure(3);imshow(I); hold on; 
% title('Gray scale Image')
% c = 'mass_cent';
% fh = str2func(c)
% [curr_x,curr_y] = fh(IC(:,:,1));
% %[curr_x,curr_y] = mass_cent(IC(:,:,1));
% [x_test, y_test] = pix_metric(x_orig, y_orig, 0)
% figure(1);plot(x_test,y_test,'ro')
% title('Original Mass Center Estimation')
% figure(3);plot(curr_x,curr_y,'ro')
% title('ICA image Mass Center Estimation')

%% Test error rate
[error, inds] = error_rate(cell2vec(x_cell), cell2vec(y_cell), z_orig, IC, 'euc_dist', num_planes, 1);
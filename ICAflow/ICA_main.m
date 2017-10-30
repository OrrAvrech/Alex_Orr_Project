clc; close all;
%% Generate Tetrapod using simulated emitters object

original_path = cdir('..\DataSimulation\');
m = matfile('DataObjects.mat');
Emitters = m.EmittersObj(1,4); % Example object
NumSources = Emitters.NumSources;

NumFrames = 150;
if NumFrames <= NumSources
    NumFrames = NumSources + 1; %NumFrames minimal value
end

Sequence2 = Data2TetrapodGen(Emitters, NumFrames, 0);
cd (original_path);

%% Run FastICA

IC = Tetrapod_ICA(NumSources, Sequence2.LinearCombinations);

%% Visualize ICA Input and Output
imagesIC_flag = 0;
BlinkMovie_flag = 0;

Visualize_Sources(Sequence2.LinearCombinations, IC, imagesIC_flag, BlinkMovie_flag);
%% Test error rate by Criterion
flag_cmpImg = 1;
Criterion   = 'PoissCost';
[error, inds] = error_rate(Emitters, IC, Criterion, flag_cmpImg);
total_result = sum(error)/NumSources;
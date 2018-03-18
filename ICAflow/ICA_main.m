clc; close all;
%% Simulated Emitters Object %%
% Generate Tetrapod using simulated emitters object
original_path = cdir('..\DataSimulation\');
m = matfile('DataObjects.mat');
Emitters = m.EmittersObj(1,4); % Example object
NumSources = Emitters.NumSources;

NumFrames = 20;
if NumFrames <= NumSources
    NumFrames = NumSources + 1; %NumFrames minimal value
end

Sequence2 = Data2TetrapodGen(Emitters, NumFrames, 0);
cd (original_path);

% Run FastICA

IC = Tetrapod_ICA(NumSources, Sequence2.LinearCombinations);

% Visualize ICA Input and Output
imagesIC_flag = 0;
BlinkMovie_flag = 0;

Visualize_Sources(Sequence2.LinearCombinations, IC, imagesIC_flag, BlinkMovie_flag);
% Test error rate by Criterion
flag_cmpImg = 1;
Criterion   = 'PoissCost';
[error, inds] = error_rate(Emitters, IC, Criterion, flag_cmpImg);
total_result = sum(error)/NumSources;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Random Emitters %%

Emitters = load('..\DataSimulation\Dataset3\3.mat');
LinComb = Emitters.features;

% Run FastICA
MaxSources = 4;
IC = Tetrapod_ICA(MaxSources, LinComb); % IC is normalized (mean-std)

% Visualize ICA Input and Output
imagesIC_flag = 1;
BlinkMovie_flag = 0;

Visualize_Sources(LinComb, IC, imagesIC_flag, BlinkMovie_flag);
%% Test error rate by Criterion
flag_cmpImg = 1;
Criterion   = 'ImgXcorr';
[corr, inds] = error_rateRnd(Emitters, IC, Criterion, flag_cmpImg);
NumSourcesEst = length(inds);
SampleAvgCorr = sum(corr)/NumSourcesEst;
%% Average Correlation Vs. Max Sources
tic

flag_cmpImg = 0;
Criterion   = 'ImgXcorr';

MaxSources = 2:10;
NumFrames = 20;
NumSamples = 100;
SampleAvgCorr = zeros(NumSamples, 1);
SourceAvgCorr = zeros(numel(MaxSources), 1);
for ii = 1 : numel(MaxSources)
    for jj = 1 : NumSamples
    s1 = strcat('..\DataSimulation\DataSetMid\DatasetS\Dataset',...
        num2str(MaxSources(ii)), '\');
    s2 = strcat(num2str(jj), '.mat');
    Emitters = load(strcat(s1, s2));
    LinComb = Emitters.features;
    IC = Tetrapod_ICA(MaxSources(ii), LinComb);
    [corr, inds] = error_rateRnd(Emitters, IC, Criterion, flag_cmpImg);
    NumSourcesEst = length(inds);
    SampleAvgCorr(jj) = sum(corr)/NumSourcesEst; 
    end
    SourceAvgCorr(ii) = mean(SampleAvgCorr);
end
toc

figure;
plot(MaxSources, SourceAvgCorr);
title(['Average NCC Vs. Max Sources for NumFrames = ', num2str(NumFrames)]);
xlabel('Max Sources');
ylabel('Average NCC');
%% Average Correlation Vs. Num Frames
tic

flag_cmpImg = 0;
Criterion   = 'ImgXcorr';

MaxSources = 5;
NumFrames = 5:20;
NumSamples = 100;
SampleAvgCorr = zeros(NumSamples, 1);
FrameAvgCorr = zeros(numel(NumFrames), 1);
for ii = 1 : numel(NumFrames)
    for jj = 1 : NumSamples
    s1 = strcat('..\DataSimulation\DataSetMid\DatasetF\Dataset',...
        num2str(NumFrames(ii)), '\');
    s2 = strcat(num2str(jj), '.mat');
    Emitters = load(strcat(s1, s2));
    LinComb = Emitters.features;
    IC = Tetrapod_ICA(MaxSources, LinComb);
    [corr, inds] = error_rateRnd(Emitters, IC, Criterion, flag_cmpImg);
    NumSourcesEst = length(inds);
    SampleAvgCorr(jj) = sum(corr)/NumSourcesEst; 
    end
    FrameAvgCorr(ii) = mean(SampleAvgCorr);
end
toc

figure;
plot(NumFrames, FrameAvgCorr);
title(['Average NCC Vs. Num Frames for MaxSources = ', num2str(MaxSources)]);
xlabel('Num Frames');
ylabel('Average NCC');

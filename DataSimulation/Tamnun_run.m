close all; clc;
NumSamples = 2;
NumFrames  = 20;
MaxSources = 5;
Tamnun_gen(NumSamples, NumFrames, MaxSources, 'Dataset3');
%% MaxSources per specified NumFrames
NumSamples = 100;
NumFrames  = 20;
MaxSources = 2:15;
for ii = 1 : numel(MaxSources)
    Tamnun_gen(NumSamples, NumFrames, MaxSources(ii),... 
    strcat('DataSetMid\DatasetS\Dataset', num2str(MaxSources(ii))));
end
%% NumFrames per specified MaxSources 
NumSamples = 100;
MaxSources = 5;
NumFrames = 5:20;
for jj = 1 : numel(NumFrames)
  Tamnun_gen(NumSamples, NumFrames(jj), MaxSources,...
      strcat('DataSetMid\DatasetF\Dataset', num2str(NumFrames(jj))));
end  
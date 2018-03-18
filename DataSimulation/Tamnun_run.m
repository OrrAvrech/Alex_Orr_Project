close all; clc;
NumSamples = 4995;
NumFrames  = 8;
MaxSources = 2;
Tamnun_gen(NumSamples, NumFrames, MaxSources, 'Dataset_im64_f8_s2');
%% MaxSources per specified NumFrames
NumSamples = 100;
NumFrames  = 16;
MaxSources = 2:8;
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
%% Obtain Random Emitters

NumZplanes = 80;
MaxSources = 5;
zVec = linspace(-2e-6,2e-6,NumZplanes+1);
NumSources = randi(MaxSources);
x = (rand(1,NumSources)-0.5)*1e-7;
y = (rand(1,NumSources)-0.5)*1e-7;
z = (randi(NumZplanes,[1,NumSources]));

Objects = load('DataObjects.mat');
flag_save2DS = 1;

original_path = cdir('..\DataSimulation\');
for ii = 1 : numel(Objects.EmittersObj)
    Emitters = Objects.EmittersObj(1,ii);
    NumSources = Emitters.NumSources;

    NumFrames = randi(15);
    if NumFrames <= NumSources
        NumFrames = NumSources + 1; %NumFrames minimal value
    end

    Sequence2 = Data2TetrapodGen(Emitters, NumFrames, flag_save2DS);
end
cd (original_path);
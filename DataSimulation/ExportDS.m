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
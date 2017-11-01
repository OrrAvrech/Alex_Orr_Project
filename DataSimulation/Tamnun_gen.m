function finished = Tamnun_gen(samp_num, save_to)
    %inputs
        %save_to string in '' not ""
    %choose:
    %samp_num = 100; %temp
    
    NumZplanes = 80;
    MaxSources = 5;
    x_center = 0;
    y_center = 0;
    x_width = 2e-7;
    y_width = 2e-7;
    
    %helpers:
    zVec = linspace(-2e-6,2e-6,NumZplanes+1);
    %name_ind:
    file_naming_ind = 1;
    cdir(['..\',save_to,'\']); %change to saving destination
    listing = dir; % dir(name) returns file list containg name sub string
    listing = {listing.name};
    listing = sort(result);
    last_file = listing{end};
    %last_file = '190.mat' %test
    file_naming_ind = str2num(last_file(1:end-4)) %remove '.mat'
    
    
    for i=1:samp_num
        file_ind = file_naming_ind + i;
        %stochastic:
        NumSources = randi(MaxSources);
        %actual data:
        x = (rand(1,NumSources)+x_center-0.5)*x_width/2;
        y = (rand(1,NumSources)+y_center-0.5)*y_width/2;
        z = (randi(NumZplanes,[1,NumSources]));
        
        EmittersObj.x           = x;
        EmittersObj.y           = y;
        EmittersObj.zVec        = z;
        EmittersObj.NumSources  = NumSources; 

        if flag_Save2File == 1  %%remove condition
            m = matfile(['DataObjects',num2str(file_ind),'.mat'],'Writable',true);
            try
                m.EmittersObj = [m.EmittersObj, EmittersObj];
            catch
                save('DataObjects.mat', 'EmittersObj', '-v7.3');
            end

            whos('-file','DataObjects.mat')
        end
        original_path = cdir('..\DataSimulation\');
        Emitters = Objects.EmittersObj(1,ii);
        NumSources = Emitters.NumSources;

        NumFrames = randi(15);
        if NumFrames <= NumSources
            NumFrames = NumSources + 1; %NumFrames minimal value
        end

        Sequence2 = Data2TetrapodGen(Emitters, NumFrames, flag_save2DS);
    cd (original_path);
       
    end
    
    return 1
end
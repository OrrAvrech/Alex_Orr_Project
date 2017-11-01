function finished = Tamnun_gen(samp_num, save_to_dir)
        %%inputs
        %save_to string in '' not ""
        
        %%choose:
    NumZplanes = 80;
    MaxSources = 5;
    x_center = 0;
    y_center = 0;
    x_width = 2e-7;
    y_width = 2e-7;
    
        %%helpers:
        
    zVec = linspace(-2e-6,2e-6,NumZplanes+1);
        
        %%file handling:
        
    file_naming_ind = 0;
            %%make sure the folder exists
    orig_listing = dir; % dir(name) returns file list containg name sub string
    orig_listing = {orig_listing.name};
    exists = find(ismember(orig_listing,save_to_dir));
    if ~numel(exists)
        mkdir(save_to_dir);
    end
    
    cdir([save_to_dir,'\']); %change to saving destination
    
    dest_listing = dir
    if ~dest_listing(end).isdir %is empty?
        dest_names = ({dest_listing.name});
        last_file = dest_names{end};
        file_naming_ind = str2num(last_file(1:end-4)) %remove '.mat'
    end
    
    
    for i=1:samp_num
        file_ind = file_naming_ind + i;
            %%stochastic:
        NumSources = randi(MaxSources);
            %%actual data:
        x = (rand(1,NumSources)+x_center-0.5)*x_width/2;
        y = (rand(1,NumSources)+y_center-0.5)*y_width/2;
        z = (randi(NumZplanes,[1,NumSources]));
        
        EmittersObj.x           = x;
        EmittersObj.y           = y;
        EmittersObj.zVec        = z;
        EmittersObj.NumSources  = NumSources; 

%         if flag_Save2File == 1  %%remove condition
        m = matfile([num2str(file_ind),'.mat'],'Writable',true);
        try
            m.EmittersObj = [m.EmittersObj, EmittersObj];
        catch
            save([num2str(file_ind),'.mat'], 'EmittersObj', '-v7.3');
        end

        whos('-file',[num2str(file_ind),'.mat'])
%         end
        
        original_path = cd('..\..\DataSimulation\'); %%cdir doesn't work not sure why
        Emitters = Objects.EmittersObj(1,ii);
        NumSources = Emitters.NumSources;

        NumFrames = randi(15);
        if NumFrames <= NumSources
            NumFrames = NumSources + 1; %NumFrames minimal value
        end

        Sequence2 = Data2TetrapodGen(Emitters, NumFrames, flag_save2DS);
    cd (original_path);
       
    end
    
    finished = 1
end
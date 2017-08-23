function [ Error_vec, Ind_vec ] = error_rate(x_orig, y_orig, z_orig, IC, Criterion, num_planes, Display)
    %get error per emitter and its matching index
    Ind_vec = [];
    Error_vec = [];
    for ii = 1:size(IC,3)
        [Error, Ind] = match_by_crit(x_orig,y_orig,IC(:,:,ii),Criterion, Ind_vec);
%         x_orig(ind) = [] ; %delete element
%         y_orig(ind) = [] ; %delete element
        Ind_vec(end+1) = Ind;
        Error_vec(end+1) = Error;
    end


%% img display
if Display == 1
    for ii = 1:size(IC,3)
        figure(ii); subplot 121 ; imagesc (IC(:,:,ii));
%         get_z(IC(:,:,ii))
        x_cell = cell(1,num_planes);
        y_cell = cell(1,num_planes);
        x_cell{z_orig(Ind_vec(ii))} = x_orig(Ind_vec(ii)); %% put x_orig in the index that corresponds to the right depth
        y_cell{z_orig(Ind_vec(ii))} = y_orig(Ind_vec(ii)); 
        original_path = cdir('..\TetrapodPSF\');
        Sequence = TetrapodGenerator(x_cell, y_cell, num_planes, 1);
        cd (original_path);
        figure(ii); subplot 122 ; imagesc(Sequence.LinearCombinations(:,:,1));
    end
end

end


function [ Error_vec, Ind_vec ] = error_rate(z_xy_orig, IC, Criterion, num_planes, Display)
%% Separate ICA results by plane index
IC_z = cell(1,num_planes);
IC_z_inds = [];
for ii = 1:size(IC,3) %put IC results in the z plane they fit most
    curr_z = fit_to_plane(IC(:,:,ii));
    if isempty(IC_z{curr_z}) next_row_ind = 1;
    else next_row_ind = size(IC_z{curr_z}(1,1,:),3)+1; 
    end
    IC_z{curr_z}(1:numel(IC(:,1,1)),1:numel(IC(1,:,1)),next_row_ind) = IC(:,:,ii); 
    IC_z_inds(end+1) = curr_z;
end



%% Get error per emitter and its matching index
    Ind_vec = [];
    Error_vec = [];
    for jj = 1:size(IC_z,2)
        temp_Ind_vec = [];
        for ii = 1:size(IC_z{jj},3)
            [Error, Ind] = match_by_crit(z_xy_orig{jj}(:,1),z_xy_orig{jj}(:,2),IC_z{jj}(:,:,ii),Criterion, Ind_vec);
            temp_Ind_vec(end+1) = Ind; 
            Error_vec(end+1) = Error;
        end
        Ind_vec((end+1):(end+numel(temp_Ind_vec))) = temp_Ind_vec;
    end

%% img display
if Display == 1
    kk=1;
    for jj = 1:num_planes
        for ii = 1:size(IC_z{jj},3)
            figure(ii); subplot 121 ; imagesc (IC_z{jj}(:,:,ii));
%             get_z(IC(:,:,ii))
            x_cell = cell(1,num_planes);
            y_cell = cell(1,num_planes);
            x_cell{jj} = z_xy_orig{jj}(Ind_vec(kk),1); %% put x_orig in the index that corresponds to the right depth
            y_cell{jj} = z_xy_orig{jj}(Ind_vec(kk),2);
            
            original_path = cdir('..\TetrapodPSF\');
            Sequence = TetrapodGenerator(x_cell,y_cell, num_planes, 1);
            cd (original_path);
            figure(ii); subplot 122 ; imagesc(Sequence.LinearCombinations(:,:,1));
            kk = kk+1;
        end
    end
end

end


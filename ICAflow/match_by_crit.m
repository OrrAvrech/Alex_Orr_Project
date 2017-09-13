function [ error, ind ] = match_by_crit( SingleImg, IC, Criterion, used_inds )

%match given Independent Component (IC) to its XY estimated origin, and
%calculate error

error = inf;
crit_error = str2func (Criterion);
crit_list = {'euc_dist', 'ImgXcorr', 'ImgL2'};
enable = sum(strcmp(crit_list,Criterion)); %% checks if related to xy list

if enable 
    im = SingleImg; % Single Tetrapod Image 
    it_num = size(IC,3); 
end 

for i = 1:it_num
    if any(i == used_inds)
        continue;
    end
    if enable 
        temp_error = crit_error(im, IC(:,:,i));
    else
        disp('no Criterion found')
    end
    if error > temp_error
        error = temp_error;
        ind = i;
    end
end

end




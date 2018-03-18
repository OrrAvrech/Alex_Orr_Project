function [ error, ind ] = match_by_crit( SingleImg, IC, Criterion, used_inds )

%match given Independent Component (IC) to its XY estimated origin, and
%calculate error

error = 0;
flag = 0;
crit_error = str2func (Criterion);
crit_list = {'euc_dist', 'ImgXcorr', 'PoissCost'};
enable = sum(strcmp(crit_list,Criterion)); %% checks if related to xy list

if enable 
    im = SingleImg; % Single Tetrapod Image 
    % MinMax Normalization of Original Sources
    %im = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    %im = im - mean(im(:));
    
    
    % Go over all IC's diff from already used indices
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
    if error < temp_error
        flag = flag + 1;
        error = temp_error;
        ind = i;
    end
end

if (flag == 0)
    ind = 0;
end

end




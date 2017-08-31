function [ error, ind ] = match_by_crit( input1, input2, IC, Criterion,used_inds )
%match given Independent Component (IC) to its XY estimated origin, and
%calculate error
error = inf;
crit_error = str2func (Criterion);
xy_crit_list = {'euc_dist', 'anotherOne'};
im_crit_list = {'L2', 'anotherOne'};
ena_xy = sum(strcmp(xy_crit_list,Criterion)); %% checks if related to xy list
ena_im = sum(strcmp(im_crit_list,Criterion));
if ena_xy X_orig = input1; Y_orig = input2; it_num = length(X_orig); end %rename inputs
if ena_im im = input1;  it_num = size(X_orig,3); end %rename inputs

for i = 1:it_num
    if any(i == used_inds)
        continue;
    end
    if ena_xy 
        temp_error = crit_error(X_orig(i),Y_orig(i),IC);
    elseif ena_im
        temp_error = crit_error(im,IC);
    else
        disp("no Criterion found")
    end
    if error > temp_error
        error = temp_error;
        ind = i;
    end
end

end


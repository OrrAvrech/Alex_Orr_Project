function [ error, ind ] = match_by_crit( X_orig, Y_orig, IC, Criterion,used_inds )
%match given Independent Component (IC) to its XY estimated origin, and
%calculate error
error = inf;
crit_error = str2func (Criterion);
for i = 1:length(X_orig)
    if any(i == used_inds)
        continue;
    end
    temp_error = crit_error(X_orig(i),Y_orig(i),IC);
    if error > temp_error
        error = temp_error;
        ind = i;
    end
end

end


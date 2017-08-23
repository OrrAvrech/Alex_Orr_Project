function [ x_vec] = cell2vec( x_cell )
% allign all cell elements in one vector
%   Detailed explanation goes here
x_vec = [];
for i = 1:size(x_cell,2) %%cell to vec
    curr_x = cell2mat(x_cell(i));
    el_num = numel(curr_x);
    x_vec(end+(1:el_num)) =  curr_x;
end


end


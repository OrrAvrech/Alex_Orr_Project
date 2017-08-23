function [ x,y ] = mass_cent( Im)
    I = mat2gray(Im(:,:));
    Ibw = im2bw(I);
    if sum(sum(I)) > 200*200/2 
        Ibw = imcomplement(Ibw); % negative
    end
    %Ilabel = bwlabel(Ibw);
    Ilabel = logical(Ibw); %matlab suggestion to save memory
    stat = regionprops(Ilabel,'centroid');
    curr_x = 0;
    curr_y = 0;
    num_obj = numel(stat);
    for iii = 1:num_obj
        curr_x = curr_x + (stat(iii).Centroid(1));%/num_obj;
        curr_y = curr_y + (stat(iii).Centroid(2));%/num_obj;
    end
    x = curr_x/num_obj; 
    y = curr_y/num_obj;
end
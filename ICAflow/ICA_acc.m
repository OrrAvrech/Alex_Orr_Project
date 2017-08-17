function [ error ] = ICA_acc( x_orig, y_orig, ica_res )
    num_of_emitters = size(ica_res,3);
    N =25^2 ; %normalization: area of one sample
    temp_err = 0;
    best_ind = [];
    for i = 1:num_of_emitters
        I = mat2gray(ica_res(:,:,i));
        Ibw = im2bw(I);
        if sum(sum(I)) > 200*200/2 
            sum(sum(I))
            200*200/2 
            Ibw = imcomplement(Ibw); % negative
        end
        Ilabel = bwlabel(Ibw);
        %Ilabel = logical(Ibw_n); %matlab suggestion to save memory
        stat = regionprops(Ilabel,'centroid');
        imshow(I); hold on; 
        curr_x = 0;
        curr_y = 0;
        num_obj = numel(stat);
        for iii = 1:num_obj
            curr_x = curr_x + (stat(iii).Centroid(1))/num_obj;
            curr_y = curr_y + (stat(iii).Centroid(2))/num_obj;
        end
        curr_error = 1;
        plot(curr_x,curr_y,'ro');drawnow; %debug
        for ii = 1:numel(x_orig)
            it_dist = sqrt((curr_x - x_orig(ii)).^2+(curr_y - y_orig(ii)).^2);
            it_err = it_dist/N
            if it_err < curr_error
                curr_error = it_err;
                best_ind = ii; %%may be useful for debug and more
            end
        end
        x_orig(best_ind) = [] ; %delete element
        y_orig(best_ind) = [] ; %delete element
        temp_err = temp_err + curr_error;
        
    end
    error = temp_err/num_of_emitters;
end

